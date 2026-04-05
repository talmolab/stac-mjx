# FLY_MODEL: so far used only by the fly, awaiting explanation from Elliot

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Use GPU 1
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Note: jax_persistent_cache_enable_xla_caches may not be available in all JAX versions
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass  # Skip if not available in this JAX version
from pathlib import Path
from jax import numpy as jp
import mediapy as media
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

import stac_mjx
import stac_mjx.io_dict_to_hdf5 as ioh5
from stac_mjx.path_utils import convert_dict_to_path, register_custom_resolvers
register_custom_resolvers()

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def parse_hydra_config(cfg: DictConfig):
    """
    Run STAC IK pipeline on preprocessed keypoint data.
    
    Usage examples:
        # Process free_walking data with default version
        python run_stac_fly_model.py paths=workstation dataset=free_walking
        
        # Process with specific version
        python run_stac_fly_model.py paths=workstation dataset=free_walking version=Predictions_3D_20260203-103416
        
        # Multirun across versions
        python run_stac_fly_model.py -m paths=workstation dataset=free_walking version=Predictions_3D_20260114-145343,Predictions_3D_20260202-171900
        
        # Process courtship data
        python run_stac_fly_model.py paths=workstation dataset=courtship
        
        # Use V2 model
        python run_stac_fly_model.py paths=workstation dataset=free_walking anatomy=v2 stac=stac_fly_free_v2
    """
    
    # Print configuration
    print("=" * 80)
    print("STAC IK PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Convert path strings to Path objects and create directories
    cfg.paths = convert_dict_to_path(cfg.paths)
    
    # Get resolved paths
    stac_cfg = cfg.stac
    model_cfg = cfg.model
    data_path = Path(stac_cfg.data_path)
    save_path = Path(stac_cfg.save_path)
    base_path = Path(stac_cfg.xml_dir)
    
    print("Resolved paths:")
    print(f"  Data: {data_path}")
    print(f"  Save: {save_path}")
    print(f"  XML dir: {base_path}")
    print()

    
    # Load keypoint data
    print(f"Loading keypoint data from: {data_path}")
    bout_dict = ioh5.load(data_path)
    
    # Get keypoint names from data
    if 'kp_names' in bout_dict:
        # Single bout format
        sorted_kp_names = bout_dict['kp_names']
        kp_data = bout_dict['keypoints'].reshape(bout_dict['keypoints'].shape[0], -1)
        kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']
        print(f"Loaded single bout: {kp_data.shape[0]} frames, {len(sorted_kp_names)} keypoints")
        
    elif any(k.startswith('bout_') for k in bout_dict.keys()):
        # Multi-bout format - concatenate all bouts
        # Filter out non-bout keys (like 'info')
        bout_keys = [k for k in bout_dict.keys() if k.startswith('bout_')]
        print(f"Detected multi-bout format with {len(bout_keys)} bouts")
        
        # Get keypoint names from first bout
        first_key = bout_keys[0]
        sorted_kp_names = bout_dict[first_key]['kp_names']
        
        # Collect all clips
        clips = []
        max_length = 0
        for nbout, key in enumerate(sorted(bout_keys)):
            bout_data = bout_dict[key]['keypoints']
            bout_data = bout_data.reshape(bout_data.shape[0], -1)
            clips.append(bout_data)
            max_length = max(max_length, bout_data.shape[0])
            print(f"  {key}: {bout_data.shape[0]} frames")
        
        # Handle padding if enabled
        if stac_cfg.enable_padding:
            print(f"\nPadding enabled: padding all clips to {max_length} frames")
            cfg.stac.n_frames_per_clip = max_length
            
            padded_clips = []
            for bout_data in clips:
                if bout_data.shape[0] < max_length:
                    # Pad with the last valid frame
                    last_frame = bout_data[-1:, :]
                    padding_needed = max_length - bout_data.shape[0]
                    padding = jp.repeat(last_frame, padding_needed, axis=0)
                    bout_data = jp.concatenate([bout_data, padding], axis=0)
                padded_clips.append(bout_data)
            clips = padded_clips
        
        # Concatenate all bouts
        kp_data = jp.concatenate(clips, axis=0)
        kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']
        print(f"\nConcatenated data: {kp_data.shape[0]} total frames")
        
    else:
        raise ValueError("Unknown data format: expected 'kp_names' (single bout) or 'bout_XXX' keys (multi-bout)")
    
    print(f"Keypoint data shape: {kp_data.shape}")
    print(f"Keypoints: {sorted_kp_names[:5]}...")
    print(f"Frames per clip: {cfg.stac.n_frames_per_clip}")
    print()

    # Run STAC IK solver
    print("="* 80)
    print("RUNNING STAC IK SOLVER")
    print("=" * 80)
    fit_path, transform_path = stac_mjx.run_stac(
        cfg, kp_data, sorted_kp_names, base_path=base_path, save_path=save_path
    )
    print(f"\n✓ STAC IK complete!")
    print(f"  Fit output: {fit_path}")
    print(f"  Transform output: {transform_path}")

    # Render visualization video
    print("\n" + "=" * 80)
    print("RENDERING VISUALIZATION")
    print("=" * 80)
    data_path = save_path / cfg.stac["ik_only_path"]
    n_frames = min(1000, kp_data.shape[0])  # Render up to 1000 frames
    video_dir = data_path.parent / f"{cfg.dataset.name}_{cfg.anatomy.name}.mp4"

    frames = stac_mjx.viz_stac(
        data_path,
        cfg,
        n_frames,
        video_dir,
        start_frame=0,
        camera='track1',
        height=544,
        width=832,
        base_path=base_path,
    )

    print(f"✓ Video saved to: {video_dir}")
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    parse_hydra_config()
