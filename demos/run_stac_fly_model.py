# FLY_MODEL: so far used only by the fly, awaiting explanation from Elliot

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Use GPU 1
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# Note: jax_persistent_cache_enable_xla_caches may not be available in all JAX versions
try:
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
except AttributeError:
    pass  # Skip if not available in this JAX version
import stac_mjx
from pathlib import Path
from jax import numpy as jp
import mediapy as media
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from stac_mjx import io

OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def parse_hydra_config(cfg: DictConfig):

    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.stac.gpu  # Use GPU 1
    # Choose parent directory as base path to make relative pathing easier
    stac_cfg = cfg.stac
    model_cfg = cfg.model
    save_path = Path(cfg.stac.save_path)
    # save_path = Path('/data/users/eabe/biomech_model/Flybody/datasets/Johnsonlab/')
    # base_path = Path.cwd().parent
    base_path = Path('/home/eabe/Research/MyRepos/Fly_tracking/assets/fruitfly_v1')
    import stac_mjx.io_dict_to_hdf5 as ioh5

    # ##### Full fly #####
    data_path = stac_cfg.data_path
    # data_path = base_path / stac_cfg.data_path
    bout_dict = ioh5.load(data_path)
    legs_data = ['L1', 'R1', 'L2','R2', 'L3','R3']
    joints_data = ['A','B','C','D','E']
    sorted_kp_names = [leg + joint for leg in legs_data for joint in joints_data]
    xpos_all = []

    # First pass: collect all clips and find max length
    clips = []
    max_length = 0
    for nbout, key in enumerate(bout_dict.keys()):
        bout_data = bout_dict[key]['clipped_kp']
        bout_data = bout_data - bout_data[0:1,0:1,:]  # Center to first frame
        # bout_data = bout_dict[key]['aligned_xpos']
        # bout_data[:,4::5,-1] = jp.clip(bout_data[:,4::5,-1], -0.125)
        bout_data = bout_data.reshape(bout_data.shape[0],-1)
        clips.append(bout_data)
        max_length = max(max_length, bout_data.shape[0])

    # Check if padding is enabled (add this config option)
    enable_padding = getattr(cfg.stac, 'enable_padding', False)

    if enable_padding:
        # Update cfg with max clip length only if padding is enabled
        print(f"Before update - cfg.stac.n_frames_per_clip: {cfg.stac.n_frames_per_clip}")
        print(f"Max length found: {max_length}")
        cfg.stac.n_frames_per_clip = max_length
        print(f"After update - cfg.stac.n_frames_per_clip: {cfg.stac.n_frames_per_clip}")

        # Second pass: pad clips to max length
        for bout_data in clips:
            if bout_data.shape[0] < max_length:
                # Pad with the last valid entry
                last_frame = bout_data[-1:, :]  # Keep last frame with same shape
                padding_needed = max_length - bout_data.shape[0]
                padding = jp.repeat(last_frame, padding_needed, axis=0)
                bout_data = jp.concatenate([bout_data, padding], axis=0)
            xpos_all.append(bout_data)
    else:
        # No padding: just concatenate clips as-is
        print(f"Padding disabled - using clips with original lengths")
        print(f"Clip lengths: {[clip.shape[0] for clip in clips]}")
        for bout_data in clips:
            xpos_all.append(bout_data)

    kp_data = jp.concatenate(xpos_all, axis=0)
    kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']
    print(f"kp_data shape: {kp_data.shape}")
    print(f"Updated n_frames_per_clip to: {cfg.stac.n_frames_per_clip}")
    
    
    # ###### Johnson Lab Fly #####
    # data_dict = ioh5.load(cfg.stac.data_path)
    # kp_data = jp.array(data_dict['aligned_keypoints'].reshape(data_dict['aligned_keypoints'].shape[0],-1))
    # kp_data = kp_data * model_cfg['MOCAP_SCALE_FACTOR']
    # sorted_kp_names = data_dict['kp_names']
    # print(f"kp_data shape: {kp_data.shape}")
    # cfg.stac.n_frames_per_clip = kp_data.shape[0]
    # print(f"Updated n_frames_per_clip to: {cfg.stac.n_frames_per_clip}")

    fit_path, transform_path = stac_mjx.run_stac(
        cfg, kp_data, sorted_kp_names, base_path=base_path, save_path=save_path
    )

    # set args
    data_path = save_path / cfg.stac["ik_only_path"]
    n_frames = 500
    video_dir = Path.cwd().parent / f"videos/{cfg.model.name}.mp4"

    # Call mujoco_viz
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

    # Show the video in the notebook (it is also saved to the save_path)
    # media.show_video(frames, fps=model_cfg["RENDER_FPS"])

    print("Done!")


if __name__ == "__main__":
    parse_hydra_config()
