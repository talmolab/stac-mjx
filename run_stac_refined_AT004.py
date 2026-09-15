#!/usr/bin/env python
import os

# Force GPU mode before any JAX imports
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["MUJOCO_GL"] = "osmesa"

from glob import glob
import h5py
import numpy as np
import logging
import gc
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Import STAC module
import stac_mjx
import jax


# Monkey-patch the enable_xla_flags function to handle JAX API changes
def patched_enable_xla_flags():
    """Enable XLA flags for GPU if available."""
    try:
        import jax.extend

        backend = jax.extend.backend.get_backend()
        platform = backend.platform
    except (ImportError, AttributeError):
        try:
            from jax.lib import xla_bridge

            backend = xla_bridge.get_backend()
            platform = backend.platform
        except AttributeError:
            platform = "cpu"

    if platform == "gpu":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# Replace the function in stac_mjx.utils
stac_mjx.utils.enable_xla_flags = patched_enable_xla_flags

logging.info(f"Using JAX devices: {jax.devices()}")

# ---------------------- Helper Functions ------------------------------


def get_at004_clip_files(clips_dir):
    """
    Get all AT004 clip files from the directory.

    Args:
        clips_dir (str): Directory containing clip H5 files

    Returns:
        list: Sorted list of clip file paths
    """
    clips_path = Path(clips_dir)
    clip_files = sorted(clips_path.glob("AT004_2023-10-26_13-01-42_trial*_clip.h5"))
    logging.info(f"Found {len(clip_files)} clip files")
    return clip_files


def extract_trial_number(clip_path):
    """
    Extract trial number from clip filename.

    Args:
        clip_path (Path): Path to clip file

    Returns:
        int: Trial number
    """
    # Format: AT004_2023-10-26_13-01-42_trial{XX}_clip.h5
    trial_str = clip_path.stem.split("trial")[1].split("_")[0]
    return int(trial_str)


def load_clip_data(clip_path):
    """
    Load clip data from H5 file.

    Args:
        clip_path (Path): Path to clip H5 file

    Returns:
        np.ndarray: Clip data with shape (frames, 1, nodes, 3)
    """
    with h5py.File(clip_path, "r") as f:
        clip_data = f["tracks"][:]
    logging.info(f"Loaded clip with shape {clip_data.shape}")
    return clip_data


def ensure_config_structure(cfg, base_path):
    """
    Ensure the configuration has the proper structure for STAC.

    Args:
        cfg (DictConfig): Original configuration
        base_path (str): Base path for configuration files

    Returns:
        DictConfig: Updated configuration with proper structure
    """
    # Make the configuration mutable
    OmegaConf.set_struct(cfg, False)

    # Create empty structures if they don't exist
    if not hasattr(cfg, "stac"):
        cfg.stac = OmegaConf.create({})

    # Make sure we have a model section
    if not hasattr(cfg, "model"):
        model_name = "mouse_arm"
        model_config_path = os.path.join(base_path, f"configs/model/{model_name}.yaml")
        if os.path.exists(model_config_path):
            model_config = OmegaConf.load(model_config_path)
            cfg.model = model_config
            logging.info(f"Loaded model config from {model_name}")
        else:
            logging.warning(f"Could not find model config at {model_config_path}")

    # Ensure stac has required fields
    if not hasattr(cfg.stac, "data_path"):
        cfg.stac.data_path = ""

    if not hasattr(cfg.stac, "n_frames_per_clip"):
        cfg.stac.n_frames_per_clip = 60

    return cfg


# ---------------------- Main STAC Pipeline ------------------------------


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm", version_base=None)
def main(cfg: DictConfig):
    base_path = os.path.abspath("/mnt/vast/niyo/stac-mjx")
    logging.info(f"Base path: {base_path}")

    # Set up paths for AT004
    clips_dir = "/mnt/vast/eric/CVAT_full_dataset/reach_clips_AT004"
    output_dir = os.path.join(clips_dir, "stac_output")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output files will be saved to: {output_dir}")

    # Data paths for AT004
    points3d_path = "/mnt/vast/eric/CVAT_full_dataset/triangulate_AT004_2023-10-26_13-01-42_umeyama_aligned/points3d_aligned.h5"
    reach_metrics_path = "/mnt/vast/eric/azim_matlab_code/AT004_reach_metrics.csv"

    logging.info(f"Points3D data: {points3d_path}")
    logging.info(f"Reach metrics: {reach_metrics_path}")

    # Ensure proper configuration structure
    cfg = ensure_config_structure(cfg, base_path)

    # Update model configuration
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.update(cfg.model, "SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "TRUNK_OPTIMIZATION_KEYPOINTS", ["Wrist"], merge=True)
    OmegaConf.update(cfg.model, "N_ITERS", 6, merge=True)
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", 50, merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)

    # Get all clip files
    clip_files = get_at004_clip_files(clips_dir)

    if not clip_files:
        logging.error(f"No clip files found in {clips_dir}")
        return

    # Process each clip
    for clip_path in clip_files:
        trial_num = extract_trial_number(clip_path)
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing AT004 trial {trial_num:02d}: {clip_path.name}")
        logging.info(f"{'='*60}")

        try:
            # Load clip data
            clip_data = load_clip_data(clip_path)

            # Verify clip data shape
            if clip_data.shape[0] != 60:
                logging.warning(f"Clip has {clip_data.shape[0]} frames instead of 60")

            # Update stac configuration
            OmegaConf.set_struct(cfg.stac, False)
            cfg.stac.data_path = str(clip_path)

            # Update n_frames_per_clip to match the clip length
            clip_length = clip_data.shape[0]
            OmegaConf.update(cfg.stac, "n_frames_per_clip", clip_length, merge=True)

            # Load mocap data
            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, Path(base_path))
            logging.info(f"Loaded mocap data with shape {kp_data.shape}")

            # Create unique output filenames for AT004
            clip_suffix = f"AT004_trial{trial_num:02d}"
            fit_pattern = os.path.join(output_dir, f"{clip_suffix}_fit.h5")
            ik_pattern = os.path.join(output_dir, f"{clip_suffix}_ik.h5")

            # Skip if both output files already exist
            if os.path.exists(fit_pattern) and os.path.exists(ik_pattern):
                logging.info(f"Skipping trial {trial_num}: outputs already exist")
                continue

            # Set output paths
            cfg.stac.fit_offsets_path = f"{clip_suffix}_fit.h5"
            cfg.stac.ik_only_path = f"{clip_suffix}_ik.h5"

            # Run STAC
            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=Path(output_dir)
            )
            logging.info(f"Trial {trial_num}: saved fit to {fit_path}")
            logging.info(f"Trial {trial_num}: saved ik to {ik_only_path}")

            # Create visualization
            n_frames = min(cfg.model.N_SAMPLE_FRAMES, kp_data.shape[0])
            viz_cfg, render = stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=n_frames,
                save_path=os.path.join(output_dir, f"{clip_suffix}_stac.mp4"),
                height=480,
                width=640,
            )
            logging.info(f"Trial {trial_num}: STAC visualization saved")

        except Exception as e:
            logging.error(f"Error processing trial {trial_num}: {str(e)}")
            logging.error(traceback.format_exc())

        # Clean up memory
        gc.collect()

    logging.info(f"\n{'='*60}")
    logging.info("All AT004 clips processed!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
