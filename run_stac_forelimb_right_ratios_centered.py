#!/usr/bin/env python
"""Run STAC-MJX on per-reach clips from v4 registered forelimb data."""
import os
from glob import glob
import h5py
import numpy as np
import gc
import logging
import traceback
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

import stac_mjx
import jax
logging.info(f"Using JAX devices: {jax.devices()}")

# ---------------------- Config ------------------------------

SESSION_ID = "A36-1_2023-07-18_16-54-01_lightOff"
REGISTERED_FILE = (
    "/mnt/vast/eric/CVAT_mouse_reach/all_data/"
    f"{SESSION_ID}_video/"
    f"{SESSION_ID}_video_mujoco_registered_forelimb_right_ratios_centered.h5"
)
REACHES_CSV = (
    "/mnt/vast/niyo/stac-mjx/refined_reaches/"
    "A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv"
)
N_CLIP_FRAMES = 100

# ---------------------- Helpers ------------------------------

def load_refined_reach_events(csv_path, janelia_clip_dir):
    """Load reach events from the refined CSV, using only trials that exist in janelia clips."""
    df = pd.read_csv(csv_path)

    # Get trial indices from existing janelia clips
    janelia_trials = set()
    for fname in os.listdir(janelia_clip_dir):
        if fname.endswith("_clip.h5"):
            # Extract trial number from e.g. "..._trial01_clip.h5"
            trial_str = fname.split("_trial")[1].split("_clip")[0]
            janelia_trials.add(int(trial_str))
    logging.info(f"Found {len(janelia_trials)} trials in janelia clips: {sorted(janelia_trials)}")

    # Filter CSV to only these trials
    valid = df[df['trial'].isin(janelia_trials)].copy()
    valid['trial'] = valid['trial'].astype(int)
    valid['trial_start'] = valid['trial_start'].astype(int)
    valid['start'] = valid['start'].astype(int)
    valid['end'] = valid['end'].astype(int)
    valid['reach_start_frame'] = valid['trial_start'] + valid['start']
    return valid


def extract_clip_data(registered_file, start_frame, n_frames=N_CLIP_FRAMES):
    """Extract exactly n_frames from registered H5 starting at start_frame."""
    start_frame = int(start_frame)
    end_frame = start_frame + n_frames

    with h5py.File(registered_file, "r") as f:
        tracks = f["tracks"][:]  # (frames, 1, nodes, 3)
        total_frames = tracks.shape[0]

        if end_frame > total_frames:
            adjustment = n_frames - (total_frames - start_frame)
            start_frame = max(0, start_frame - adjustment)
            end_frame = start_frame + n_frames
            logging.warning(f"Adjusted window to {start_frame}-{end_frame}")

        clip_data = tracks[start_frame:end_frame]

        if clip_data.shape[0] < n_frames:
            padding = n_frames - clip_data.shape[0]
            clip_data = np.concatenate(
                [clip_data, clip_data[-1:].repeat(padding, axis=0)], axis=0
            )
            logging.info(f"Padded clip with {padding} repeated frames")

    assert clip_data.shape[0] == n_frames
    return clip_data


# ---------------------- Config Structure --------------------------

def ensure_config_structure(cfg, base_path):
    """Ensure cfg has proper stac/model structure for STAC-MJX.

    When Hydra loads a stac config directly (not through config.yaml defaults),
    the stac fields are at the top level and the model config needs to be loaded
    separately from the model YAML file.
    """
    OmegaConf.set_struct(cfg, False)

    # Create cfg.stac if stac fields are at top level
    if not hasattr(cfg, 'stac'):
        cfg.stac = OmegaConf.create({})

    # Ensure stac has required fields from top-level
    for key in ['data_path', 'n_frames_per_clip', 'n_fit_frames', 'fit_offsets_path',
                'ik_only_path', 'num_clips', 'skip_fit_offsets', 'skip_ik_only',
                'infer_qvels', 'mujoco']:
        if hasattr(cfg, key) and not hasattr(cfg.stac, key):
            cfg.stac[key] = cfg[key]

    # Load model config from YAML if cfg.model is a string (config group name)
    if not hasattr(cfg, 'model') or isinstance(cfg.model, str):
        model_name = cfg.model if hasattr(cfg, 'model') and isinstance(cfg.model, str) else 'mouse_forelimb_right'
        model_config_path = os.path.join(base_path, f"configs/model/{model_name}.yaml")
        if os.path.exists(model_config_path):
            cfg.model = OmegaConf.load(model_config_path)
            logging.info(f"Loaded model config: {model_name}")
        else:
            logging.error(f"Model config not found: {model_config_path}")

    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.stac, False)
    return cfg


# ---------------------- Main ------------------------------

@hydra.main(config_path="configs", config_name="stac/stac_mouse_forelimb_right_ratios_centered", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()

    base_path = os.path.abspath("/mnt/vast/niyo/stac-mjx")
    output_dir = os.path.join(base_path, "refined_STACed_data_forelimb_right_ratios_centered")
    clip_dir = os.path.join(output_dir, "reach_clips")
    os.makedirs(clip_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Ensure proper config structure (loads model YAML if needed)
    cfg = ensure_config_structure(cfg, base_path)

    # Load reach events — use same trial indices as janelia runs
    janelia_clip_dir = os.path.join(base_path, "refined_STACed_data_janelia", "reach_clips")
    valid_reaches = load_refined_reach_events(REACHES_CSV, janelia_clip_dir)
    logging.info(f"Found {len(valid_reaches)} valid reaches")

    # Verify registered file
    if not os.path.exists(REGISTERED_FILE):
        logging.error(f"Registered file not found: {REGISTERED_FILE}")
        return

    with h5py.File(REGISTERED_FILE, "r") as f:
        total_frames = f["tracks"].shape[0]
    logging.info(f"Registered file: {total_frames} total frames")

    for _, reach in valid_reaches.iterrows():
        trial_num = int(reach['trial'])
        start_frame = int(reach['reach_start_frame'])
        clip_suffix = f"{SESSION_ID}_trial{trial_num:02d}"

        logging.info(f"Trial {trial_num}: start_frame={start_frame}")

        try:
            # Extract clip
            clip_data = extract_clip_data(REGISTERED_FILE, start_frame, N_CLIP_FRAMES)
            logging.info(f"Clip shape: {clip_data.shape}")

            # Save clip H5
            clip_file = os.path.join(clip_dir, f"{clip_suffix}_clip.h5")
            with h5py.File(clip_file, "w") as h5f:
                h5f.create_dataset("tracks", data=clip_data)

            # Point config at this clip
            orig_data_path = cfg.stac.data_path
            cfg.stac.data_path = clip_file

            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, Path(base_path))

            cfg.stac.data_path = orig_data_path

            # Set per-clip params
            clip_length = kp_data.shape[0]
            OmegaConf.update(cfg.stac, "n_frames_per_clip", clip_length, merge=True)
            OmegaConf.update(cfg.stac, "n_fit_frames", clip_length, merge=True)
            cfg.stac.fit_offsets_path = f"{clip_suffix}_fit.h5"
            cfg.stac.ik_only_path = f"{clip_suffix}_ik.h5"

            logging.info(f"Mocap shape: {kp_data.shape}, kp_names: {sorted_kp_names}")

            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=Path(output_dir)
            )
            logging.info(f"Trial {trial_num}: fit={fit_path}, ik={ik_only_path}")

            # Render visualization using janelia_cam (matches SCAMPER/vnl-playground)
            n_viz_frames = min(cfg.model.N_SAMPLE_FRAMES, kp_data.shape[0])
            viz_cfg, render = stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=n_viz_frames,
                save_path=os.path.join(output_dir, f"{clip_suffix}_stac.mp4"),
                camera="janelia_cam",
                height=480,
                width=640,
            )
            logging.info(f"Trial {trial_num}: visualization saved")

        except Exception as e:
            logging.error(f"Error on trial {trial_num}: {e}")
            logging.error(traceback.format_exc())

        gc.collect()

    logging.info("All reaches processed")


if __name__ == "__main__":
    main()
