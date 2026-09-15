#!/usr/bin/env python
"""Run STAC on the janelia model with moving shoulder v2 (loose joints) for trial 01, then render reprojection video (v16)."""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import logging
import traceback
import gc
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import stac_mjx
import jax

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"Using JAX devices: {jax.devices()}")


def load_refined_reach_events(csv_path):
    df = pd.read_csv(csv_path)
    valid_trials = df[(df['start'] != 0) & (df['end'] != 0)].copy()
    valid_trials['trial'] = valid_trials['trial'].astype(int)
    valid_trials['trial_start'] = valid_trials['trial_start'].astype(int)
    valid_trials['start'] = valid_trials['start'].astype(int)
    valid_trials['end'] = valid_trials['end'].astype(int)
    valid_trials['reach_start_frame'] = valid_trials['trial_start'] + valid_trials['start']
    return valid_trials


def extract_clip_data(registered_file, start_frame, n_frames=100):
    start_frame = int(start_frame)
    end_frame = start_frame + n_frames
    with h5py.File(registered_file, "r") as f:
        tracks = f["tracks"][:]
        total_frames = tracks.shape[0]
        if end_frame > total_frames:
            frames_available = total_frames - start_frame
            if frames_available < n_frames:
                adjustment = n_frames - frames_available
                start_frame = max(0, start_frame - adjustment)
                end_frame = start_frame + n_frames
        clip_data = tracks[start_frame:end_frame]
        actual_frames = clip_data.shape[0]
        if actual_frames < n_frames:
            padding_needed = n_frames - actual_frames
            last_frame = clip_data[-1:].repeat(padding_needed, axis=0)
            clip_data = np.concatenate([clip_data, last_frame], axis=0)
        elif actual_frames > n_frames:
            clip_data = clip_data[:n_frames]
    assert clip_data.shape[0] == n_frames
    return clip_data


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()

    base_path = os.path.abspath("/mnt/vast/niyo/stac-mjx")
    output_dir = os.path.join(base_path, "refined_STACed_data_janelia_moving_shoulder_v2")
    os.makedirs(output_dir, exist_ok=True)

    # Ensure config is mutable
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, 'model') or isinstance(cfg.model, str):
        model_config = OmegaConf.load(os.path.join(base_path, "configs/model/mouse_arm.yaml"))
        cfg.model = model_config
    OmegaConf.set_struct(cfg.model, False)

    # v16: use the v2 model with loosened joint limits
    OmegaConf.update(cfg.model, "MJCF_PATH", "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml", merge=True)
    OmegaConf.update(cfg.model, "KEYPOINT_MODEL_PAIRS", {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"}, merge=True)
    OmegaConf.update(cfg.model, "SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "MOCAP_SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "N_ITERS", 6, merge=True)
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", 50, merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)

    reg_file = "/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5"

    # Load reach events
    refined_csv_path = os.path.join(base_path, "refined_reaches/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv")
    valid_reaches = load_refined_reach_events(refined_csv_path)
    logging.info(f"Loaded {len(valid_reaches)} valid reaches")

    session_id = "A36-1_2023-07-18_16-54-01_lightOff"

    with h5py.File(reg_file, "r") as f:
        total_frames = f["tracks"].shape[0]
    logging.info(f"Registered file: {total_frames} total frames")

    # Process all trials
    for i, reach in valid_reaches.iterrows():
        trial_num = int(reach['trial'])
        start_frame = int(reach['reach_start_frame'])
        logging.info(f"Processing trial {trial_num} with start frame {start_frame}")

        try:
            clip_data = extract_clip_data(reg_file, start_frame, n_frames=100)
            logging.info(f"Extracted clip shape: {clip_data.shape}")

            clip_dir = os.path.join(output_dir, "reach_clips")
            os.makedirs(clip_dir, exist_ok=True)
            clip_file = os.path.join(clip_dir, f"{session_id}_trial{trial_num:02d}_clip.h5")

            with h5py.File(clip_file, "w") as h5f:
                h5f.create_dataset("tracks", data=clip_data)

            OmegaConf.set_struct(cfg.stac, False)
            cfg.stac.data_path = clip_file

            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, Path(base_path))

            clip_length = kp_data.shape[0]
            OmegaConf.update(cfg.stac, "n_frames_per_clip", clip_length, merge=True)
            logging.info(f"Loaded mocap data shape: {kp_data.shape}")

            clip_suffix = f"{session_id}_trial{trial_num:02d}"
            cfg.stac.fit_offsets_path = f"{clip_suffix}_fit.h5"
            cfg.stac.ik_only_path = f"{clip_suffix}_ik.h5"

            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=Path(output_dir)
            )
            logging.info(f"Trial {trial_num}: fit={fit_path}, ik={ik_only_path}")

            n_frames = min(50, kp_data.shape[0])
            viz_cfg, render = stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=n_frames,
                save_path=os.path.join(output_dir, f"{clip_suffix}_stac.mp4"),
                camera="janelia_cam",
                height=480,
                width=640,
            )
            logging.info(f"Trial {trial_num}: visualization saved")

            # Render reprojection video (v16)
            from stac_mjx.reproject_stac_model import render_reprojection_video

            reprojection_output = os.path.join(output_dir, f"{clip_suffix}_reprojection_v16.mp4")
            render_reprojection_video(
                stac_h5_path=str(ik_only_path),
                calibration_path="/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/calibration.toml",
                video_dir="/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env",
                env_points3d_path="/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
                registered_h5_path=reg_file,
                raw_points3d_path="/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/points3d.h5",
                clip_csv_path=refined_csv_path,
                trial_number=trial_num,
                output_path=reprojection_output,
                fps=30,
            )
            logging.info(f"Reprojection video saved to {reprojection_output}")

        except Exception as e:
            logging.error(f"Error processing trial {trial_num}: {e}")
            logging.error(traceback.format_exc())

        gc.collect()

    logging.info("Done")


if __name__ == "__main__":
    main()
