#!/usr/bin/env python
"""Smoke test: run STAC v16 recipe on existing v7.h5 for one clip per animal.

Picks the first valid reach trial per session from _reach_csv_clean CSVs and
renders a mujoco viz mp4. Skips reprojection (no env labels needed).
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gc
import logging
import traceback
from pathlib import Path

import h5py
import hydra
import jax
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import stac_mjx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"Using JAX devices: {jax.devices()}")


SESSIONS = [
    {
        "id": "A36-1_2023-07-18_16-54-01_lightOff",
        "v7":   "/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "csv":  "/mnt/vast/eric/azim-emg-data/_reach_csv_clean/A36-1_reach_metrics.csv",
    },
    {
        "id": "AT006_2024-02-28_15-39-40_LightOff",
        "v7":   "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video/AT006_2024-02-28_15-39-40_LightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "csv":  "/mnt/vast/eric/azim-emg-data/_reach_csv_clean/AT006_reach_metrics.csv",
    },
    {
        "id": "AT009_2024-04-25_13-50-34_LightOff",
        "v7":   "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video/AT009_2024-04-25_13-50-34_LightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "csv":  "/mnt/vast/eric/azim-emg-data/_reach_csv_clean/AT009_reach_metrics.csv",
    },
    {
        "id": "AT012_2024-06-21_12-32-30_LightOff",
        "v7":   "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video/AT012_2024-06-21_12-32-30_LightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "csv":  "/mnt/vast/eric/azim-emg-data/_reach_csv_clean/AT012_reach_metrics.csv",
    },
    {
        "id": "AT013_2024-06-20_13-41-20_LightOff",
        "v7":   "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video/AT013_2024-06-20_13-41-20_LightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "csv":  "/mnt/vast/eric/azim-emg-data/_reach_csv_clean/AT013_reach_metrics.csv",
    },
]


def first_valid_reach(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["reach_startPos"])
    if len(df) == 0:
        raise RuntimeError(f"No valid reaches in {csv_path}")
    row = df.iloc[0]
    trial_start = int(row["start_idx"])
    reach_off = int(row["reach_startPos"])
    return trial_start + reach_off


def extract_clip(v7_path, start_frame, n_frames=100):
    start_frame = int(start_frame)
    end_frame = start_frame + n_frames
    with h5py.File(v7_path, "r") as f:
        tracks = f["tracks"][:]
        total = tracks.shape[0]
        if end_frame > total:
            shortfall = end_frame - total
            start_frame = max(0, start_frame - shortfall)
            end_frame = start_frame + n_frames
        clip = tracks[start_frame:end_frame]
        if clip.shape[0] < n_frames:
            pad = clip[-1:].repeat(n_frames - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        else:
            clip = clip[:n_frames]
    assert clip.shape[0] == n_frames
    return clip, start_frame


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()

    base_path = Path("/mnt/vast/niyo/stac-mjx").resolve()
    output_dir = base_path / "smoke_test_v7_5sessions"
    output_dir.mkdir(exist_ok=True)
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "model") or isinstance(cfg.model, str):
        cfg.model = OmegaConf.load(base_path / "configs/model/mouse_arm.yaml")
    OmegaConf.set_struct(cfg.model, False)

    OmegaConf.update(cfg.model, "MJCF_PATH",
                     str(base_path / "models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml"),
                     merge=True)
    OmegaConf.update(cfg.model, "KEYPOINT_MODEL_PAIRS",
                     {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"},
                     merge=True)
    OmegaConf.update(cfg.model, "SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "MOCAP_SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "N_ITERS", 6, merge=True)
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", 50, merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)

    for sess in SESSIONS:
        sid = sess["id"]
        logging.info(f"==== {sid} ====")
        try:
            start_frame = first_valid_reach(sess["csv"])
            logging.info(f"  First valid reach starts at frame {start_frame}")

            clip_data, eff_start = extract_clip(sess["v7"], start_frame, n_frames=100)
            logging.info(f"  Extracted clip shape={clip_data.shape} from frame {eff_start}")

            clip_file = clip_dir / f"{sid}_clip.h5"
            with h5py.File(clip_file, "w") as h5f:
                h5f.create_dataset("tracks", data=clip_data)

            OmegaConf.set_struct(cfg.stac, False)
            cfg.stac.data_path = str(clip_file)

            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
            OmegaConf.update(cfg.stac, "n_frames_per_clip", int(kp_data.shape[0]), merge=True)
            logging.info(f"  Loaded mocap shape={kp_data.shape}")

            cfg.stac.fit_offsets_path = f"{sid}_fit.h5"
            cfg.stac.ik_only_path = f"{sid}_ik.h5"

            fit_path, ik_only_path = stac_mjx.run_stac(
                cfg, kp_data, sorted_kp_names, base_path=output_dir
            )
            logging.info(f"  STAC done: fit={fit_path}, ik={ik_only_path}")

            n_frames = min(50, kp_data.shape[0])
            stac_mjx.viz_stac(
                data_path=ik_only_path,
                n_frames=n_frames,
                save_path=str(output_dir / f"{sid}_stac.mp4"),
                camera="janelia_cam",
                height=480,
                width=640,
            )
            logging.info(f"  Mujoco viz saved -> {output_dir / f'{sid}_stac.mp4'}")

        except Exception as e:
            logging.error(f"  FAILED {sid}: {e}")
            logging.error(traceback.format_exc())

        gc.collect()

    logging.info("Smoke test complete")


if __name__ == "__main__":
    main()
