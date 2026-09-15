#!/usr/bin/env python
"""Batch STAC + v16 reprojection over all valid manual reaches in all 5 sessions.

Filter sessions by name substring(s) via env var STAC_SESSION_FILTER (comma-separated).
Set CUDA_VISIBLE_DEVICES to pin to one GPU when running 2 parallel instances.

For each session:
  - Use _reach_csv_manual_refined CSV
  - For each row with non-NaN reach_startPos:
      - Extract 50-frame clip starting at (start_idx + reach_startPos - 5)  (5-frame pre-roll)
      - Run STAC (v16 settings)
      - Render v16 reprojection mp4

Output:
  /mnt/vast/niyo/stac-mjx/final_STACed_5sessions_v16/
    <sid>/<sid>_trial<NN>_fit.h5
    <sid>/<sid>_trial<NN>_ik.h5
    <sid>/<sid>_trial<NN>_reproj_v16.mp4
    <sid>/<sid>_trial<NN>_v16_trials.csv

Reuses existing STAC IK if found (idempotent on re-runs).
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gc
import logging
import time
import traceback
from pathlib import Path

import h5py
import hydra
import jax
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import stac_mjx
from stac_mjx.reproject_stac_model import render_reprojection_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"JAX devices: {jax.devices()}")


PRE_ROLL = 5
N_FRAMES = 50

SESSIONS = [
    {
        "id": "A36-1_2023-07-18_16-54-01_lightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video",
        # A36 keeps the OLD CVAT v7 (matches the original test_reprojection_trial01_v16_fixed_ref.mp4)
        "v7": "/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "raw_pts": "/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/A36-1_reach_metrics.csv",
        "env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
    },
    {
        "id": "AT006_2024-02-28_15-39-40_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video/AT006_2024-02-28_15-39-40_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video/AT006_2024-02-28_15-39-40_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT006_reach_metrics.csv",
        "env_h5": "/mnt/vast/niyo/stac-mjx/smoke_test_v16_reproj/AT006_2024-02-28_15-39-40_LightOff_env_points3d.h5",
    },
    {
        "id": "AT009_2024-04-25_13-50-34_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video/AT009_2024-04-25_13-50-34_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video/AT009_2024-04-25_13-50-34_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT009_reach_metrics.csv",
        # AT009/12/13 fall back to A36's env points (no env labels for those sessions)
        "env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
    },
    {
        "id": "AT012_2024-06-21_12-32-30_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video/AT012_2024-06-21_12-32-30_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video/AT012_2024-06-21_12-32-30_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT012_reach_metrics.csv",
        "env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
    },
    {
        "id": "AT013_2024-06-20_13-41-20_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video/AT013_2024-06-20_13-41-20_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video/AT013_2024-06-20_13-41-20_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT013_reach_metrics.csv",
        "env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
    },
]


def load_valid_reaches(csv_path):
    df = pd.read_csv(csv_path)
    # Preserve original row index → trial number
    df = df.reset_index(drop=False).rename(columns={"index": "trial_idx"})
    df = df.dropna(subset=["reach_startPos"])
    if os.environ.get("STAC_REACH_REVERSE"):
        df = df.iloc[::-1].reset_index(drop=True)
    return df


def extract_clip(v7_path, start_frame, n_frames):
    start_frame = int(start_frame)
    end_frame = start_frame + n_frames
    with h5py.File(v7_path, "r") as f:
        tracks = f["tracks"][:]
        total = tracks.shape[0]
        if start_frame < 0:
            return None, "start_frame_negative"
        if end_frame > total:
            return None, "clip_runs_off_end"
        clip = tracks[start_frame:end_frame]
    return clip, None


def write_v16_csv(csv_path, trial_number, trial_start, start, end):
    with open(csv_path, "w") as f:
        f.write("trial,trial_start,start,end,\n")
        f.write(f"{trial_number},{trial_start},{start},{end},\n")


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()

    base_path = Path("/mnt/vast/niyo/stac-mjx").resolve()
    output_root = base_path / "final_STACed_5sessions_v16"
    output_root.mkdir(exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "model") or isinstance(cfg.model, str):
        cfg.model = OmegaConf.load(base_path / "configs/model/mouse_arm.yaml")
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.update(cfg.model, "MJCF_PATH",
                     str(base_path / "models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml"),
                     merge=True)
    OmegaConf.update(cfg.model, "KEYPOINT_MODEL_PAIRS",
                     {"Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist"}, merge=True)
    OmegaConf.update(cfg.model, "SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "MOCAP_SCALE_FACTOR", 1.0, merge=True)
    OmegaConf.update(cfg.model, "N_ITERS", 6, merge=True)
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", min(N_FRAMES, 50), merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)

    t_start = time.time()
    n_done = n_skipped = n_failed = 0

    # Filter sessions if STAC_SESSION_FILTER env var is set (comma-separated substrings)
    filt = os.environ.get("STAC_SESSION_FILTER", "")
    if filt:
        wanted = [w.strip() for w in filt.split(",") if w.strip()]
        sessions_to_run = [s for s in SESSIONS if any(w in s["id"] for w in wanted)]
        logging.info(f"Session filter: {wanted} -> {len(sessions_to_run)}/{len(SESSIONS)} sessions")
    else:
        sessions_to_run = SESSIONS

    for sess in sessions_to_run:
        sid = sess["id"]
        sess_out = output_root / sid
        sess_out.mkdir(exist_ok=True)
        clip_dir = sess_out / "clips"
        clip_dir.mkdir(exist_ok=True)

        reaches = load_valid_reaches(sess["csv"])
        logging.info(f"==== {sid}: {len(reaches)} reaches ====")

        for _, row in reaches.iterrows():
            trial_number = int(row["trial_idx"])  # 0-indexed row in CSV
            trial_start = int(row["start_idx"])
            reach_off = int(row["reach_startPos"]) - PRE_ROLL
            reach_end = int(row["reach_endPos"]) if not pd.isna(row.get("reach_endPos")) else reach_off + N_FRAMES
            clip_start = trial_start + reach_off

            tag = f"{sid}_trial{trial_number:03d}"
            ik_path = sess_out / f"{tag}_ik.h5"
            mp4_path = sess_out / f"{tag}_reproj_v16.mp4"

            if mp4_path.exists():
                n_skipped += 1
                continue

            try:
                # 1. Extract clip + run STAC if no IK yet
                if not ik_path.exists():
                    clip_data, err = extract_clip(sess["v7"], clip_start, N_FRAMES)
                    if err:
                        logging.warning(f"[{tag}] skip: {err} (clip_start={clip_start})")
                        n_failed += 1
                        continue

                    clip_file = clip_dir / f"{tag}_clip.h5"
                    with h5py.File(clip_file, "w") as h5f:
                        h5f.create_dataset("tracks", data=clip_data)

                    OmegaConf.set_struct(cfg.stac, False)
                    cfg.stac.data_path = str(clip_file)
                    kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
                    OmegaConf.update(cfg.stac, "n_frames_per_clip", int(kp_data.shape[0]), merge=True)
                    cfg.stac.fit_offsets_path = f"{tag}_fit.h5"
                    cfg.stac.ik_only_path = f"{tag}_ik.h5"
                    _, ik_path_out = stac_mjx.run_stac(
                        cfg, kp_data, sorted_kp_names, base_path=sess_out
                    )
                    ik_path = Path(ik_path_out)

                # 2. Write v16-format CSV (single-trial) and render
                v16_csv = sess_out / f"{tag}_v16_trials.csv"
                write_v16_csv(str(v16_csv), trial_number, trial_start, reach_off, reach_end)

                render_reprojection_video(
                    stac_h5_path=str(ik_path),
                    calibration_path=str(Path(sess["session_dir"]) / "calibration.toml"),
                    video_dir=sess["session_dir"],
                    env_points3d_path=sess["env_h5"],
                    registered_h5_path=sess["v7"],
                    raw_points3d_path=sess["raw_pts"],
                    clip_csv_path=str(v16_csv),
                    trial_number=trial_number,
                    output_path=str(mp4_path),
                    fps=30,
                )
                n_done += 1
                elapsed = time.time() - t_start
                logging.info(
                    f"[{n_done + n_skipped}/{278}] {tag} done — elapsed {elapsed/60:.1f}min"
                )
            except Exception as e:
                logging.error(f"[{tag}] FAILED: {e}")
                logging.error(traceback.format_exc())
                n_failed += 1

            gc.collect()

    elapsed = time.time() - t_start
    logging.info(
        f"All sessions complete: {n_done} done, {n_skipped} skipped, "
        f"{n_failed} failed in {elapsed/60:.1f}min"
    )


if __name__ == "__main__":
    main()
