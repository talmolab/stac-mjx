#!/usr/bin/env python
"""Smoke test: replicate test_reprojection_trial01_v16_fixed_ref.mp4 style for
AT006 (new) and A36 (re-render with new env labels).

Steps per session:
  1. Triangulate env labels from draw-env-emg-project_with-A36-template.slp
     (frame 0 labels per camera + calibration.toml -> env_points3d.h5).
  2. Pick first valid reach trial from reach_metrics.csv.
  3. Reuse existing STAC ik H5 if present, else run STAC v16 recipe on the clip.
  4. Render multi-view reprojection mp4 via render_reprojection_video().

AT009/AT012/AT013 are skipped — no env labels exist for those sessions yet.
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gc
import json
import logging
import traceback
from pathlib import Path

import cv2
import h5py
import hydra
import jax
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import stac_mjx
from stac_mjx.multicam import load_calibration
from stac_mjx.reproject_stac_model import render_reprojection_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"JAX devices: {jax.devices()}")


# -----------------------------------------------------------------------------
# Sessions to run reprojection for. Only sessions with env labels in the new slp.
# -----------------------------------------------------------------------------
SLP_PATH = "/mnt/vast/eric/azim-emg-data/draw-env/draw-env-emg-project_with-A36-template.slp"

SESSIONS = [
    {
        "id": "A36-1_2023-07-18_16-54-01_lightOff",
        "video_match": "A36-1_2023-07-18_16-54-01_lightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video",
        # Use OLD CVAT_mouse_reach v7 — its shoulder_reference_mm matches the first
        # raw frame, which is what produced the working test_reprojection_trial01_v16_fixed_ref.mp4.
        # The NEW azim-emg-data v7 uses median-resting reference (different rotation, flips arm).
        "v7": "/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5",
        "raw_pts": "/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/A36-1_reach_metrics.csv",
        # Use new manual CSV (different from original v16 baseline) — re-fit STAC for consistency
        # If a smoke-test IK already exists from a prior run, reuse it. Recipe
        # is deterministic so re-fitting yields the same IK, just slower.
        "existing_ik": "AUTO",
        # Use the ORIGINAL env points used by test_reprojection_trial01_v16_fixed_ref.mp4
        # (skip triangulating the new slp; that introduced small differences)
        "existing_env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
    },
    {
        "id": "AT006_2024-02-28_15-39-40_LightOff",
        "video_match": "AT006_2024-02-28_15-39-40_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video/AT006_2024-02-28_15-39-40_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT006_2024-02-28_15-39-40_LightOff_video/AT006_2024-02-28_15-39-40_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT006_reach_metrics.csv",
        # If a smoke-test IK already exists from a prior run, reuse it. Recipe
        # is deterministic so re-fitting yields the same IK, just slower.
        "existing_ik": "AUTO",
    },
    {
        "id": "AT009_2024-04-25_13-50-34_LightOff",
        "video_match": "AT009_2024-04-25_13-50-34_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video/AT009_2024-04-25_13-50-34_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT009_2024-04-25_13-50-34_LightOff_video/AT009_2024-04-25_13-50-34_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT009_reach_metrics.csv",
        # No env labels for AT009 — use A36's env points as fallback (will appear ~few px off)
        "existing_env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
        # If a smoke-test IK already exists from a prior run, reuse it. Recipe
        # is deterministic so re-fitting yields the same IK, just slower.
        "existing_ik": "AUTO",
    },
    {
        "id": "AT012_2024-06-21_12-32-30_LightOff",
        "video_match": "AT012_2024-06-21_12-32-30_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video/AT012_2024-06-21_12-32-30_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT012_2024-06-21_12-32-30_LightOff_video/AT012_2024-06-21_12-32-30_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT012_reach_metrics.csv",
        "existing_env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
        # If a smoke-test IK already exists from a prior run, reuse it. Recipe
        # is deterministic so re-fitting yields the same IK, just slower.
        "existing_ik": "AUTO",
    },
    {
        "id": "AT013_2024-06-20_13-41-20_LightOff",
        "video_match": "AT013_2024-06-20_13-41-20_LightOff",
        "session_dir": "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video",
        "v7": "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video/AT013_2024-06-20_13-41-20_LightOff_video_mujoco_registered_forelimb_right_v7_oldrecipe.h5",
        "raw_pts": "/mnt/vast/eric/azim-emg-data/AT013_2024-06-20_13-41-20_LightOff_video/AT013_2024-06-20_13-41-20_LightOff_video_points3d.h5",
        "csv": "/mnt/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT013_reach_metrics.csv",
        "existing_env_h5": "/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
        # If a smoke-test IK already exists from a prior run, reuse it. Recipe
        # is deterministic so re-fitting yields the same IK, just slower.
        "existing_ik": "AUTO",
    },
]


# -----------------------------------------------------------------------------
# 1. Env triangulation from slp + calibration.toml
# -----------------------------------------------------------------------------
def read_slp_skeleton(slp_path):
    with h5py.File(slp_path, "r") as f:
        skel = json.loads(f["metadata"].attrs["json"])
    # nodes_json may be string-form
    nodes_json = skel.get("nodes")
    if isinstance(nodes_json, str):
        nodes_json = json.loads(nodes_json)
    return [n["name"] if isinstance(n, dict) else n for n in nodes_json]


def find_videos_for_session(slp_path, video_match_substr):
    """Return {cam_name: video_idx} for the first triplet matching the session."""
    with h5py.File(slp_path, "r") as f:
        vids_json = f["videos_json"][()]
    cam_to_idx = {}
    for i, v in enumerate(vids_json):
        d = json.loads(v if isinstance(v, str) else v.decode())
        fn = d.get("backend", {}).get("filename", "") or d.get("filename", "")
        if video_match_substr not in fn:
            continue
        for cam in ("CamA", "CamB", "CamC"):
            if f"/{cam}/" in fn and cam not in cam_to_idx:
                cam_to_idx[cam] = i
                break
        if len(cam_to_idx) == 3:
            break
    return cam_to_idx


def get_frame0_points(slp_path, video_idx):
    """Return (n_nodes, 2) array of [x, y] for frame 0 of video_idx, NaN if missing."""
    with h5py.File(slp_path, "r") as f:
        frames = f["frames"][:]
        instances = f["instances"][:]
        points = f["points"][:]
        # find a frame with this video and frame_idx==0
        mask = (frames["video"] == video_idx) & (frames["frame_idx"] == 0)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            raise RuntimeError(f"No frame 0 label for video {video_idx} in {slp_path}")
        frame_id = int(frames["frame_id"][idxs[0]])
        # find instance for this frame
        inst_idxs = np.where(instances["frame_id"] == frame_id)[0]
        if len(inst_idxs) == 0:
            raise RuntimeError(f"No instance for frame {frame_id}")
        inst = instances[inst_idxs[0]]
        ps = int(inst["point_id_start"])
        pe = int(inst["point_id_end"])
        pts = points[ps:pe]
    n = len(pts)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    for i, p in enumerate(pts):
        if p["visible"]:
            out[i] = [p["x"], p["y"]]
    return out


def triangulate_env(slp_path, calib_path, video_match, out_h5):
    """Triangulate frame-0 env labels for one session -> env_points3d.h5."""
    node_names = read_slp_skeleton(slp_path)
    logging.info(f"  Skeleton nodes ({len(node_names)}): {node_names}")

    cam_to_idx = find_videos_for_session(slp_path, video_match)
    logging.info(f"  Camera -> video index: {cam_to_idx}")
    if len(cam_to_idx) != 3:
        raise RuntimeError(f"Expected 3 cameras, got {len(cam_to_idx)}")

    # Load calibration
    cams = load_calibration(calib_path)
    cams_by_name = {c.name: c for c in cams}

    # Get 2D points per camera
    pts_2d = {}
    for cam_name in ("CamA", "CamB", "CamC"):
        v_idx = cam_to_idx[cam_name]
        pts_2d[cam_name] = get_frame0_points(slp_path, v_idx)
    n_nodes = pts_2d["CamA"].shape[0]
    assert all(p.shape[0] == n_nodes for p in pts_2d.values())

    # Undistort -> normalized image coords -> back to pixel coords using K
    P_per_cam = {}
    pts_norm = {}
    for cam_name in ("CamA", "CamB", "CamC"):
        c = cams_by_name[cam_name]
        # Undistort to pixel coords (P=K so output is in pixel coords undistorted)
        pts = pts_2d[cam_name].astype(np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, c.K, c.dist, P=c.K).reshape(-1, 2)
        pts_norm[cam_name] = und
        Rt = np.hstack([c.R, c.tvec.reshape(3, 1)])
        P_per_cam[cam_name] = c.K @ Rt  # 3x4

    # Triangulate each node using DLT with all 3 cameras
    points_3d = np.full((n_nodes, 3), np.nan, dtype=np.float64)
    reproj_errs = np.full((n_nodes, 3), np.nan, dtype=np.float64)
    for i in range(n_nodes):
        # Skip if any camera has NaN
        cams_use = []
        for cam_name in ("CamA", "CamB", "CamC"):
            if not np.isnan(pts_norm[cam_name][i]).any():
                cams_use.append(cam_name)
        if len(cams_use) < 2:
            logging.warning(f"  Node '{node_names[i]}': only {len(cams_use)} cameras have it, skipping")
            continue
        # Build A matrix for DLT: each camera contributes 2 rows
        A = []
        for cam_name in cams_use:
            x, y = pts_norm[cam_name][i]
            P = P_per_cam[cam_name]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X = X_h[:3] / X_h[3]
        points_3d[i] = X
        # reprojection error
        for cam_name in cams_use:
            c = cams_by_name[cam_name]
            proj = c.project_points(X[None]).ravel()
            err = np.linalg.norm(pts_norm[cam_name][i] - proj)
            reproj_errs[i, ("CamA", "CamB", "CamC").index(cam_name)] = err

    # Save in load_env_points3d format: points_3d (1,1,N,3), node_names
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("points_3d", data=points_3d.reshape(1, 1, n_nodes, 3))
        f.create_dataset("node_names", data=np.array(node_names, dtype="S64"))
        f.create_dataset("reprojection_error", data=reproj_errs.reshape(1, 1, n_nodes, 3))
    logging.info(f"  Triangulated env -> {out_h5}")
    logging.info(f"  Reprojection error: mean={np.nanmean(reproj_errs):.3f}px, max={np.nanmax(reproj_errs):.3f}px")
    return out_h5


# -----------------------------------------------------------------------------
# 2. Reach trial picker
# -----------------------------------------------------------------------------
def first_valid_reach(csv_path, pre_roll=5):
    """Pick first valid reach. Subtract `pre_roll` frames so the clip starts a bit
    before the labeled reach_start (first frames show pre-reach rest)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["reach_startPos"])
    if len(df) == 0:
        raise RuntimeError(f"No valid reaches in {csv_path}")
    row = df.iloc[0]
    trial_start = int(row["start_idx"])
    reach_off = int(row["reach_startPos"]) - pre_roll
    return {
        "trial_start": trial_start,
        "start": reach_off,
        "end": int(row["reach_endPos"]) if not pd.isna(row.get("reach_endPos")) else reach_off + 200,
        "reach_start_frame": trial_start + reach_off,
    }


def write_v16_csv(csv_path, trial_info, trial_number=1):
    """Write a CSV in v16 format expected by load_trial_csv: trial,trial_start,start,end,"""
    with open(csv_path, "w") as f:
        f.write("trial,trial_start,start,end,\n")
        f.write(f"{trial_number},{trial_info['trial_start']},{trial_info['start']},{trial_info['end']},\n")


def extract_clip(v7_path, start_frame, n_frames=50):  # 50 frames @ 200fps real = 250ms
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
    return clip, start_frame


# -----------------------------------------------------------------------------
# 3. Main: per-session smoke test
# -----------------------------------------------------------------------------
@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()

    base_path = Path("/mnt/vast/niyo/stac-mjx").resolve()
    output_dir = base_path / "smoke_test_v16_reproj"
    output_dir.mkdir(exist_ok=True)
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(exist_ok=True)

    # STAC config (matches v16)
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
    OmegaConf.update(cfg.model, "N_SAMPLE_FRAMES", 50, merge=True)
    OmegaConf.update(cfg.model, "M_REG_COEF", 0, merge=True)

    for sess in SESSIONS:
        sid = sess["id"]
        logging.info(f"==== {sid} ====")
        try:
            calib_path = Path(sess["session_dir"]) / "calibration.toml"

            # Step 1: env triangulation (or reuse existing)
            if sess.get("existing_env_h5") and Path(sess["existing_env_h5"]).exists():
                env_h5 = Path(sess["existing_env_h5"])
                logging.info(f"Step 1: reusing existing env points -> {env_h5}")
            else:
                env_h5 = output_dir / f"{sid}_env_points3d.h5"
                logging.info(f"Step 1: triangulating env labels ({sid})...")
                triangulate_env(SLP_PATH, str(calib_path), sess["video_match"], str(env_h5))

            # Step 2: reach trial -- prefer existing v16-format CSV when provided
            if sess.get("v16_csv") and Path(sess["v16_csv"]).exists():
                v16_csv = Path(sess["v16_csv"])
                trial_number = sess.get("trial_number", 1)
                logging.info(f"Step 2: using existing v16 CSV {v16_csv} trial {trial_number}")
            else:
                trial_info = first_valid_reach(sess["csv"])
                trial_number = 1
                start_frame = trial_info["reach_start_frame"]
                logging.info(f"Step 2: synthesized trial trial_start={trial_info['trial_start']}, "
                             f"reach_start_frame={start_frame}")

            # Step 3: STAC (reuse existing if present)
            existing_ik = sess.get("existing_ik")
            if existing_ik == "AUTO":
                # Auto-detect from prior smoke-test run
                candidate = output_dir / f"{sid}_ik.h5"
                existing_ik = str(candidate) if candidate.exists() else None
            if existing_ik and Path(existing_ik).exists():
                ik_path = existing_ik
                logging.info(f"Step 3: reusing existing STAC IK -> {ik_path}")
            else:
                clip_data, eff_start = extract_clip(sess["v7"], start_frame, n_frames=50)
                clip_file = clip_dir / f"{sid}_clip.h5"
                with h5py.File(clip_file, "w") as h5f:
                    h5f.create_dataset("tracks", data=clip_data)

                OmegaConf.set_struct(cfg.stac, False)
                cfg.stac.data_path = str(clip_file)
                kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
                OmegaConf.update(cfg.stac, "n_frames_per_clip", int(kp_data.shape[0]), merge=True)
                cfg.stac.fit_offsets_path = f"{sid}_fit.h5"
                cfg.stac.ik_only_path = f"{sid}_ik.h5"
                _, ik_path = stac_mjx.run_stac(
                    cfg, kp_data, sorted_kp_names, base_path=output_dir
                )
                logging.info(f"Step 3: STAC done -> {ik_path}")

            # Step 4: render reprojection video (use v16 csv from above)
            if not sess.get("v16_csv"):
                v16_csv = output_dir / f"{sid}_v16_trials.csv"
                write_v16_csv(str(v16_csv), trial_info, trial_number=trial_number)

            output_mp4 = output_dir / f"{sid}_trial{trial_number:02d}_reproj_v16.mp4"
            logging.info(f"Step 4: rendering reprojection video -> {output_mp4}")
            render_reprojection_video(
                stac_h5_path=str(ik_path),
                calibration_path=str(calib_path),
                video_dir=sess["session_dir"],
                env_points3d_path=str(env_h5),
                registered_h5_path=sess["v7"],
                raw_points3d_path=sess["raw_pts"],
                clip_csv_path=str(v16_csv),
                trial_number=trial_number,
                output_path=str(output_mp4),
                fps=30,
            )
            logging.info(f"DONE: {output_mp4}")
        except Exception as e:
            logging.error(f"FAILED {sid}: {e}")
            logging.error(traceback.format_exc())

        gc.collect()

    logging.info("Smoke test complete")


if __name__ == "__main__":
    main()
