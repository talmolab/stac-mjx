"""v17 reproject that translates the rendered MuJoCo model in image space so its
Wrist position lands at the raw SLEAP `right_wrist` 2D pixel position. This
bypasses the bone-rescaling inverse-transform issue (see
azim-emg-data/scratch/plans/V17_REPROJ_BONE_RESCALING_BUG.md) at the cost of
visual-only correction (the underlying STAC fit is unchanged).

Mocap dots = raw `points_2d` from SLEAP h5 directly (no inverse projection).
STAC X marks = `marker_sites` projected then shifted by the same per-frame offset.
Model overlay = mujoco renderer output, translated in 2D to align Wrist.
"""
from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
import cv2
import h5py
import mujoco
import imageio
from tqdm import tqdm
from omegaconf import OmegaConf

from stac_mjx.multicam import (
    load_calibration, load_registration_transform, load_env_points3d,
    load_raw_shoulder_positions,
)
from stac_mjx.reproject_stac_model import (
    _setup_mujoco_model, _set_green_background, _make_scene_option,
    _compute_calibration_cam_xyaxes, _render_model_frame, _composite,
    _draw_env_markers, _draw_mocap_keypoints, load_trial_csv,
    _BG_COLOR, _BG_THRESH,
)
from stac_mjx import io as smjx_io


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ik-h5", required=True)
    p.add_argument("--registered-h5", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--env-points3d", required=True)
    p.add_argument("--raw-points3d", required=True)
    p.add_argument("--digits-points3d", required=True,
                    help="<session>_digits_points3d.h5 — for the d2_mid/d5_mid raw 2D")
    p.add_argument("--clip-csv", required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--render-w", type=int, default=512)
    p.add_argument("--render-h", type=int, default=512)
    args = p.parse_args()

    print(f"Loading data for trial {args.trial}...")
    cameras = load_calibration(args.calibration)
    print(f"  Loaded {len(cameras)} cameras: {[c.name for c in cameras]}")

    cfg_loaded, stac_data = smjx_io.load_stac_data(args.ik_h5)
    kp_data = stac_data.kp_data
    marker_sites = stac_data.marker_sites
    qposes = stac_data.qpos

    reg = load_registration_transform(args.registered_h5)
    env_data = load_env_points3d(args.env_points3d)
    raw_shoulders = load_raw_shoulder_positions(args.raw_points3d)

    trials = load_trial_csv(args.clip_csv)
    trial_info = trials[args.trial]
    video_start_frame = trial_info["trial_start"] + trial_info["start"]
    n_frames = qposes.shape[0]
    print(f"  Trial {args.trial}: video frames {video_start_frame}..{video_start_frame + n_frames - 1}")

    n_kp = len(stac_data.kp_names)
    wrist_kp_idx = stac_data.kp_names.index("Wrist")

    # Load raw 2D for the 5 keypoints from points3d (SEW) and digits (d2_mid, d5_mid)
    with h5py.File(args.raw_points3d) as f:
        raw3_2d = f["points_2d"][:, video_start_frame:video_start_frame + n_frames]   # (cams, T, 3, 2)
        n3 = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    with h5py.File(args.digits_points3d) as f:
        dig_2d = f["points_2d"][:, video_start_frame:video_start_frame + n_frames]
        nd = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]

    iSh = n3.index("right_shoulder"); iEl = n3.index("right_elbow"); iWr = n3.index("right_wrist")
    iC = nd.index("right_d2_mid"); iL = nd.index("right_d5_mid")
    # Stack raw 2D to (cams, T, 5kp, 2) in same order as stac_data.kp_names
    raw_2d = np.full((len(cameras), n_frames, 5, 2), np.nan)
    for c in range(len(cameras)):
        raw_2d[c, :, 0] = raw3_2d[c, :, iSh]
        raw_2d[c, :, 1] = raw3_2d[c, :, iEl]
        raw_2d[c, :, 2] = raw3_2d[c, :, iWr]
        raw_2d[c, :, 3] = dig_2d[c, :, iC]
        raw_2d[c, :, 4] = dig_2d[c, :, iL]

    # Set up MuJoCo cameras (use fixed shoulder ref)
    shoulder_frame = reg.shoulder_reference_mm if reg.shoulder_reference_mm is not None else raw_shoulders[video_start_frame]
    mj_cam_params = []
    for cam_params in cameras:
        cp = _compute_calibration_cam_xyaxes(cam_params, reg, shoulder_frame)
        cp["name"] = cam_params.name
        mj_cam_params.append(cp)

    cfg = OmegaConf.create({
        "model": {
            "MJCF_PATH": "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v17.xml",
            "KP_NAMES": list(stac_data.kp_names),
            "KEYPOINT_MODEL_PAIRS": {
                "Shoulder": "humerus", "Elbow": "ulna", "Wrist": "wrist",
                "KnuckleC": "wrist", "KnuckleL": "wrist",
            },
            "KEYPOINT_INITIAL_OFFSETS": {
                "Shoulder": "0 0 0", "Elbow": "0 0 0", "Wrist": "0 0 0",
                "KnuckleC": "-0.0018 0.0032 -0.0009",
                "KnuckleL": "-0.0001 0.0029 -0.0019",
            },
            "KEYPOINT_COLOR_PAIRS": {
                "Shoulder": "1 0 0 1", "Elbow": "1 0.6 0 1", "Wrist": "1 1 0 1",
                "KnuckleC": "0 1 1 1", "KnuckleL": "1 0 1 1",
            },
            "KEYPOINT_INITIAL_OFFSET_AS_LIST": False,
        }
    })
    mj_model, mj_data = _setup_mujoco_model(cfg, stac_data.kp_names, mj_cam_params)
    _set_green_background(mj_model)
    scene_option = _make_scene_option()
    h, w = args.render_h, args.render_w

    cam_name_to_idx = {}
    for cam_params in cameras:
        for i in range(mj_model.ncam):
            if f"calib_{cam_params.name}" in mj_model.camera(i).name:
                cam_name_to_idx[cam_params.name] = i
                break

    renderers = {c.name: mujoco.Renderer(mj_model, height=h, width=w) for c in cameras}

    kp_colors = []
    for name in stac_data.kp_names:
        rgba = [float(x) for x in cfg.model.KEYPOINT_COLOR_PAIRS[name].split()]
        kp_colors.append((int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255)))

    video_files = {c.name: str(Path(args.video_dir) / c.name / f"{Path(args.video_dir).name}_{c.name}_resample.mp4") for c in cameras}
    caps = {c.name: cv2.VideoCapture(video_files[c.name]) for c in cameras}
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    print(f"  Rendering {n_frames} frames × {len(cameras)} cameras (with 2D alignment shift)...")
    with imageio.get_writer(args.output, fps=args.fps) as writer:
        for frame_idx in tqdm(range(n_frames)):
            abs_frame = video_start_frame + frame_idx
            shoulder_calib = raw_shoulders[abs_frame]
            inv_shoulder = reg.shoulder_reference_mm if reg.shoulder_reference_mm is not None else shoulder_calib

            # Project model wrist (= marker_sites Wrist) to each cam (UNSHIFTED, using current inverse)
            ms_mj = marker_sites[frame_idx]
            ms_calib = reg.mujoco_to_calibration(ms_mj, inv_shoulder)

            panels = []
            for c, cam_params in enumerate(cameras):
                ret, frame = caps[cam_params.name].read()
                if not ret:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    frame = cam_params.undistort_frame(frame)
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Render mujoco model
                cam_idx = cam_name_to_idx[cam_params.name]
                model_rgba = _render_model_frame(
                    mj_model, mj_data, renderers[cam_params.name],
                    qposes[frame_idx], cam_idx, scene_option,
                )

                # 2D alignment shift: raw SLEAP wrist - projected model wrist
                ms_2d = cam_params.project_points(ms_calib)   # (5, 2)
                model_W_2d = ms_2d[wrist_kp_idx]
                raw_W_2d = raw_2d[c, frame_idx, wrist_kp_idx]
                if np.isnan(raw_W_2d).any():
                    shift = np.zeros(2)
                else:
                    shift = raw_W_2d - model_W_2d   # (dx, dy) px

                # Apply shift to model_rgba (translate the layer)
                M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                model_rgba_shifted = cv2.warpAffine(
                    model_rgba, M, (w, h),
                    flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0),
                )
                frame = _composite(frame, model_rgba_shifted, alpha_scale=0.5)

                # Env markers (on raw image — these are world-fixed calibration points,
                # not affected by the shoulder mismatch)
                env_points = env_data["points"]
                env_names = env_data["names"]
                frame = _draw_env_markers(frame, env_points, env_names, cam_params)

                # MOCAP dots — use raw points_2d directly (always correct)
                for ki in range(n_kp):
                    p = raw_2d[c, frame_idx, ki]
                    if not np.isnan(p).any():
                        cv2.circle(frame, (int(p[0]), int(p[1])), 5, kp_colors[ki], 2)
                # STAC X marks — projected marker_sites + same 2D shift (so they overlay the shifted model)
                for ki in range(n_kp):
                    p = ms_2d[ki] + shift
                    cv2.drawMarker(frame, (int(p[0]), int(p[1])), kp_colors[ki],
                                    markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

                # Annotation
                cv2.putText(frame, f"{cam_params.name} f={abs_frame} shift=({shift[0]:+.0f},{shift[1]:+.0f})px",
                             (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                panels.append(frame)

            row = np.concatenate(panels, axis=1)
            writer.append_data(cv2.cvtColor(row, cv2.COLOR_BGR2RGB))

    for cap in caps.values():
        cap.release()
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
