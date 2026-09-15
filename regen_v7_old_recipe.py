#!/usr/bin/env python
"""Regenerate v7 mocap registration using the OLD recipe.

OLD recipe (matches the working test_reprojection_trial01_v16_fixed_ref.mp4):
  shoulder_reference_mm = first-frame smoothed shoulder position (NOT median of resting).

The buggy NEW recipe in azim-emg-data/register_to_mujoco.py used median of low-velocity
resting frames over a 1000-frame window, which produced a different Procrustes
rotation (causing model to mirror-flip in some cameras).

Steps:
  1. Smooth raw mocap (savgol w=15, p=3)
  2. shoulder_ref_mm = first frame's shoulder (post-smoothing)
  3. Center on shoulder_ref, convert to meters
  4. Rescale bones to MODEL bone lengths
  5. Procrustes-align median of first 20 frames to MODEL_CENTERED
  6. Apply rotation, translate to MODEL_SHOULDER

Writes <session>_mujoco_registered_forelimb_right_v7_oldrecipe.h5.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import mujoco
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.signal import savgol_filter

MODEL_PATH = (
    "/mnt/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/"
    "mouse_forelimb_right.xml"
)


def gravity_settled_pose(model_path: str, settle_time: float = 2.0):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    humerus_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus")
    ulna_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    data.qpos[:] = 0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    n_steps = int(settle_time / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    return (
        data.xpos[humerus_id].copy(),
        data.xpos[ulna_id].copy(),
        data.xpos[wrist_id].copy(),
    )


def smooth_positions(positions: np.ndarray, window: int = 15, polyorder: int = 3):
    T = positions.shape[0]
    if T < window:
        window = max(polyorder + 2, (T // 2) * 2 + 1)
        if window < polyorder + 2:
            return positions
    out = np.copy(positions)
    for kp in range(positions.shape[1]):
        for d in range(3):
            out[:, kp, d] = savgol_filter(positions[:, kp, d], window, polyorder)
    return out


def env_anchored_rotation(session_env_h5: Path, ref_env_h5: Path, ref_R: np.ndarray):
    """Derive the registration rotation by Procrustes-aligning this session's env
    points to a reference session's env points, then chaining with the reference R.

    This bypasses the unreliable "mocap resting median" Procrustes (which fails
    when the mouse moves at the start of the session). The rig (bars, water spout)
    is physically fixed across sessions, so env-to-env alignment recovers the
    relative camera-frame rotation between sessions, and chaining with the
    reference R yields a registration rotation in the same coordinate convention.
    """
    with h5py.File(session_env_h5, "r") as f:
        sess_pts = f["points_3d"][:].squeeze()
    with h5py.File(ref_env_h5, "r") as f:
        ref_pts = f["points_3d"][:].squeeze()
    A = sess_pts - sess_pts.mean(0)
    B = ref_pts - ref_pts.mean(0)
    R_sess_to_ref, _ = orthogonal_procrustes(A, B)
    if np.linalg.det(R_sess_to_ref) < 0:
        U, S, Vt = np.linalg.svd(A.T @ B)
        Vt[-1] *= -1
        R_sess_to_ref = U @ Vt
    err = np.linalg.norm(A @ R_sess_to_ref - B, axis=1)
    print(f"  env Procrustes: mean err {err.mean():.2f} mm, max {err.max():.2f} mm")
    R_total = R_sess_to_ref @ ref_R
    return R_total, R_sess_to_ref


def register_session(points3d_h5: Path, output_h5: Path, model_path: str = MODEL_PATH,
                     anchor_env_h5: Path = None, anchor_ref_env_h5: Path = None,
                     anchor_ref_R: np.ndarray = None):
    sh, el, wr = gravity_settled_pose(model_path)
    H_LEN = float(np.linalg.norm(el - sh))
    F_LEN = float(np.linalg.norm(wr - el))
    MODEL_CENTERED = np.array([[0.0, 0.0, 0.0], el - sh, wr - sh])
    print(f"[{points3d_h5.name}]")
    print(f"  Model bones (mm): humerus={H_LEN*1000:.3f}, forearm={F_LEN*1000:.3f}")
    print(f"  MODEL_SHOULDER (m): {sh}")

    with h5py.File(points3d_h5, "r") as f:
        raw = f["tracks"][:]
    if raw.ndim != 4 or raw.shape[1] != 1 or raw.shape[2] != 3 or raw.shape[3] != 3:
        raise ValueError(f"Unexpected tracks shape: {raw.shape}")
    T = raw.shape[0]
    kps_mm = raw[:, 0, :, :].astype(np.float64)  # (T, 3, 3) — assumed mm
    print(f"  raw frames: {T}, NaN frac: {np.isnan(kps_mm).any(axis=(1,2)).mean():.3f}")

    # Linear-interp NaN holes per coord/kp (savgol can't handle NaN)
    kps_filled = kps_mm.copy()
    for k in range(3):
        for d in range(3):
            arr = kps_filled[:, k, d]
            nans = np.isnan(arr)
            if nans.any():
                arr[nans] = np.interp(
                    np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans]
                )
                kps_filled[:, k, d] = arr

    # 1. Smooth (savgol w=15, p=3)
    sm_mm = smooth_positions(kps_filled, window=15, polyorder=3)

    # 2. Find the first sustained low-velocity window (true resting period).
    #    Otherwise the first frame may be mid-motion (e.g. AT006).
    from scipy.ndimage import uniform_filter1d
    n_frames = sm_mm.shape[0]
    vel = np.zeros(n_frames)
    vel[1:] = (
        np.linalg.norm(np.diff(sm_mm[:, 1, :], axis=0), axis=1)
        + np.linalg.norm(np.diff(sm_mm[:, 2, :], axis=0), axis=1)
    )
    vel_smooth = uniform_filter1d(vel, size=21)

    # Search first 500 frames for low-velocity frames
    search_window = min(500, n_frames)
    early_vel = vel_smooth[:search_window]
    # Threshold: 25th percentile of velocities in this window — robust to whether
    # the start is moving or not.
    thr = np.percentile(early_vel, 25)
    rest_mask = early_vel <= thr
    rest_idx = np.where(rest_mask)[0]
    if len(rest_idx) < 20:
        # fallback: 20 lowest-velocity frames
        rest_idx = np.argsort(early_vel)[:20]
    rest_idx = np.sort(rest_idx)
    print(f"  Resting frames in first {search_window}: {len(rest_idx)} "
          f"(vel mean {vel_smooth[rest_idx].mean():.3f}, threshold {thr:.3f}, "
          f"vs early-window mean {early_vel.mean():.3f})")
    print(f"    indices range: {rest_idx[0]}..{rest_idx[-1]}")

    # shoulder_reference_mm = median shoulder over true resting frames
    shoulder_ref_mm = np.median(sm_mm[rest_idx, 0, :], axis=0)
    print(f"  shoulder_reference_mm (median of resting): {shoulder_ref_mm}")

    # 3. Center and convert to meters
    centered_m = (sm_mm - shoulder_ref_mm[None, None, :]) / 1000.0

    # 4. Rescale bones to model lengths (preserve direction; per frame)
    shoulder_per_frame = centered_m[:, 0, :]
    h_vec = centered_m[:, 1, :] - shoulder_per_frame
    f_vec = centered_m[:, 2, :] - centered_m[:, 1, :]
    hn = np.maximum(np.linalg.norm(h_vec, axis=1, keepdims=True), 1e-10)
    fn = np.maximum(np.linalg.norm(f_vec, axis=1, keepdims=True), 1e-10)
    new_elbow = shoulder_per_frame + (h_vec / hn) * H_LEN
    new_wrist = new_elbow + (f_vec / fn) * F_LEN
    rescaled = np.stack([shoulder_per_frame, new_elbow, new_wrist], axis=1)

    # 5. Compute rotation R: prefer env-anchored alignment when available
    if anchor_env_h5 is not None and anchor_ref_env_h5 is not None and anchor_ref_R is not None:
        R, R_sess_to_ref = env_anchored_rotation(anchor_env_h5, anchor_ref_env_h5, anchor_ref_R)
        print(f"  env-anchored rotation, det(R)={np.linalg.det(R):+.4f}")
    else:
        # Fallback: Procrustes-align median of TRUE resting frames to MODEL_CENTERED
        median_resting = np.nanmedian(rescaled[rest_idx], axis=0)
        R, _ = orthogonal_procrustes(median_resting, MODEL_CENTERED)
        if np.linalg.det(R) < 0:
            U, S, Vt = np.linalg.svd(median_resting.T @ MODEL_CENTERED)
            Vt[-1, :] *= -1
            R = U @ Vt
        print(f"  mocap-Procrustes on {len(rest_idx)} resting frames, det(R)={np.linalg.det(R):+.4f}")

    # 6. Apply rotation, translate to MODEL_SHOULDER
    aligned = np.einsum("fkd,dj->fkj", rescaled, R)
    registered = aligned + sh[None, None, :]

    final_h = np.linalg.norm(registered[:, 1] - registered[:, 0], axis=1).mean()
    final_f = np.linalg.norm(registered[:, 2] - registered[:, 1], axis=1).mean()
    print(f"  registered bones (mm): humerus={final_h*1000:.3f}, forearm={final_f*1000:.3f}")

    out = registered[:, None, :, :]
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("tracks", data=out, compression="gzip", compression_opts=4)
        f.attrs["registration_method"] = "fixed_shoulder_reference_with_bone_rescaling"
        f.attrs["registration_rotation"] = R
        f.attrs["registration_translation"] = sh
        f.attrs["shoulder_reference_mm"] = shoulder_ref_mm
        f.attrs["smoothing_polyorder"] = 3
        f.attrs["smoothing_window"] = 15
        f.attrs["target_forearm_len"] = F_LEN
        f.attrs["target_humerus_len"] = H_LEN
        f.attrs["model_path"] = str(model_path)
        f.attrs["landmark_names"] = ["humerus (shoulder)", "ulna (elbow)", "wrist body"]
        f.attrs["kp_names"] = ["Shoulder", "Elbow", "Wrist"]
        f.attrs["recipe"] = "first_frame_shoulder_ref_v7"
    print(f"  -> {output_h5}")
    return output_h5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/vast/eric/azim-emg-data")
    p.add_argument(
        "--events", nargs="*",
        default=[
            "AT006_2024-02-28_15-39-40_LightOff_video",
            "AT009_2024-04-25_13-50-34_LightOff_video",
            "AT012_2024-06-21_12-32-30_LightOff_video",
            "AT013_2024-06-20_13-41-20_LightOff_video",
        ],
    )
    p.add_argument(
        "--anchor-ref-env",
        default="/mnt/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5",
        help="Reference session's env_points3d.h5 (used as world-frame anchor).",
    )
    p.add_argument(
        "--anchor-ref-v7",
        default="/mnt/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5",
        help="Reference session's working v7.h5 (provides reference rotation).",
    )
    p.add_argument(
        "--session-env-dir",
        default="/mnt/vast/niyo/stac-mjx/smoke_test_v16_reproj",
        help="Directory containing freshly-triangulated <sid>_env_points3d.h5 per session.",
    )
    p.add_argument(
        "--no-anchor", action="store_true",
        help="Skip env-anchoring; fall back to mocap-resting Procrustes.",
    )
    args = p.parse_args()

    # Load reference R from the reference v7
    anchor_ref_R = None
    anchor_ref_env_h5 = None
    if not args.no_anchor:
        with h5py.File(args.anchor_ref_v7, "r") as f:
            anchor_ref_R = np.array(f.attrs["registration_rotation"])
        anchor_ref_env_h5 = Path(args.anchor_ref_env)
        print(f"[anchor] ref env: {anchor_ref_env_h5}")
        print(f"[anchor] ref R from: {args.anchor_ref_v7}")

    for ev in args.events:
        ed = Path(args.root) / ev
        in_h5 = ed / f"{ev}_points3d.h5"
        out_h5 = ed / f"{ev}_mujoco_registered_forelimb_right_v7_oldrecipe.h5"
        if not in_h5.exists():
            print(f"[skip] no points3d for {ev}")
            continue
        # Find this session's env points
        sid_short = ev.replace("_video", "")
        anchor_env_h5 = Path(args.session_env_dir) / f"{sid_short}_env_points3d.h5"
        if not args.no_anchor and not anchor_env_h5.exists():
            print(f"[warn] no env points for {ev} at {anchor_env_h5}; falling back to mocap-Procrustes")
            anchor_env_h5 = None
        register_session(
            in_h5, out_h5,
            anchor_env_h5=anchor_env_h5 if not args.no_anchor else None,
            anchor_ref_env_h5=anchor_ref_env_h5,
            anchor_ref_R=anchor_ref_R,
        )


if __name__ == "__main__":
    main()
