#!/usr/bin/env python
"""Register js_luc3d_hand mocap (arm + hand + joystick) onto v24's rig, for v25.

v25 = v24's arm/hand/neck/head body (build_v24_stac_xml.py's STAC-safe copy)
plus a joystick body added in the same style v22 used. register_v24_mocap.py
already does the arm/hand half of this (v24 body names, v24's own rest-pose
Procrustes target) but explicitly DROPS js_base/js_ball (its own docstring:
"joystick is coming back in a later version"). This script is that later
version: identical recipe to register_v24_mocap.py (itself copied from
register_v22_mocap.py's v5 "own rotation per trial" recipe -- see that file's
docstring for the full rationale on why per-trial-own-rotation beats a
shared/averaged rotation), except it KEEPS all 19 raw keypoint columns instead
of slicing to 17, so js_base/js_ball get carried through the exact same
rotation+scale+translation as the arm/hand keypoints in every frame.

Registers all 15 CFL_35/36/37 trials (not just the 6-trial v24 smoke subset)
so compute_joystick_geometry_v25.py has as many trials as v22's original 12
to average a robust rotation from, matching v22's own precedent.

Per trial <session>/<trial>_points3d_refined.h5 ("tracks", shape (T,1,19,3),
mm, calibration frame) -> writes <trial>_registered_v25_refined.h5 ("tracks",
meters, v24/v25 world frame, 19-keypoint order) with attrs matching what
compute_joystick_geometry_v22.py's recipe (and stac_mjx/multicam.py) expect.

19-keypoint order (same as register_v22_mocap.py's, confirmed via
merge_arm_hand.py's construction + points3d.h5's n_nodes=19):
  shoulder, elbow, Wrist, d1_k, d1_t, d2_k, d2_m, d2_t, d3_k, d3_m, d3_t,
  d4_k, d4_m, d4_t, d5_k, d5_m, d5_t, js_base, js_ball
"""
from __future__ import annotations

from pathlib import Path

import os
os.environ.setdefault("MUJOCO_GL", "disable")

import h5py
import mujoco
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

MODEL_PATH = "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v24_stac.xml"
TRAINING_ROOT = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData")

KP_NAMES = ["shoulder", "elbow", "Wrist", "d1_k", "d1_t", "d2_k", "d2_m", "d2_t",
            "d3_k", "d3_m", "d3_t", "d4_k", "d4_m", "d4_t", "d5_k", "d5_m", "d5_t",
            "js_base", "js_ball"]
N_KP = len(KP_NAMES)
IDX = {n: i for i, n in enumerate(KP_NAMES)}
ARM_HAND_KP = KP_NAMES[:17]  # everything except js_base/js_ball -- the Procrustes fit set

TREE_EDGES = [
    ("shoulder", "elbow"), ("elbow", "Wrist"),
    ("Wrist", "d1_k"), ("Wrist", "d2_k"), ("Wrist", "d3_k"), ("Wrist", "d4_k"), ("Wrist", "d5_k"),
    ("d1_k", "d1_t"),
    ("d2_k", "d2_m"), ("d2_m", "d2_t"),
    ("d3_k", "d3_m"), ("d3_m", "d3_t"),
    ("d4_k", "d4_m"), ("d4_m", "d4_t"),
    ("d5_k", "d5_m"), ("d5_m", "d5_t"),
]

# v24 body names carry an explicit "_right" suffix (humerus_right, ulna_right)
# and the wrist-equivalent is N_L_C_right (fused carpal complex) -- see
# build_v24_stac_xml.py / register_v24_mocap.py.
KP_TARGET_SPEC = {
    "shoulder": ("humerus_right", None),
    "elbow": ("ulna_right", None),
    "Wrist": ("N_L_C_right", None),
    "d1_k": ("Phalanx_hand_1_1_right", None),
    "d1_t": ("Phalanx_hand_2_1_right", np.array([0, 0.00066, 0])),
    "d2_k": ("Phalanx_hand_1_2_right", None),
    "d2_m": ("Phalanx_hand_2_2_right", None),
    "d2_t": ("Phalanx_hand_3_2_right", np.array([0, 0.00127, 0])),
    "d3_k": ("Phalanx_hand_1_3_right", None),
    "d3_m": ("Phalanx_hand_2_3_right", None),
    "d3_t": ("Phalanx_hand_3_3_right", np.array([0, 0.00146, 0])),
    "d4_k": ("Phalanx_hand_1_4_right", None),
    "d4_m": ("Phalanx_hand_2_4_right", None),
    "d4_t": ("Phalanx_hand_3_4_right", np.array([0, 0.00141, 0])),
    "d5_k": ("Phalanx_hand_1_5_right", None),
    "d5_m": ("Phalanx_hand_2_5_right", None),
    "d5_t": ("Phalanx_hand_3_5_right", np.array([0, 0.00098, 0])),
}


def model_rest_targets(model_path):
    """Rest-pose (qpos=0) world positions for every keypoint's model target."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    targets = {}
    for kp, (body, offset) in KP_TARGET_SPEC.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        xpos = d.xpos[bid].copy()
        if offset is not None:
            xmat = d.xmat[bid].reshape(3, 3)
            xpos = xpos + xmat @ offset
        targets[kp] = xpos
    return targets


def segment_lengths(targets):
    return {(p, c): float(np.linalg.norm(targets[c] - targets[p])) for p, c in TREE_EDGES}


def fill_nans(kps_mm):
    filled = kps_mm.copy()
    for k in range(filled.shape[1]):
        for dim in range(3):
            arr = filled[:, k, dim]
            nans = np.isnan(arr)
            if nans.all():
                raise ValueError(f"keypoint {KP_NAMES[k]} dim {dim} is all-NaN")
            if nans.any():
                arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
                filled[:, k, dim] = arr
    return filled


def smooth(filled_mm, window=15, polyorder=3):
    T = filled_mm.shape[0]
    w = window if T >= window else max(polyorder + 2, (T // 2) * 2 + 1)
    out = filled_mm.copy()
    for k in range(filled_mm.shape[1]):
        for dim in range(3):
            out[:, k, dim] = savgol_filter(filled_mm[:, k, dim], w, polyorder)
    return out


def find_resting_window(smoothed_mm, search_frames=60):
    """Low-velocity window (Wrist+Elbow combined speed) within the first
    `search_frames` frames -- these CFL trials are short (126 frames total,
    a single reach), so "resting" is expected near the start."""
    T = smoothed_mm.shape[0]
    search_frames = min(search_frames, T)
    wrist_i, elbow_i = IDX["Wrist"], IDX["elbow"]
    vel = np.zeros(T)
    vel[1:] = (np.linalg.norm(np.diff(smoothed_mm[:, wrist_i, :], axis=0), axis=1)
               + np.linalg.norm(np.diff(smoothed_mm[:, elbow_i, :], axis=0), axis=1))
    vel_smooth = uniform_filter1d(vel, size=max(3, min(21, T // 4)))
    early = vel_smooth[:search_frames]
    thr = np.percentile(early, 25)
    idx = np.where(early <= thr)[0]
    if len(idx) < 10:
        idx = np.argsort(early)[:10]
    return np.sort(idx)


def prepare_trial(points3d_h5: Path, targets):
    """Load + fill/smooth/find-resting-window for one trial, compute its own
    scale factor and Procrustes rotation against v24's rest pose (fit on the
    17 arm/hand markers only -- js_base/js_ball ride along on the same
    rotation+scale below, not part of the fit set, same as register_v22_mocap.py)."""
    with h5py.File(points3d_h5, "r") as f:
        raw = f["tracks"][:]
    if raw.shape[1:] != (1, N_KP, 3):
        raise ValueError(f"unexpected tracks shape {raw.shape}, expected (T,1,{N_KP},3)")
    kps_mm = raw[:, 0, :, :].astype(np.float64)  # keep all 19 columns, incl. js_base/js_ball
    T = kps_mm.shape[0]
    nan_frac = np.isnan(kps_mm).any(axis=(1, 2)).mean()

    filled = fill_nans(kps_mm)
    smoothed = smooth(filled)
    rest_idx = find_resting_window(smoothed)

    shoulder_ref_mm = np.median(smoothed[rest_idx, IDX["shoulder"], :], axis=0)
    centered_m = (smoothed - shoulder_ref_mm[None, None, :]) / 1000.0

    model_arm_span = (np.linalg.norm(targets["elbow"] - targets["shoulder"])
                       + np.linalg.norm(targets["Wrist"] - targets["elbow"]))
    trial_arm_span = np.median(
        np.linalg.norm(centered_m[rest_idx, IDX["elbow"], :] - centered_m[rest_idx, IDX["shoulder"], :], axis=1)
        + np.linalg.norm(centered_m[rest_idx, IDX["Wrist"], :] - centered_m[rest_idx, IDX["elbow"], :], axis=1)
    )
    scale_factor = model_arm_span / trial_arm_span
    scaled_m = centered_m * scale_factor  # uniform -- affects arm/hand/joystick identically

    model_centered = np.array([targets[kp] - targets["shoulder"] for kp in ARM_HAND_KP])
    median_resting = np.median(scaled_m[rest_idx][:, :17, :], axis=0)
    R, _ = orthogonal_procrustes(median_resting, model_centered)
    if np.linalg.det(R) < 0:
        U, S, Vt = np.linalg.svd(median_resting.T @ model_centered)
        Vt[-1] *= -1
        R = U @ Vt

    return {
        "T": T, "nan_frac": nan_frac, "rest_idx": rest_idx,
        "shoulder_ref_mm": shoulder_ref_mm, "scaled_m": scaled_m,
        "scale_factor": scale_factor, "own_rotation": R,
    }


def register_trial(prep, output_h5: Path, targets, target_lengths, trial_name=""):
    """Register one trial's triangulated mocap into v24 model/world frame,
    using its own independently-fit rotation (see register_v22_mocap.py's
    register_trial docstring for why per-trial-own beats shared-averaged).
    js_ball is stabilized off js_base's per-frame position plus a fixed
    early-frame offset, same rationale/window as register_v22_mocap.py
    (raw per-frame js_ball tracking is unreliable under hand occlusion)."""
    T = prep["T"]
    nan_frac = prep["nan_frac"]
    rest_idx = prep["rest_idx"]
    shoulder_ref_mm = prep["shoulder_ref_mm"]
    scaled_m = prep["scaled_m"]
    R = prep["own_rotation"]

    aligned = np.einsum("fkd,dj->fkj", scaled_m, R)
    registered_m = aligned + targets["shoulder"][None, None, :]

    max_len_err = 0.0
    for (p, c), L in target_lengths.items():
        got = np.linalg.norm(registered_m[:, IDX[c], :] - registered_m[:, IDX[p], :], axis=1)
        max_len_err = max(max_len_err, float(np.abs(np.median(got) - L)))

    offset_window = np.arange(min(15, T))
    js_ball_offset = np.median(
        registered_m[offset_window, IDX["js_ball"], :] - registered_m[offset_window, IDX["js_base"], :], axis=0
    )
    registered_m[:, IDX["js_ball"], :] = registered_m[:, IDX["js_base"], :] + js_ball_offset[None, :]

    js_base_resting = np.median(registered_m[rest_idx, IDX["js_base"], :], axis=0)
    js_ball_resting = np.median(registered_m[rest_idx, IDX["js_ball"], :], axis=0)

    print(f"[{trial_name}] T={T} nan_frac={nan_frac:.3f} "
          f"rest_idx=[{rest_idx[0]}..{rest_idx[-1]}] (n={len(rest_idx)}) "
          f"scale_factor={prep['scale_factor']:.4f} "
          f"max_bone_len_err={max_len_err*1e6:.3f}um "
          f"js_base_resting={js_base_resting}")

    out = registered_m[:, None, :, :]
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("tracks", data=out, compression="gzip", compression_opts=4)
        f.attrs["registration_rotation"] = R
        f.attrs["registration_translation"] = targets["shoulder"]
        f.attrs["shoulder_reference_mm"] = shoulder_ref_mm
        f.attrs["scale_factor"] = prep["scale_factor"]
        f.attrs["kp_names"] = KP_NAMES
        f.attrs["resting_frame_indices"] = rest_idx
        f.attrs["js_base_resting_position"] = js_base_resting
        f.attrs["js_ball_resting_position"] = js_ball_resting
        f.attrs["js_ball_offset_from_base_m"] = js_ball_offset
        f.attrs["max_bone_length_error_m"] = max_len_err
        f.attrs["model_path"] = str(MODEL_PATH)
        f.attrs["recipe"] = "register_v25_mocap_own_rotation_v1_keeps_joystick"
        for (p, c), L in target_lengths.items():
            f.attrs[f"target_length_{p}_to_{c}"] = L

    return {
        "trial": trial_name, "T": T, "nan_frac": nan_frac,
        "rest_idx": rest_idx, "max_len_err": max_len_err,
        "js_base_resting": js_base_resting, "js_ball_resting": js_ball_resting,
    }


def main():
    targets = model_rest_targets(MODEL_PATH)
    target_lengths = segment_lengths(targets)
    print("v24 model target segment lengths (mm):")
    for (p_, c_), L in target_lengths.items():
        print(f"  {p_:10s} -> {c_:8s}: {L*1000:.4f} mm")
    print(f"MODEL shoulder (world, m): {targets['shoulder']}")
    print()

    trial_dirs = [d for d in sorted(TRAINING_ROOT.iterdir())
                  if d.is_dir() and d.name.startswith(("CFL_35", "CFL_36", "CFL_37"))]

    results = []
    for trial_dir in trial_dirs:
        points3d = trial_dir / f"{trial_dir.name}_points3d_refined.h5"
        if not points3d.exists():
            print(f"[skip] no _points3d_refined.h5 in {trial_dir.name}")
            continue
        prep = prepare_trial(points3d, targets)
        out_h5 = trial_dir / f"{trial_dir.name}_registered_v25_refined.h5"
        res = register_trial(prep, out_h5, targets, target_lengths, trial_name=trial_dir.name)
        results.append(res)

    print(f"\nRegistered {len(results)}/{len(trial_dirs)} trials.")
    js_base_all = np.array([r["js_base_resting"] for r in results])
    js_ball_all = np.array([r["js_ball_resting"] for r in results])
    print(f"js_base resting position across trials: mean={js_base_all.mean(0)}, "
          f"std={js_base_all.std(0)}, range={js_base_all.max(0) - js_base_all.min(0)}")
    print(f"js_ball resting position across trials: mean={js_ball_all.mean(0)}, "
          f"std={js_ball_all.std(0)}, range={js_ball_all.max(0) - js_ball_all.min(0)}")


if __name__ == "__main__":
    main()
