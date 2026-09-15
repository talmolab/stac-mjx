#!/usr/bin/env python
"""Register js_luc3d_hand mocap (arm + hand only, no joystick) onto v24.

Same recipe as register_v22_mocap.py (see that file's docstring for the full
rationale on tree-rescale removal, per-trial-own-rotation vs shared-rotation
averaging, and uniform scale correction) -- differences:

  - MODEL_PATH points at the STAC-safe v24 model (built by
    build_v24_stac_xml.py), whose rest pose differs from v21/v22's (a
    completely separate Blender export, not a v22 derivative) -- so, as in
    v22's own script, model_rest_targets() re-derives the Procrustes
    alignment target fresh from v24's OWN rest-pose (qpos=0) forward
    kinematics rather than transplanting any prior version's registration.
  - Body names differ (humerus_right/ulna_right/N_L_C_right instead of
    humerus/ulna/wrist -- see build_v24_stac_xml.py's docstring).
  - No joystick handling at all: v24 has no joystick body, and Eric only
    wants shoulder/arm/wrist/digits for this smoke test (joystick is coming
    back in a later version). js_base/js_ball columns are dropped from the
    raw (T,1,19,3) points3d_refined.h5 entirely rather than registered.
  - Restricted to only the 6 trials Eric picked (same 6 used for the
    parallel v23 RL-training effort in vnl-playground), not a full
    session-glob -- see TRIAL_NAMES below.

Per trial <session>/<trial>_points3d_refined.h5 ("tracks", shape (T,1,19,3),
mm, calibration frame) -> writes <trial>_registered_v24_refined.h5 ("tracks",
meters, v24 world frame, 17-keypoint arm+hand order) with attrs matching what
stac_mjx/multicam.py::load_registration_transform expects to read back.

17-keypoint order (first 17 of the raw 19-keypoint js_luc3d_hand order --
see register_v22_mocap.py's docstring for how that 19-order was confirmed):
  shoulder, elbow, Wrist, d1_k, d1_t, d2_k, d2_m, d2_t, d3_k, d3_m, d3_t,
  d4_k, d4_m, d4_t, d5_k, d5_m, d5_t
"""
from __future__ import annotations

import json
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

# Eric's pick for v24 (same 6 trials as the parallel v23 RL-training subset).
TRIAL_NAMES = [
    "CFL_35_20240128_trial_0101",
    "CFL_35_20240128_trial_0201",
    "CFL_35_20240128_trial_0301",
    "CFL_35_20240128_trial_0401",
    "CFL_37_20240423_trial_0301",
    "CFL_37_20240423_trial_0401",
]

KP_NAMES = ["shoulder", "elbow", "Wrist", "d1_k", "d1_t", "d2_k", "d2_m", "d2_t",
            "d3_k", "d3_m", "d3_t", "d4_k", "d4_m", "d4_t", "d5_k", "d5_m", "d5_t"]
N_KP = len(KP_NAMES)
IDX = {n: i for i, n in enumerate(KP_NAMES)}
ARM_HAND_KP = KP_NAMES  # all 17 -- no joystick columns kept at all

TREE_EDGES = [
    ("shoulder", "elbow"), ("elbow", "Wrist"),
    ("Wrist", "d1_k"), ("Wrist", "d2_k"), ("Wrist", "d3_k"), ("Wrist", "d4_k"), ("Wrist", "d5_k"),
    ("d1_k", "d1_t"),
    ("d2_k", "d2_m"), ("d2_m", "d2_t"),
    ("d3_k", "d3_m"), ("d3_m", "d3_t"),
    ("d4_k", "d4_m"), ("d4_m", "d4_t"),
    ("d5_k", "d5_m"), ("d5_m", "d5_t"),
]

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
    scale factor and Procrustes rotation against v24's rest pose."""
    with h5py.File(points3d_h5, "r") as f:
        raw = f["tracks"][:]
    if raw.shape[1] != 1 or raw.shape[2] < N_KP or raw.shape[3] != 3:
        raise ValueError(f"unexpected tracks shape {raw.shape}, expected (T,1,>={N_KP},3)")
    # Drop js_base/js_ball (indices 17,18 in the raw 19-keypoint order) --
    # not used for v24's arm/hand-only smoke test.
    kps_mm = raw[:, 0, :N_KP, :].astype(np.float64)
    T = kps_mm.shape[0]
    nan_frac = np.isnan(kps_mm).any(axis=(1, 2)).mean()

    filled = fill_nans(kps_mm)
    smoothed = smooth(filled)
    rest_idx = find_resting_window(smoothed)

    shoulder_ref_mm = np.median(smoothed[rest_idx, IDX["shoulder"], :], axis=0)
    centered_m = (smoothed - shoulder_ref_mm[None, None, :]) / 1000.0

    # per-trial uniform scale correction (see register_v22_mocap.py's
    # prepare_trial docstring for the full rationale) -- a single scalar
    # referenced against the long, stable shoulder->elbow->Wrist chain.
    model_arm_span = (np.linalg.norm(targets["elbow"] - targets["shoulder"])
                       + np.linalg.norm(targets["Wrist"] - targets["elbow"]))
    trial_arm_span = np.median(
        np.linalg.norm(centered_m[rest_idx, IDX["elbow"], :] - centered_m[rest_idx, IDX["shoulder"], :], axis=1)
        + np.linalg.norm(centered_m[rest_idx, IDX["Wrist"], :] - centered_m[rest_idx, IDX["elbow"], :], axis=1)
    )
    scale_factor = model_arm_span / trial_arm_span
    scaled_m = centered_m * scale_factor

    model_centered = np.array([targets[kp] - targets["shoulder"] for kp in ARM_HAND_KP])
    median_resting = np.median(scaled_m[rest_idx], axis=0)
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
    register_trial docstring for why per-trial-own beats shared-averaged)."""
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

    print(f"[{trial_name}] T={T} nan_frac={nan_frac:.3f} "
          f"rest_idx=[{rest_idx[0]}..{rest_idx[-1]}] (n={len(rest_idx)}) "
          f"scale_factor={prep['scale_factor']:.4f} "
          f"max_bone_len_err={max_len_err*1e6:.3f}um")

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
        f.attrs["max_bone_length_error_m"] = max_len_err
        f.attrs["model_path"] = str(MODEL_PATH)
        f.attrs["recipe"] = "register_v24_mocap_own_rotation_arm_hand_only_v1"
        for (p, c), L in target_lengths.items():
            f.attrs[f"target_length_{p}_to_{c}"] = L

    return {
        "trial": trial_name, "T": T, "nan_frac": nan_frac,
        "rest_idx": rest_idx, "max_len_err": max_len_err,
    }


def main():
    targets = model_rest_targets(MODEL_PATH)
    target_lengths = segment_lengths(targets)
    print("v24 model target segment lengths (mm):")
    for (p_, c_), L in target_lengths.items():
        print(f"  {p_:10s} -> {c_:8s}: {L*1000:.4f} mm")
    print(f"MODEL shoulder (world, m): {targets['shoulder']}")
    print()

    results = []
    for trial_name in TRIAL_NAMES:
        trial_dir = TRAINING_ROOT / trial_name
        points3d = trial_dir / f"{trial_name}_points3d_refined.h5"
        if not points3d.exists():
            print(f"[skip] no _points3d_refined.h5 in {trial_name}")
            continue
        prep = prepare_trial(points3d, targets)
        out_h5 = trial_dir / f"{trial_name}_registered_v24_refined.h5"
        res = register_trial(prep, out_h5, targets, target_lengths, trial_name=trial_name)
        results.append(res)

    print(f"\nRegistered {len(results)}/{len(TRIAL_NAMES)} trials.")


if __name__ == "__main__":
    main()
