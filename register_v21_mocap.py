#!/usr/bin/env python
"""Register js_luc3d_hand mocap (arm + hand + joystick) onto v21.

Generalizes regen_v7_old_recipe.py's 2-segment (humerus/forearm) rescale
recipe to the full 16-segment arm+hand tree, and separately anchors the
joystick (js_base/js_ball) by rotation+translation only (no rescale, since
it's a fixed rig object, not a mouse bone).

Per trial <session>/<trial>_points3d.h5 ("tracks", shape (T,1,19,3), mm,
calibration frame) -> writes <trial>_registered_v21.h5 ("tracks", meters,
v21 world frame, same 19-keypoint order) with attrs matching what
stac_mjx/multicam.py::load_registration_transform expects to read back
(registration_rotation, registration_translation, shoulder_reference_mm),
plus provenance attrs (segment-length table, resting-window indices).

19-keypoint order (confirmed via merge_arm_hand.py's construction + node
count match against points3d.h5's n_nodes=19):
  shoulder, elbow, Wrist, d1_k, d1_t, d2_k, d2_m, d2_t, d3_k, d3_m, d3_t,
  d4_k, d4_m, d4_t, d5_k, d5_m, d5_t, js_base, js_ball
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

import h5py
import mujoco
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

MODEL_PATH = "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v21.xml"
TRAINING_ROOT = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData")

KP_NAMES = ["shoulder", "elbow", "Wrist", "d1_k", "d1_t", "d2_k", "d2_m", "d2_t",
            "d3_k", "d3_m", "d3_t", "d4_k", "d4_m", "d4_t", "d5_k", "d5_m", "d5_t",
            "js_base", "js_ball"]
N_KP = len(KP_NAMES)
IDX = {n: i for i, n in enumerate(KP_NAMES)}
ARM_HAND_KP = KP_NAMES[:17]  # everything except js_base/js_ball

# Parent->child edges forming the rescale tree (already in topological order:
# every edge's parent has been assigned before it's used as a parent itself).
# js_base/js_ball are NOT in this tree -- handled separately (rotate+translate only).
TREE_EDGES = [
    ("shoulder", "elbow"), ("elbow", "Wrist"),
    ("Wrist", "d1_k"), ("Wrist", "d2_k"), ("Wrist", "d3_k"), ("Wrist", "d4_k"), ("Wrist", "d5_k"),
    ("d1_k", "d1_t"),
    ("d2_k", "d2_m"), ("d2_m", "d2_t"),
    ("d3_k", "d3_m"), ("d3_m", "d3_t"),
    ("d4_k", "d4_m"), ("d4_m", "d4_t"),
    ("d5_k", "d5_m"), ("d5_m", "d5_t"),
]

# Model body (or body + local-frame site offset) target for each keypoint.
KP_TARGET_SPEC = {
    "shoulder": ("humerus", None),
    "elbow": ("ulna", None),
    "Wrist": ("wrist", None),
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


# Joystick keypoint targets on the MODEL side: NOT part of KP_TARGET_SPEC/the
# rescale tree (js_base/js_ball are a fixed rig object, not a mouse bone --
# see register_trial's rotate+translate-only handling), but needed for the
# model-vs-mocap comparison plot. Both offsets are in the "joystick" body's
# own local (tilted) frame (child of joystick_base), matching
# configs/model/mouse_arm_hand_joystick_v21.yaml's KEYPOINT_INITIAL_OFFSETS
# exactly -- measured from static_joystick_labels_withbottom.v001.slp via
# compute_joystick_geometry.py, not guessed from the shaft geom's size.
JOYSTICK_TARGET_SPEC = {
    "js_base": ("joystick", np.array([0, 0, 0.014363])),
    "js_ball": ("joystick", np.array([0, 0, 0.016184])),
}


def model_joystick_targets(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    targets = {}
    for kp, (body, offset) in JOYSTICK_TARGET_SPEC.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        xmat = d.xmat[bid].reshape(3, 3)
        targets[kp] = d.xpos[bid].copy() + xmat @ offset
    return targets


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
    a single reach), so "resting" is expected near the start, before the
    mouse initiates the reach."""
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


def tree_rescale(centered_m, target_lengths):
    """Rescale each tree edge's mocap vector to the model's segment length,
    direction preserved, walking the tree in TREE_EDGES' topological order.
    js_base/js_ball (not in the tree) pass through untouched.

    NOT used by register_trial anymore (kept for reference/diagnostics) --
    see register_trial's docstring for why: forcing every measured bone to
    the MODEL's assumed length distorts the (well-triangulated, per the
    calibration refinement work) real mocap data whenever the model's own
    proportions don't match the actual animal, which they often don't --
    e.g. measured digit-tip segments in CFL_36_20231220_trial_0001 are
    20-63% SHORTER than the model assumes, and Wrist->d1_k is 81% LONGER.
    Rescaling every segment to match a mismatched model moves knuckle/tip
    keypoints away from their real, correctly-triangulated positions,
    directly causing the hand to visibly stop matching the video."""
    rescaled = centered_m.copy()
    for p, c in TREE_EDGES:
        pi, ci = IDX[p], IDX[c]
        vec = centered_m[:, ci, :] - centered_m[:, pi, :]  # observed direction
        unit = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
        rescaled[:, ci, :] = rescaled[:, pi, :] + unit * target_lengths[(p, c)]
    return rescaled


def average_rotations(Rs):
    """Proper rotation average (SVD/chordal mean) of a list of rotation
    matrices -- same method used in compute_joystick_geometry.py."""
    Rsum = np.sum(Rs, axis=0)
    U, S, Vt = np.linalg.svd(Rsum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def prepare_trial(points3d_h5: Path, targets):
    """Load + fill/smooth/find-resting-window for one trial, compute its own
    (uncorrected) scale factor and Procrustes rotation. Split out from
    register_trial so main() can do two passes: (1) compute every reliable
    trial's own rotation to average into one shared, canonical rotation
    (see register_trial's docstring for why), then (2) actually register
    every trial using that shared rotation instead of its own noisy fit."""
    with h5py.File(points3d_h5, "r") as f:
        raw = f["tracks"][:]
    if raw.shape[1:] != (1, N_KP, 3):
        raise ValueError(f"unexpected tracks shape {raw.shape}, expected (T,1,{N_KP},3)")
    kps_mm = raw[:, 0, :, :].astype(np.float64)
    T = kps_mm.shape[0]
    nan_frac = np.isnan(kps_mm).any(axis=(1, 2)).mean()

    filled = fill_nans(kps_mm)
    smoothed = smooth(filled)
    rest_idx = find_resting_window(smoothed)

    shoulder_ref_mm = np.median(smoothed[rest_idx, IDX["shoulder"], :], axis=0)
    centered_m = (smoothed - shoulder_ref_mm[None, None, :]) / 1000.0

    # per-trial uniform scale correction: a single scalar (not per-bone, see
    # tree_rescale's docstring for why per-bone forcing is wrong) correcting
    # for gross calibration/triangulation scale error, referenced against a
    # long, stable bone chain (shoulder->elbow->Wrist). Bad bundle-adjustment
    # convergence in a trial's own calibration refinement (flagged trials,
    # e.g. CFL_35_trial_0001 ~0.46x, CFL_37_trial_0001 ~1.4x -- see
    # CALIBRATION_REFINEMENT_REPORT.md section 3) scales ALL triangulated
    # distances in that trial roughly uniformly, which showed up as the
    # whole arm looking too small/large once per-bone forced-rescaling
    # (which used to mask this) was removed. A single scale factor fixes
    # gross size without distorting relative hand/finger proportions.
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


def register_trial(prep, output_h5: Path, targets, target_lengths, shared_R, verbose=True, trial_name=""):
    """Register one trial's triangulated mocap into model/world frame, using
    a SHARED rotation (averaged across many trials' own fits, see
    average_rotations/main) rather than this trial's own independent
    Procrustes fit.

    Why a shared rotation: the joystick and cameras are physically identical
    across all 15 CFL_35/36/37 trials (confirmed byte-identical
    calibration.toml), so the joystick's TRUE angle relative to the mouse's
    arm-anchored model frame should be consistent trial-to-trial. It wasn't
    -- each trial's own Procrustes fit (from that trial's own noisy resting
    arm pose) introduces independent rotational noise, which is invisible in
    the arm/hand fit (small, nearby points) but gets amplified for the
    joystick (a rigid object far from the shoulder anchor used for the fit),
    showing up as the SAME physical joystick appearing at visibly different
    angles in different trials' reprojection videos. Using one shared,
    robustly-averaged rotation for every trial's registration fixes this
    directly, at the small cost of not perfectly capturing genuine (and
    likely tiny) resting-pose orientation differences between trials.

    Also uses a per-trial UNIFORM scale correction (not per-bone rescaling,
    see prepare_trial's docstring) instead of the raw, uncorrected proportions
    -- still no per-bone distortion, just a single scalar fixing gross
    calibration-scale error.
    """
    T = prep["T"]
    nan_frac = prep["nan_frac"]
    rest_idx = prep["rest_idx"]
    shoulder_ref_mm = prep["shoulder_ref_mm"]
    scaled_m = prep["scaled_m"]

    aligned = np.einsum("fkd,dj->fkj", scaled_m, shared_R)
    registered_m = aligned + targets["shoulder"][None, None, :]

    # diagnostic only now (not a correction target): how far the REAL,
    # rigidly-registered bone lengths sit from the model's own assumption.
    max_len_err = 0.0
    for (p, c), L in target_lengths.items():
        got = np.linalg.norm(registered_m[:, IDX[c], :] - registered_m[:, IDX[p], :], axis=1)
        max_len_err = max(max_len_err, float(np.abs(np.median(got) - L)))

    # js_ball stabilization: raw per-frame js_ball tracking is unreliable
    # (occlusion by the mouse's hand during actual joystick contact causes
    # multi-cm jumps -- see CALIBRATION_REFINEMENT_REPORT.md's joystick
    # sections), while js_base tracks more reliably. Since js_base and
    # js_ball are two points on the SAME rigid joystick object, their
    # relative offset is physically constant -- only the whole assembly's
    # x/y translation should vary per-frame, which js_base alone already
    # captures. Rather than feed STAC two potentially-inconsistent per-frame
    # targets, derive js_ball's target for EVERY frame from js_base's
    # (reliable) per-frame position plus a single fixed offset, instead of
    # trusting js_ball's own noisy raw per-frame tracking at all. This
    # eliminates ball jitter/jumps entirely, at the cost of not capturing any
    # (physically-impossible-anyway) relative movement between ball and base.
    #
    # The offset is measured from only the first few frames (up to 15), NOT
    # the full `rest_idx` resting window: `rest_idx` is chosen by low ARM
    # velocity, which is unrelated to js_ball tracking quality -- the mouse's
    # body can be still while the ball detection itself glitches from hand
    # occlusion. Confirmed on CFL_35_20240128_trial_0001: its resting window
    # is the sparse set [0,1,2,35..46], which overlaps the known-corrupted
    # js_ball tracking stretch (frames 26-41), inflating the offset (2.35mm)
    # relative to the clean early frames (~2.0-2.2mm, frame 0 alone: 2.05mm).
    # The first few frames of a trial, before any reaching behavior begins,
    # are the safest bet for an unoccluded view of the ball.
    offset_window = np.arange(min(15, T))
    js_ball_offset = np.median(
        registered_m[offset_window, IDX["js_ball"], :] - registered_m[offset_window, IDX["js_base"], :], axis=0
    )
    registered_m[:, IDX["js_ball"], :] = registered_m[:, IDX["js_base"], :] + js_ball_offset[None, :]

    js_base_resting = np.median(registered_m[rest_idx, IDX["js_base"], :], axis=0)
    js_ball_resting = np.median(registered_m[rest_idx, IDX["js_ball"], :], axis=0)

    if verbose:
        print(f"[{trial_name}] T={T} nan_frac={nan_frac:.3f} "
              f"rest_idx=[{rest_idx[0]}..{rest_idx[-1]}] (n={len(rest_idx)}) "
              f"scale_factor={prep['scale_factor']:.4f} "
              f"max_bone_len_err={max_len_err*1e6:.3f}um "
              f"js_base_resting={js_base_resting}")

    out = registered_m[:, None, :, :]
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("tracks", data=out, compression="gzip", compression_opts=4)
        f.attrs["registration_rotation"] = shared_R
        f.attrs["registration_rotation_own"] = prep["own_rotation"]  # pre-averaging, diagnostic only
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
        f.attrs["recipe"] = "register_v21_mocap_shared_rotation_v4"  # shared rotation + uniform scale, no tree_rescale
        for (p, c), L in target_lengths.items():
            f.attrs[f"target_length_{p}_to_{c}"] = L

    return {
        "trial": trial_name,
        "T": T,
        "nan_frac": nan_frac,
        "rest_idx": rest_idx,
        "max_len_err": max_len_err,
        "js_base_resting": js_base_resting,
        "js_ball_resting": js_ball_resting,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="*", default=["CFL_35", "CFL_36", "CFL_37"])
    p.add_argument("--out-suffix", default="_registered_v21.h5")
    p.add_argument("--points3d-suffix", default="_points3d.h5",
                    help="use '_points3d_refined.h5' to register against the "
                         "aniposelib-refined-calibration triangulation instead "
                         "of the original points3d.h5")
    args = p.parse_args()

    targets = model_rest_targets(MODEL_PATH)
    target_lengths = segment_lengths(targets)
    print("Model target segment lengths (mm):")
    for (p_, c_), L in target_lengths.items():
        print(f"  {p_:10s} -> {c_:8s}: {L*1000:.4f} mm")
    print(f"MODEL shoulder (world, m): {targets['shoulder']}")
    print()

    trial_dirs = [d for d in sorted(TRAINING_ROOT.iterdir())
                  if d.is_dir() and any(d.name.startswith(s) for s in args.sessions)]

    # ---- Pass 1: prepare every trial (fill/smooth/rest-window/scale/own
    # rotation), and flag poor-calibration-convergence trials (same
    # diagnostic as before) so they're excluded from the shared-rotation
    # average -- their own Procrustes fit is presumably noisier given their
    # underlying triangulation already has known convergence problems. ----
    preps = {}
    gauge_ok = {}
    for trial_dir in trial_dirs:
        points3d = trial_dir / f"{trial_dir.name}{args.points3d_suffix}"
        if not points3d.exists():
            print(f"[skip] no {args.points3d_suffix} in {trial_dir.name}")
            continue
        preps[trial_dir.name] = prepare_trial(points3d, targets)

        ok = True
        stats_path = trial_dir / f"{trial_dir.name}_calib_refine_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                calib_stats = json.load(f)
            scale = calib_stats.get("gauge_correction", {}).get("scale", 1.0)
            if abs(np.log(scale)) > 0.3:
                ok = False
        gauge_ok[trial_dir.name] = ok

    reliable_names = [n for n, ok in gauge_ok.items() if ok]
    shared_R = average_rotations([preps[n]["own_rotation"] for n in reliable_names])
    print(f"Shared rotation averaged over {len(reliable_names)}/{len(preps)} "
          f"gauge-reliable trials (excluded: {[n for n, ok in gauge_ok.items() if not ok]}).")
    print(f"Per-trial scale factors: " +
          ", ".join(f"{n}={preps[n]['scale_factor']:.3f}" for n in preps))
    print()

    # ---- Pass 2: actually register every trial using the shared rotation. ----
    results = []
    for trial_dir in trial_dirs:
        if trial_dir.name not in preps:
            continue
        out_h5 = trial_dir / f"{trial_dir.name}{args.out_suffix}"
        res = register_trial(preps[trial_dir.name], out_h5, targets, target_lengths,
                              shared_R, trial_name=trial_dir.name)
        res["gauge_ok"] = gauge_ok[trial_dir.name]
        if not res["gauge_ok"]:
            print(f"  [{trial_dir.name}] excluded from joystick_base averaging "
                  f"(poor calibration-refinement convergence)")
        results.append(res)

    print(f"\nRegistered {len(results)} trials.")
    js_base_all = np.array([r["js_base_resting"] for r in results])
    js_ball_all = np.array([r["js_ball_resting"] for r in results])
    good = np.array([r["gauge_ok"] for r in results])
    print(f"js_base resting position across trials: mean={js_base_all.mean(0)}, "
          f"std={js_base_all.std(0)}, range={js_base_all.max(0) - js_base_all.min(0)}")
    print(f"js_ball resting position across trials: mean={js_ball_all.mean(0)}, "
          f"std={js_ball_all.std(0)}, range={js_ball_all.max(0) - js_ball_all.min(0)}")
    if (~good).any():
        print(f"{(~good).sum()}/{len(results)} trials excluded from joystick averaging "
              f"(poor calibration-refinement convergence): "
              f"{[r['trial'] for r in results if not r['gauge_ok']]}")

    # joystick_base body placement is NO LONGER derived here. Earlier
    # approaches (averaging js_base/js_ball-derived position candidates, then
    # anchoring on js_ball alone) both tried to recover the joystick's
    # position from noisy per-trial tracking of two single, isolated points
    # with no bone-length constraint -- neither could ever recover the joystick's
    # true SIZE or ORIENTATION (only a rough position), and both left visible
    # misalignment/crookedness. Superseded by compute_joystick_geometry.py,
    # which uses a dedicated static calibration (js_base/js_ball/js_bottom
    # labeled once in all 3 cameras: static_joystick_labels_withbottom.v001.slp)
    # to directly measure the joystick's true shaft length, ball offset, and
    # tilt, and computes joystick_base's position from the bottom/mount point
    # -- see that script and build_v21_xml.py's JOYSTICK_BODY comment. The
    # js_base_resting/js_ball_resting stats above remain useful as a
    # per-trial sanity check/cross-validation against the static
    # calibration's own model-frame ball/base positions, not as the position
    # source.


if __name__ == "__main__":
    main()
