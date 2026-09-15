"""Splice v7 registered 3kp (Shoulder/Elbow/Wrist) with d2_mid + d5_mid from
the digits triangulation, applying the same v7 registration recipe extended to
5 keypoints. Output per-session 5kp registered h5 file for v17 STAC input.

Output:
  <session_dir>/<session>_mujoco_registered_forelimb_right_hand_v1.h5
    /tracks    (T, 1, 5, 3) m, registered to model frame
    attrs.kp_names = ["Shoulder", "Elbow", "Wrist", "KnuckleC", "KnuckleL"]
    attrs.knuckle_C_landmark = "right_d2_mid"
    attrs.knuckle_L_landmark = "right_d5_mid"
"""
from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pathlib import Path
import numpy as np
import h5py
from scipy.signal import savgol_filter

ROOT = Path("/mnt/vast/eric/azim-emg-data")
SESSIONS = {
    "A36-1": "A36-1_2023-07-18_16-54-01_lightOff_video",
    "AT006": "AT006_2024-02-28_15-39-40_LightOff_video",
    "AT009": "AT009_2024-04-25_13-50-34_LightOff_video",
    "AT012": "AT012_2024-06-21_12-32-30_LightOff_video",
    "AT013": "AT013_2024-06-20_13-41-20_LightOff_video",
}
KNUCKLE_C_LM = "right_d2_mid"
KNUCKLE_L_LM = "right_d5_mid"


def lerp(arr):
    out = arr.copy()
    for k in range(arr.shape[1]):
        for d in range(arr.shape[2]):
            x = out[:, k, d]; nan = np.isnan(x)
            if not nan.any() or nan.all(): continue
            idx = np.flatnonzero(nan); ok = np.flatnonzero(~nan)
            x[idx] = np.interp(idx, ok, x[ok]); out[:, k, d] = x
    return out


def smooth(p, w=15, po=3):
    out = p.copy()
    for k in range(p.shape[1]):
        for d in range(3):
            out[:, k, d] = savgol_filter(p[:, k, d], w, po)
    return out


def splice_session(session: str):
    sd = ROOT / session
    v7_path = sd / f"{session}_mujoco_registered_forelimb_right_v7.h5"
    p3d_path = sd / f"{session}_points3d.h5"
    dig_path = sd / f"{session}_digits_points3d.h5"
    out_path = sd / f"{session}_mujoco_registered_forelimb_right_hand_v1.h5"

    with h5py.File(v7_path) as f:
        v7 = f["tracks"][:, 0, :, :].astype(np.float64)  # (T, 3, 3) m
        R_v7 = np.array(f.attrs["registration_rotation"])
        sh_ref_mm = np.array(f.attrs["shoulder_reference_mm"])
        sh_w = np.array(f.attrs["registration_translation"])
        # Original model bone lengths (NOT used; kept for reference)
        H_LEN_model = float(f.attrs["target_humerus_len"])
        F_LEN_model = float(f.attrs["target_forearm_len"])

    # OPTION B: Use ANIMAL median bone lengths instead of model design lengths.
    # This makes the rescaled chain match the animal's actual mean bones, so the
    # reprojection inverse lands on the raw SLEAP labels (modulo per-frame mocap
    # noise). Tradeoff: STAC fit now has a small constant residual = (model_bone -
    # animal_bone) that the model can't shrink to zero, but the visual overlay is
    # correct.
    with h5py.File(p3d_path) as f:
        raw3 = f["tracks"][:, 0, :, :].astype(np.float64)
    h_lens = np.linalg.norm(raw3[:, 1] - raw3[:, 0], axis=1)
    f_lens = np.linalg.norm(raw3[:, 2] - raw3[:, 1], axis=1)
    H_LEN = float(np.nanmedian(h_lens)) / 1000.0   # mm → m
    F_LEN = float(np.nanmedian(f_lens)) / 1000.0
    print(f"  [{session}] animal median bones (mm): H={H_LEN*1000:.3f}  F={F_LEN*1000:.3f}  (model: H={H_LEN_model*1000:.3f}  F={F_LEN_model*1000:.3f})")
    # raw3 already loaded above for bone-length computation
    with h5py.File(dig_path) as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        re_err = f["reproj_err"][:]
        nd = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]

    # Frame alignment
    T = min(v7.shape[0], raw3.shape[0], digits.shape[0])
    if not (v7.shape[0] == raw3.shape[0] == digits.shape[0]):
        print(f"  [{session}] WARN: frame counts differ — v7 {v7.shape[0]}, p3d {raw3.shape[0]}, "
              f"digits {digits.shape[0]} → truncating to {T}")
    v7 = v7[:T]; raw3 = raw3[:T]; digits = digits[:T]; re_err = re_err[:, :T]

    iC = nd.index(KNUCKLE_C_LM); iL = nd.index(KNUCKLE_L_LM)
    vC = ~np.isnan(re_err[:, :, iC]).any(axis=0)
    vL = ~np.isnan(re_err[:, :, iL]).any(axis=0)
    vC_frac = vC.mean(); vL_frac = vL.mean(); vBoth = (vC & vL).mean()

    # Build raw 5kp (mm, cam frame), lerp NaN holes, savgol-smooth — same recipe as v7
    raw5 = np.concatenate([raw3, digits[:, [iC, iL], :]], axis=1)  # (T, 5, 3) mm
    raw5 = lerp(raw5)
    sm_mm = smooth(raw5, w=15, po=3)

    # Center on shoulder_ref_mm, convert to m
    centered_m = (sm_mm - sh_ref_mm[None, None, :]) / 1000.0  # (T, 5, 3)

    # OPTION C: NO bone-rescaling. Preserve raw bone lengths per frame.
    # registered = R @ (raw - sh_ref) + sh, no length normalization.
    # STAC will fit a model with fixed bone lengths to varying-length data;
    # per-frame residual = bone-length variance (mocap noise).
    # Reproject inverse gives back raw positions → 2D dots land on actual
    # SLEAP labels exactly.
    rescaled = centered_m  # (T, 5, 3) — just centered, no rescaling

    # Apply registration rotation + translate by MODEL_SHOULDER
    aligned = np.einsum("fkd,dj->fkj", rescaled, R_v7)
    registered = aligned + sh_w[None, None, :]  # (T, 5, 3) m, model frame

    # Sanity check: registered S/E/W should equal v7 file's stored data (within machine epsilon)
    sw_diff = np.abs(registered[:, :3, :] - v7).max()
    print(f"  [{session}] vC_valid={vC_frac:.3f}  vL_valid={vL_frac:.3f}  v_both={vBoth:.3f}  "
          f"  S/E/W reproduce match: max_diff={sw_diff*1000:.6f}mm")

    # Save
    out = registered[:, None, :, :]  # (T, 1, 5, 3)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("tracks", data=out, compression="gzip", compression_opts=4)
        f.attrs["kp_names"] = np.array(["Shoulder", "Elbow", "Wrist", "KnuckleC", "KnuckleL"], dtype="S")
        f.attrs["knuckle_C_landmark"] = KNUCKLE_C_LM
        f.attrs["knuckle_L_landmark"] = KNUCKLE_L_LM
        f.attrs["registration_method"] = "v7_extended_to_5kp"
        f.attrs["registration_rotation"] = R_v7
        f.attrs["registration_translation"] = sh_w
        f.attrs["shoulder_reference_mm"] = sh_ref_mm
        f.attrs["target_humerus_len"] = H_LEN
        f.attrs["target_forearm_len"] = F_LEN
        f.attrs["target_humerus_len_model"] = H_LEN_model
        f.attrs["target_forearm_len_model"] = F_LEN_model
        f.attrs["validity_C"] = vC_frac
        f.attrs["validity_L"] = vL_frac
        f.attrs["validity_both"] = vBoth
        f.attrs["recipe"] = "splice_5kp_v17.py — extends v7 recipe; preserves wrist→knuckle delta"
    print(f"    -> {out_path.name}  ({out.shape}, gzip)")
    return out_path


def main():
    print("Splicing 5kp registered files for v17...")
    out_paths = []
    for animal, session in SESSIONS.items():
        try:
            p = splice_session(session)
            out_paths.append(p)
        except Exception as e:
            print(f"  [{session}] ERROR: {e}")
    print(f"\nDone. {len(out_paths)} files written.")


if __name__ == "__main__":
    main()
