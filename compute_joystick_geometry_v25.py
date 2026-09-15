#!/usr/bin/env python
"""Derive v25's joystick geometry (shaft length, ball offset) and placement
(position + orientation) in v24/v25's OWN model frame, for the joystick body
being added to v24's arm+hand rig.

Identical recipe to compute_joystick_geometry_v22.py -- only the source of
registration_rotation/shoulder_reference_mm changed, from
*_registered_v22_refined.h5 to *_registered_v25_refined.h5 (register_v25_mocap.py,
v24's own model frame). Not optional/cosmetic: v22's world frame and v24's
world frame are DIFFERENT frames (each is its own Procrustes best-fit of the
same real mocap arm data against that model's own, differently-postured rest
pose) -- see register_v25_mocap.py's docstring. Re-running the SAME static
calibration measurement through v24's own averaged registration rotation
gives the correct v24/v25-frame answer directly, with no transplant/rotation-
algebra risk.

Method: identical to compute_joystick_geometry_v22.py (see that file for full
rationale) -- triangulate the static SLP's 3 points, average
registration_rotation/shoulder_reference_mm over the gauge-reliable trials'
own registered_v25_refined.h5 attrs (SVD/chordal rotation mean), transform,
compute shaft_length/ball_offset/shaft_direction, derive a quat rotating
local +Z to shaft_direction for the "joystick" child body.

Output: prints values for build_v25_stac_xml.py's joystick body block and
configs/model/mouse_arm_hand_joystick_v25.yaml's KEYPOINT_INITIAL_OFFSETS.
"""
import h5py
import numpy as np
import sleap_io as sio
from aniposelib.cameras import CameraGroup
from pathlib import Path

STATIC_SLP = Path("/mnt/vast/niyo/stac-mjx/static_joystick_labels_withbottom.v001.slp")
CALIB_PATH = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData/"
                   "CFL_36_20231220_trial_0001/calibration.toml")
TRAINING_ROOT = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData")

# Same trials excluded in compute_joystick_geometry_v22.py, for the same
# reason (poor bundle-adjustment convergence in calibration refinement --
# CALIBRATION_REFINEMENT_REPORT.md section 3). register_v25_mocap.py's own
# scale_factor/max_bone_len_err output flags these same three trials as
# outliers, cross-confirming the exclusion still applies for v24/v25's frame.
EXCLUDED_TRIALS = {
    "CFL_35_20240128_trial_0001",
    "CFL_37_20240423_trial_0001",
    "CFL_37_20240423_trial_0201",
}


def quat_from_z_to(v):
    """Return a MuJoCo quat (w,x,y,z) rotating local +Z to unit vector v."""
    z = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    c = np.dot(z, v)
    if s < 1e-12:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = axis / s
    angle = np.arctan2(s, c)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, *xyz])


def main():
    labels = sio.load_slp(str(STATIC_SLP))
    pts2d = {}
    for lf in labels:
        cam = lf.video.filename.split("/")[-1].split("_")[0]
        pts2d[cam] = lf.instances[0].numpy()
    cam_order = ["Cam00", "Cam01", "Cam02"]
    node_names = [n.name for n in labels.skeleton.nodes]
    print("static SLP node order:", node_names)
    p2ds = np.stack([pts2d[c] for c in cam_order]).reshape(3, -1, 2)

    cgroup = CameraGroup.load(str(CALIB_PATH))
    p3ds_raw = cgroup.triangulate(p2ds).reshape(3, 3)
    idx = {n: i for i, n in enumerate(node_names)}
    ball_raw, base_raw, bottom_raw = p3ds_raw[idx["js_ball"]], p3ds_raw[idx["js_base"]], p3ds_raw[idx["js_bottom"]]
    print(f"raw calibration-frame (mm): ball={ball_raw} base={base_raw} bottom={bottom_raw}")
    print(f"  raw base->ball dist: {np.linalg.norm(ball_raw - base_raw):.3f} mm")
    print(f"  raw base->bottom dist: {np.linalg.norm(base_raw - bottom_raw):.3f} mm")

    all_trials = sorted(d.name for d in TRAINING_ROOT.iterdir()
                         if d.is_dir() and d.name.startswith(("CFL_35", "CFL_36", "CFL_37")))
    reliable = [t for t in all_trials if t not in EXCLUDED_TRIALS]

    Rs, shoulder_refs, model_shoulders = [], [], []
    for t in reliable:
        h5path = TRAINING_ROOT / t / f"{t}_registered_v25_refined.h5"
        if not h5path.exists():
            continue
        with h5py.File(h5path, "r") as f:
            Rs.append(np.array(f.attrs["registration_rotation"]))
            shoulder_refs.append(np.array(f.attrs["shoulder_reference_mm"]))
            model_shoulders.append(np.array(f.attrs["registration_translation"]))
    print(f"\nUsing {len(Rs)} reliable trials for rotation averaging: {reliable}")
    model_shoulders = np.array(model_shoulders)
    assert model_shoulders.std(0).max() < 1e-6, "model shoulder should be a constant across trials"
    model_shoulder = model_shoulders[0]

    Rsum = np.sum(Rs, axis=0)
    U, S, Vt = np.linalg.svd(Rsum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    shoulder_ref_avg = np.mean(shoulder_refs, axis=0)

    def to_model(p_mm):
        centered_m = (p_mm - shoulder_ref_avg) / 1000.0
        return centered_m @ R_avg + model_shoulder

    ball_m = to_model(ball_raw)
    base_m = to_model(base_raw)
    bottom_m = to_model(bottom_raw)

    shaft_vec = base_m - bottom_m
    shaft_length = np.linalg.norm(shaft_vec)
    shaft_dir = shaft_vec / shaft_length
    ball_offset = np.linalg.norm(ball_m - base_m)
    angle_from_z = np.degrees(np.arccos(np.clip(shaft_dir[2], -1, 1)))
    quat = quat_from_z_to(shaft_dir)

    print(f"\n--- Model-frame results (averaged over {len(Rs)} reliable trials) ---")
    print(f"ball_m    = {ball_m}")
    print(f"base_m    = {base_m}")
    print(f"bottom_m  = {bottom_m}")
    print(f"shaft_length (bottom->base) = {shaft_length*1000:.4f} mm = {shaft_length:.6f} m")
    print(f"ball_offset (base->ball)    = {ball_offset*1000:.4f} mm = {ball_offset:.6f} m")
    print(f"shaft_dir (bottom->base, unit vec) = {shaft_dir}")
    print(f"angle from model +Z = {angle_from_z:.2f} deg")
    print(f"quat (w,x,y,z) rotating local +Z to shaft_dir = {quat}")
    print()
    print(f"joystick_base pos=\"{bottom_m[0]:.8f} {bottom_m[1]:.8f} {bottom_m[2]:.8f}\"")
    print(f"joystick child body quat=\"{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}\"")
    print(f"js_base local offset (on joystick body) = (0, 0, {shaft_length:.6f})")
    print(f"js_ball local offset (on joystick body) = (0, 0, {shaft_length+ball_offset:.6f})")
    print(f"js_bottom local offset (on joystick body) = (0, 0, 0)")


if __name__ == "__main__":
    main()
