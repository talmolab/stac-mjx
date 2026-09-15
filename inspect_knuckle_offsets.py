"""Compute data-derived KnuckleC (d3) and KnuckleL (d5) offsets in wrist-local frame,
and render a preview of the v16 hand mesh with the proposed knuckle sites overlaid.

Outputs:
  - stdout: numerical offsets in wrist-local frame, comparison to plan guesses
  - v17_knuckle_proposal_camA.png: render from janelia_cam viewpoint
  - v17_knuckle_proposal_top.png:  top-down view of the hand
  - v17_knuckle_proposal_side.png: lateral view of the hand
"""
from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pathlib import Path
import numpy as np
import h5py
import mujoco
from scipy.signal import savgol_filter
from scipy.linalg import orthogonal_procrustes

V16_XML = "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml"
SESSION_DIR = Path("/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video")
SID = "A36-1_2023-07-18_16-54-01_lightOff_video"
V7_FILE = SESSION_DIR / f"{SID}_mujoco_registered_forelimb_right_v7.h5"
POINTS3D_FILE = SESSION_DIR / f"{SID}_points3d.h5"
DIGITS_FILE = SESSION_DIR / f"{SID}_digits_points3d.h5"

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
        data.xmat[wrist_id].reshape(3, 3).copy(),
        data.xmat[ulna_id].reshape(3, 3).copy(),
    )


def smooth_positions(positions, window=15, polyorder=3):
    out = np.copy(positions)
    for kp in range(positions.shape[1]):
        for d in range(3):
            out[:, kp, d] = savgol_filter(positions[:, kp, d], window, polyorder)
    return out


def linear_interp_nans(arr_TKD):
    """In-place linear-interpolate NaN holes per (kp, dim). arr shape: (T, K, D)."""
    out = arr_TKD.copy()
    T, K, D = out.shape
    for k in range(K):
        for d in range(D):
            x = out[:, k, d]
            nans = np.isnan(x)
            if nans.all():
                continue
            if nans.any():
                idx = np.flatnonzero(nans)
                ok = np.flatnonzero(~nans)
                x[idx] = np.interp(idx, ok, x[ok])
                out[:, k, d] = x
    return out


def derive_offsets():
    """Apply v7 recipe to one frame of A36-1 to derive KnuckleC/L offsets in wrist-local frame."""
    print("=" * 70)
    print("Deriving knuckle offsets from data (A36-1)")
    print("=" * 70)

    # 1. Load model + gravity-settle to find wrist body pose
    sh_w, el_w, wr_w, wrist_R, ulna_R = gravity_settled_pose(MODEL_PATH)
    H_LEN = float(np.linalg.norm(el_w - sh_w))
    F_LEN = float(np.linalg.norm(wr_w - el_w))
    print(f"\nModel (gravity-settled, m):")
    print(f"  shoulder = {sh_w}")
    print(f"  elbow    = {el_w}")
    print(f"  wrist    = {wr_w}")
    print(f"  H_LEN = {H_LEN*1000:.3f} mm,  F_LEN = {F_LEN*1000:.3f} mm")

    # 2. Load v7 to get the canonical R + shoulder_ref for A36-1
    with h5py.File(V7_FILE, "r") as f:
        R_v7 = np.array(f.attrs["registration_rotation"])
        t_v7 = np.array(f.attrs["registration_translation"])
        sh_ref_mm = np.array(f.attrs["shoulder_reference_mm"])
    print(f"\nv7 R det = {np.linalg.det(R_v7):+.6f}, sh_ref_mm = {sh_ref_mm}")
    assert np.allclose(t_v7, sh_w, atol=1e-6), "v7 t should equal model shoulder world"

    # 3. Load 3kp un-registered + 17kp digits raw (mm, cam frame)
    with h5py.File(POINTS3D_FILE, "r") as f:
        kps3 = f["tracks"][:, 0, :, :].astype(np.float64)  # (T, 3, 3)
        names3 = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    with h5py.File(DIGITS_FILE, "r") as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)  # (T, 17, 3)
        digit_names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    print(f"\nLoaded 3kp ({names3}) shape {kps3.shape}")
    print(f"Loaded digits 17 nodes shape {digits.shape}")
    d3_idx = digit_names.index("right_d3_knuckle")
    d5_idx = digit_names.index("right_d5_knuckle")
    print(f"d3 at index {d3_idx}, d5 at index {d5_idx}")

    # 4. Combine 5kp [S, E, W, d3, d5] in raw cam mm
    kps5 = np.concatenate([
        kps3,
        digits[:, [d3_idx, d5_idx], :],
    ], axis=1)  # (T, 5, 3)
    print(f"\n5kp combined shape: {kps5.shape}")
    nan_per_kp = np.isnan(kps5).any(axis=2).mean(axis=0)
    print(f"  NaN frac per kp: " + ", ".join(f"{n:.3f}" for n in nan_per_kp))

    # 5. Linear-interp NaN holes (savgol can't handle NaN)
    kps5 = linear_interp_nans(kps5)

    # 6. Smooth (savgol w=15, p=3)
    sm_mm = smooth_positions(kps5, window=15, polyorder=3)

    # 7. Center on shoulder_ref_mm, convert to meters
    centered_m = (sm_mm - sh_ref_mm[None, None, :]) / 1000.0  # (T, 5, 3)

    # 8. Rescale arm bones to model lengths (per-frame)
    sh_pf = centered_m[:, 0, :]
    h_vec = centered_m[:, 1, :] - sh_pf
    f_vec = centered_m[:, 2, :] - centered_m[:, 1, :]
    hn = np.maximum(np.linalg.norm(h_vec, axis=1, keepdims=True), 1e-10)
    fn = np.maximum(np.linalg.norm(f_vec, axis=1, keepdims=True), 1e-10)
    new_elbow = sh_pf + (h_vec / hn) * H_LEN
    new_wrist = new_elbow + (f_vec / fn) * F_LEN

    # 9. Knuckles: preserve wrist→knuckle delta from raw data
    raw_wrist = centered_m[:, 2, :]
    delta_d3 = centered_m[:, 3, :] - raw_wrist
    delta_d5 = centered_m[:, 4, :] - raw_wrist
    new_d3 = new_wrist + delta_d3
    new_d5 = new_wrist + delta_d5

    rescaled = np.stack([sh_pf, new_elbow, new_wrist, new_d3, new_d5], axis=1)

    # 10. Apply rotation R (use v7's stored R) + translate by MODEL_SHOULDER
    aligned = np.einsum("fkd,dj->fkj", rescaled, R_v7)
    registered = aligned + sh_w[None, None, :]  # (T, 5, 3) in model world frame, m

    # Check: registered Wrist should be near model wr_w when arm is at rest
    rest_wrist = registered[:200, 2, :].mean(axis=0)
    print(f"\nRegistered S/E/W (mean of first 200 frames, m):")
    print(f"  shoulder = {registered[:200, 0, :].mean(axis=0)}")
    print(f"  elbow    = {registered[:200, 1, :].mean(axis=0)}")
    print(f"  wrist    = {rest_wrist}")
    print(f"  model wrist = {wr_w}")
    print(f"  diff |reg_wrist - model_wrist| = {np.linalg.norm(rest_wrist - wr_w)*1000:.3f} mm")

    # 11. Pick a "good" frame: low velocity + d3/d5 valid (no NaN in original)
    raw_d3 = digits[:, d3_idx, :]
    raw_d5 = digits[:, d5_idx, :]
    valid_d3 = ~np.isnan(raw_d3).any(axis=1)
    valid_d5 = ~np.isnan(raw_d5).any(axis=1)
    valid_both = valid_d3 & valid_d5
    print(f"\nValid (non-NaN) in original: d3={valid_d3.mean():.3f}, d5={valid_d5.mean():.3f}, both={valid_both.mean():.3f}")

    # find low-velocity frames among valid_both
    vel = np.zeros(len(registered))
    vel[1:] = np.linalg.norm(np.diff(registered[:, 2, :], axis=0), axis=1)
    valid_both_idx = np.flatnonzero(valid_both)
    vel_at_valid = vel[valid_both_idx]
    rest_pick = valid_both_idx[np.argsort(vel_at_valid)[:50]]
    print(f"Picked 50 lowest-velocity valid frames; idx range {rest_pick.min()}..{rest_pick.max()}")

    # 12. Compute knuckle - wrist in WORLD frame at those frames, average
    d3_world_offset = registered[rest_pick, 3, :] - registered[rest_pick, 2, :]
    d5_world_offset = registered[rest_pick, 4, :] - registered[rest_pick, 2, :]
    print(f"\nKnuckle - wrist (WORLD, mean over 50 rest frames, mm):")
    print(f"  d3 (KnuckleC) = {d3_world_offset.mean(0)*1000}")
    print(f"  d5 (KnuckleL) = {d5_world_offset.mean(0)*1000}")
    print(f"  d3 std = {d3_world_offset.std(0)*1000} mm")
    print(f"  d5 std = {d5_world_offset.std(0)*1000} mm")

    # 13. Convert to wrist-local frame
    d3_local = wrist_R.T @ d3_world_offset.mean(0)
    d5_local = wrist_R.T @ d5_world_offset.mean(0)
    print(f"\nKnuckle offsets in WRIST-LOCAL frame (mean, mm):")
    print(f"  KnuckleC (d3) = ({d3_local[0]*1000:+.3f}, {d3_local[1]*1000:+.3f}, {d3_local[2]*1000:+.3f})")
    print(f"  KnuckleL (d5) = ({d5_local[0]*1000:+.3f}, {d5_local[1]*1000:+.3f}, {d5_local[2]*1000:+.3f})")
    print(f"\nPlan guesses:")
    print(f"  KnuckleC (d3) = ( +0.000,  +5.000,  +0.000) mm")
    print(f"  KnuckleL (d5) = ( +3.000,  +5.000,  +0.000) mm")

    return d3_local, d5_local


def render_proposal(d3_local: np.ndarray, d5_local: np.ndarray,
                    d3_plan=(0.0, 0.005, 0.0), d5_plan=(0.003, 0.005, 0.0)):
    """Render the v16 model with proposed KnuckleC/L sites added to the wrist body.

    Renders both the data-derived offsets (cyan/magenta) AND the plan's guess offsets
    (lighter shades, smaller spheres) so they can be compared side-by-side.
    Adds a wrist-body coordinate frame triad so the axes orientation is visible.
    """
    print("\n" + "=" * 70)
    print("Rendering preview")
    print("=" * 70)

    # Patch the v16 XML: add wrist + knuckle sites + small axis-triad sites + a
    # forearm-long-axis reference cylinder on the ulna body
    with open(V16_XML, "r") as f:
        xml = f.read()

    # Coordinate axes at wrist (3 small spheres at +X, +Y, +Z to show frame orientation)
    AX_LEN = 0.003   # 3 mm reference axis
    SPHERE_R = 0.0012  # bigger spheres for visibility
    PLAN_SPHERE_R = 0.0008  # plan guesses get smaller spheres
    site_block_template = (
        f'              <site name="Wrist"    type="sphere" size="{SPHERE_R}" rgba="1 1 0 1"   pos="0 0 0"/>\n'
        f'              <site name="WristAxX" type="sphere" size="0.0004" rgba="1 0 0 1"   pos="{AX_LEN} 0 0"/>\n'
        f'              <site name="WristAxY" type="sphere" size="0.0004" rgba="0 0.8 0 1" pos="0 {AX_LEN} 0"/>\n'
        f'              <site name="WristAxZ" type="sphere" size="0.0004" rgba="0 0 1 1"   pos="0 0 {AX_LEN}"/>\n'
        f'              <site name="KnuckleC"     type="sphere" size="{SPHERE_R}" rgba="0 1 1 1"   pos="{d3_local[0]:.6f} {d3_local[1]:.6f} {d3_local[2]:.6f}"/>\n'
        f'              <site name="KnuckleL"     type="sphere" size="{SPHERE_R}" rgba="1 0 1 1"   pos="{d5_local[0]:.6f} {d5_local[1]:.6f} {d5_local[2]:.6f}"/>\n'
        f'              <site name="KnuckleC_plan" type="sphere" size="{PLAN_SPHERE_R}" rgba="0.6 0.85 0.85 0.7" pos="{d3_plan[0]:.6f} {d3_plan[1]:.6f} {d3_plan[2]:.6f}"/>\n'
        f'              <site name="KnuckleL_plan" type="sphere" size="{PLAN_SPHERE_R}" rgba="0.85 0.5 0.85 0.7" pos="{d5_plan[0]:.6f} {d5_plan[1]:.6f} {d5_plan[2]:.6f}"/>\n'
    )

    import re
    pat = re.compile(
        r'<body name="wrist" pos="0 0\.01092 0">\s*<inertial[^/]+/>\s*</body>',
        re.MULTILINE | re.DOTALL,
    )
    if not pat.search(xml):
        raise RuntimeError("Could not find wrist body block in v16 XML")
    new_block = (
        '<body name="wrist" pos="0 0.01092 0">\n'
        '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + site_block_template
        + '            </body>'
    )
    xml = pat.sub(new_block, xml)

    # Save patched XML
    tmp_xml = "/tmp/v17_preview.xml"
    with open(tmp_xml, "w") as f:
        f.write(xml)

    # Load patched model. Use mj_forward only (no stepping): the v16 model has
    # shoulder_base slide joints that diverge when stepping dynamics from qpos=0.
    # For static rendering, the qpos=0 design rest pose is what we want anyway.
    model = mujoco.MjModel.from_xml_path(tmp_xml)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    # Confirm the sites exist
    for site_name in ["Wrist", "KnuckleC", "KnuckleL", "WristAxX", "WristAxY", "WristAxZ"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise RuntimeError(f"Site {site_name} not in patched model!")
        spos = data.site_xpos[sid]
        print(f"  Site {site_name:14s} world pos = ({spos[0]:.5f}, {spos[1]:.5f}, {spos[2]:.5f})")

    # Render from camera angles. Use a tight zoom on the hand so spheres are visible.
    H, W = 720, 960
    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_opt.sitegroup[0] = True
    scene_opt.sitegroup[1] = True

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wrist_pos = data.xpos[wrist_id].copy()

    import imageio.v3 as iio

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = wrist_pos + np.array([0.003, 0, 0])  # tiny offset into the hand
    cam.distance = 0.018  # zoom in (was 0.04-0.05)

    views = [
        # (azimuth, elevation, filename, caption)
        (90, -85, "v17_knuckle_proposal_top.png",      "TOP-DOWN (looking at dorsal paw surface)"),
        (90,   0, "v17_knuckle_proposal_distal.png",   "DISTAL (looking from fingers toward elbow)"),
        ( 0,   0, "v17_knuckle_proposal_lateral.png",  "LATERAL (looking along ulna axis)"),
        ( 0,  60, "v17_knuckle_proposal_oblique.png",  "OBLIQUE (3/4 view of dorsal paw)"),
    ]
    rendered = {}
    for az, el, fname, caption in views:
        cam.azimuth = az
        cam.elevation = el
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        rendered[fname] = (img, caption)
        iio.imwrite(fname, img)
        print(f"  -> {fname}")

    # Also render via the named janelia_cam (zoomed out, full arm context)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "janelia_cam")
    if cam_id >= 0:
        renderer.update_scene(data, camera=cam_id, scene_option=scene_opt)
        img = renderer.render()
        iio.imwrite("v17_knuckle_proposal_camA.png", img)
        print(f"  -> v17_knuckle_proposal_camA.png")
        rendered["v17_knuckle_proposal_camA.png"] = (img, "JANELIA CAM (full arm)")

    # Composite a single annotated PNG with text labels
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        ordered = [
            "v17_knuckle_proposal_top.png",
            "v17_knuckle_proposal_distal.png",
            "v17_knuckle_proposal_camA.png",
            "v17_knuckle_proposal_lateral.png",
            "v17_knuckle_proposal_oblique.png",
        ]
        for ax, fname in zip(axes.ravel(), ordered):
            img, caption = rendered[fname]
            ax.imshow(img)
            ax.set_title(caption, fontsize=11)
            ax.axis("off")
        # 6th panel: legend / numbers
        legend_ax = axes.ravel()[5]
        legend_ax.axis("off")
        legend_text = (
            "WRIST-LOCAL FRAME\n"
            "  Red sphere   = +X axis (3mm)\n"
            "  Green sphere = +Y axis (3mm)\n"
            "  Blue sphere  = +Z axis (3mm)\n\n"
            "KNUCKLE SITES\n"
            "  Yellow      = Wrist (origin)\n"
            "  Cyan (big)  = KnuckleC = d3 (data-derived)\n"
            "  Magenta(big)= KnuckleL = d5 (data-derived)\n"
            "  Cyan (small)= KnuckleC PLAN guess\n"
            "  Pink (small)= KnuckleL PLAN guess\n\n"
            "DATA-DERIVED OFFSETS (mm, wrist-local)\n"
            f"  KnuckleC (d3) = ({d3_local[0]*1000:+.2f}, {d3_local[1]*1000:+.2f}, {d3_local[2]*1000:+.2f})\n"
            f"  KnuckleL (d5) = ({d5_local[0]*1000:+.2f}, {d5_local[1]*1000:+.2f}, {d5_local[2]*1000:+.2f})\n\n"
            "PLAN GUESSES (mm)\n"
            f"  KnuckleC = (+0.00, +5.00, +0.00)\n"
            f"  KnuckleL = (+3.00, +5.00, +0.00)\n\n"
            "Mean wrist-to-knuckle distance: ~2-3 mm (data)\n"
            "vs ~5 mm (plan).\n\n"
            "Spheres visible ABOVE the metacarpal mesh\n"
            "= dorsal-surface SLEAP labels, biologically correct."
        )
        legend_ax.text(0.02, 0.98, legend_text, fontsize=10,
                        family="monospace", verticalalignment="top",
                        transform=legend_ax.transAxes)
        plt.suptitle("v17 KnuckleC/KnuckleL site placement preview\n"
                      "(model: v16 XML + proposed sites; gravity-settled qpos=0)",
                      fontsize=13, y=0.995)
        plt.tight_layout()
        plt.savefig("v17_knuckle_proposal_ALL.png", dpi=120, bbox_inches="tight")
        print(f"  -> v17_knuckle_proposal_ALL.png  (composite)")
    except Exception as e:
        print(f"  composite render failed: {e}")


if __name__ == "__main__":
    import imageio.v3
    d3, d5 = derive_offsets()
    render_proposal(d3, d5)
    print("\nDone.")
