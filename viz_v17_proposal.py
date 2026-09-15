"""Two visualizations for v17 proposal review:
  1. Supination axis demo — proposed forearm_supination hinge rotates arm + hand
     through {-60, -30, 0, +30, +60} deg, rendered as a strip.
  2. Mocap overlay — model rendered with proposed KnuckleC/L sites + a swarm of
     ~200 actual d3/d5 mocap points (registered to model frame) so we can see
     whether the proposed sites match the data cloud.

Outputs:
  v17_supination_strip.png      (5-panel rotation demo)
  v17_mocap_overlay_top.png     (top-down)
  v17_mocap_overlay_distal.png  (distal view, looking down forearm)
  v17_mocap_overlay_oblique.png (3/4 view)
  v17_proposal_REVIEW.png       (composite for review)
"""
from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import re
from pathlib import Path
import numpy as np
import h5py
import mujoco
import imageio.v3 as iio
from scipy.signal import savgol_filter

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
    )


def linear_interp_nans(arr_TKD):
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


def smooth_positions(positions, window=15, polyorder=3):
    out = np.copy(positions)
    for kp in range(positions.shape[1]):
        for d in range(3):
            out[:, kp, d] = savgol_filter(positions[:, kp, d], window, polyorder)
    return out


def get_registered_d3_d5_cloud(n_frames: int = 200):
    """Return (n_frames, 2, 3) array of registered d3, d5 positions in model world frame."""
    sh_w, el_w, wr_w, wrist_R = gravity_settled_pose(MODEL_PATH)
    H_LEN = float(np.linalg.norm(el_w - sh_w))
    F_LEN = float(np.linalg.norm(wr_w - el_w))

    with h5py.File(V7_FILE, "r") as f:
        R_v7 = np.array(f.attrs["registration_rotation"])
        sh_ref_mm = np.array(f.attrs["shoulder_reference_mm"])
    with h5py.File(POINTS3D_FILE, "r") as f:
        kps3 = f["tracks"][:, 0, :, :].astype(np.float64)
    with h5py.File(DIGITS_FILE, "r") as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        digit_names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    d3_idx = digit_names.index("right_d3_mid")
    d5_idx = digit_names.index("right_d5_mid")

    raw_d3 = digits[:, d3_idx, :]
    raw_d5 = digits[:, d5_idx, :]
    valid = ~np.isnan(raw_d3).any(axis=1) & ~np.isnan(raw_d5).any(axis=1)
    valid_idx = np.flatnonzero(valid)

    # Combine 5kp
    kps5 = np.concatenate([kps3, digits[:, [d3_idx, d5_idx], :]], axis=1)
    kps5 = linear_interp_nans(kps5)
    sm_mm = smooth_positions(kps5, window=15, polyorder=3)

    # Center + register
    centered_m = (sm_mm - sh_ref_mm[None, None, :]) / 1000.0
    sh_pf = centered_m[:, 0, :]
    h_vec = centered_m[:, 1, :] - sh_pf
    f_vec = centered_m[:, 2, :] - centered_m[:, 1, :]
    hn = np.maximum(np.linalg.norm(h_vec, axis=1, keepdims=True), 1e-10)
    fn = np.maximum(np.linalg.norm(f_vec, axis=1, keepdims=True), 1e-10)
    new_elbow = sh_pf + (h_vec / hn) * H_LEN
    new_wrist = new_elbow + (f_vec / fn) * F_LEN
    raw_wrist = centered_m[:, 2, :]
    new_d3 = new_wrist + (centered_m[:, 3, :] - raw_wrist)
    new_d5 = new_wrist + (centered_m[:, 4, :] - raw_wrist)

    rescaled_5 = np.stack([sh_pf, new_elbow, new_wrist, new_d3, new_d5], axis=1)
    aligned = np.einsum("fkd,dj->fkj", rescaled_5, R_v7)
    registered = aligned + sh_w[None, None, :]  # (T, 5, 3) m, model world frame

    # Pick low-velocity (rest) frames where d3+d5 are valid — same set used for cloud
    # AND for the d3_local / d5_local computation so the bright site lands at the
    # exact mean of the cloud.
    vel_all = np.zeros(len(registered))
    vel_all[1:] = np.linalg.norm(np.diff(registered[:, 2, :], axis=0), axis=1)
    valid_low_vel = valid_idx[np.argsort(vel_all[valid_idx])[:n_frames]]
    cloud = registered[valid_low_vel][:, [3, 4], :]  # (n_frames, 2, 3) — d3, d5
    print(f"  Rest-frame mocap cloud: {n_frames} frames")

    d3_world_offset = (registered[valid_low_vel, 3, :] - registered[valid_low_vel, 2, :]).mean(0)
    d5_world_offset = (registered[valid_low_vel, 4, :] - registered[valid_low_vel, 2, :]).mean(0)
    d3_local = wrist_R.T @ d3_world_offset
    d5_local = wrist_R.T @ d5_world_offset

    # Also return the rest-frame slice of the full registered array for downstream
    # alignment of the model render to the cloud's mean wrist.
    rest_slice = registered[valid_low_vel]
    return cloud, d3_local, d5_local, wr_w, rest_slice


def build_xml(d3_local, d5_local, mocap_cloud=None, with_supination=True,
              reparent_hand=True):
    """Build a patched XML with proposed sites + optional supination joint + optional
    a swarm of static world-frame mocap-point sites (d3 cyan, d5 magenta).

    If reparent_hand=True, also reparents all Retopo_* hand mesh geoms from the ulna
    body onto the wrist body (with pos="0 -0.01092 0" compensation so they render in
    the same world position when supination=0). This makes the supination DOF
    visually rotate the hand — the v17 behavior we want.
    """
    with open(V16_XML, "r") as f:
        xml = f.read()

    # 1. Optionally reparent hand meshes from ulna to wrist
    extra_wrist_geoms = ""
    if reparent_hand:
        # Pull all <geom type="mesh" mesh="Retopo_..." class="hand_visual"/> lines
        retopo_pat = re.compile(
            r'^[ \t]*<geom type="mesh" mesh="Retopo_[^"]+" class="hand_visual"/>\s*\n',
            re.MULTILINE,
        )
        retopo_lines = retopo_pat.findall(xml)
        # Remove from ulna body
        xml = retopo_pat.sub("", xml)
        # Build new lines for wrist body, each with pos="0 -0.01092 0" compensation
        wrist_lines = []
        for line in retopo_lines:
            m = re.search(r'mesh="(Retopo_[^"]+)"', line)
            if m:
                wrist_lines.append(
                    f'              <geom type="mesh" mesh="{m.group(1)}" '
                    f'class="hand_visual" pos="0 -0.01092 0"/>'
                )
        extra_wrist_geoms = "\n" + "\n".join(wrist_lines) + "\n"

    # 2. Patch wrist body: add joint (optional) + sites + reparented hand
    # Smaller bright site spheres matched to cloud dot size (0.3mm) so the comparison
    # is honest — bright site CENTER coincides with cloud mean if math is right.
    SPHERE_R = 0.0006
    if with_supination:
        joint_xml = (
            '              <joint name="forearm_supination" type="hinge" '
            'pos="0 0 0" axis="0 1 0" range="-90 90" damping="0.01"/>\n'
        )
    else:
        joint_xml = ""
    sites_xml = (
        f'              <site name="Wrist"    type="sphere" size="{SPHERE_R}" rgba="1 1 0 1"   pos="0 0 0"/>\n'
        f'              <site name="WristAxX" type="sphere" size="0.0004" rgba="1 0 0 1"   pos="0.003 0 0"/>\n'
        f'              <site name="WristAxY" type="sphere" size="0.0004" rgba="0 0.8 0 1" pos="0 0.003 0"/>\n'
        f'              <site name="WristAxZ" type="sphere" size="0.0004" rgba="0 0 1 1"   pos="0 0 0.003"/>\n'
        f'              <site name="KnuckleC" type="sphere" size="{SPHERE_R}" rgba="0 1 1 1"   pos="{d3_local[0]:.6f} {d3_local[1]:.6f} {d3_local[2]:.6f}"/>\n'
        f'              <site name="KnuckleL" type="sphere" size="{SPHERE_R}" rgba="1 0 1 1"   pos="{d5_local[0]:.6f} {d5_local[1]:.6f} {d5_local[2]:.6f}"/>\n'
    )

    pat = re.compile(
        r'<body name="wrist" pos="0 0\.01092 0">\s*<inertial[^/]+/>\s*</body>',
        re.MULTILINE | re.DOTALL,
    )
    new_block = (
        '<body name="wrist" pos="0 0.01092 0">\n'
        + joint_xml
        + '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + sites_xml
        + extra_wrist_geoms
        + '            </body>'
    )
    xml = pat.sub(new_block, xml)

    # 2. If mocap_cloud provided, add world-fixed sites at each cloud point.
    #    cloud shape: (n_frames, 2, 3). All d3 cyan-ish, d5 magenta-ish.
    if mocap_cloud is not None:
        n_frames = mocap_cloud.shape[0]
        cloud_sites = []
        for i in range(n_frames):
            for k, (color, prefix) in enumerate([
                ("0 0.7 0.7 0.4", "d3"),  # cyan-ish translucent
                ("0.7 0 0.7 0.4", "d5"),  # magenta-ish translucent
            ]):
                p = mocap_cloud[i, k]
                cloud_sites.append(
                    f'<site name="cloud_{prefix}_{i:03d}" type="sphere" size="0.0003" '
                    f'rgba="{color}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
                )
        # Inject these into the worldbody (right before </worldbody>)
        cloud_block = "\n      ".join(cloud_sites)
        xml = xml.replace("</worldbody>", f"      <!-- mocap cloud -->\n      {cloud_block}\n    </worldbody>")

    return xml


def render_supination_strip(d3_local, d5_local):
    print("\n=== Supination strip ===")
    xml = build_xml(d3_local, d5_local, mocap_cloud=None, with_supination=True)
    tmp_xml = "/tmp/v17_supination.xml"
    with open(tmp_xml, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(tmp_xml)
    data = mujoco.MjData(model)
    sup_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "forearm_supination")
    sup_qpos_idx = model.jnt_qposadr[sup_jnt]
    print(f"  forearm_supination joint id={sup_jnt}, qpos idx={sup_qpos_idx}")

    H, W = 540, 720
    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.sitegroup[0] = True

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")

    angles_deg = [-60, -30, 0, 30, 60]
    frames = []
    for ang_deg in angles_deg:
        data.qpos[:] = 0
        data.qpos[sup_qpos_idx] = np.deg2rad(ang_deg)
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        wrist_pos = data.xpos[wrist_id].copy()
        cam.lookat[:] = wrist_pos + np.array([0.003, 0, 0])
        cam.distance = 0.020
        cam.azimuth = 90
        cam.elevation = -45  # 3/4 view from above
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        frames.append((ang_deg, img))

    # Compose strip
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, (ang, img) in zip(axes, frames):
        ax.imshow(img)
        ax.set_title(f"forearm_supination = {ang:+d}°", fontsize=12)
        ax.axis("off")
    plt.suptitle(
        "Proposed supination axis: hinge on wrist body, axis = (0, 1, 0) in ulna-local "
        "(green sphere = +Y direction)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("v17_supination_strip.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  -> v17_supination_strip.png")


def render_mocap_overlay(d3_local, d5_local, cloud, wrist_w, registered_full):
    print("\n=== Mocap overlay ===")
    # The model's wrist body (at qpos=0) sits at the gravity-settled position, but the
    # cloud's mean rest-frame wrist is at a slightly different world location due to
    # per-frame shoulder variation. To make the comparison fair, shift the model via
    # sh_tx/ty/tz to put its wrist body at the mean-rest-frame registered wrist.
    cloud_wrist_mean = registered_full[:, 2, :].mean(0)  # mean over the same rest-frame subset
    print(f"  Mean rest-frame registered wrist: {cloud_wrist_mean*1000} mm")

    # Settle the OLD model to get the qpos joint values for sh_rotation, sh_extension,
    # sh_elv, elbow that the v7 registration is anchored to. Apply those same joint
    # angles to the v16 model so its wrist body lands in the same orientation as the
    # registration's reference.
    OLD = mujoco.MjModel.from_xml_path(MODEL_PATH)
    od = mujoco.MjData(OLD); od.qpos[:] = 0; mujoco.mj_forward(OLD, od)
    n_steps = int(2.0 / OLD.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(OLD, od)
    settled_qpos = {
        mujoco.mj_id2name(OLD, mujoco.mjtObj.mjOBJ_JOINT, j): od.qpos[OLD.jnt_qposadr[j]]
        for j in range(OLD.njnt)
    }
    print(f"  OLD-settled joint qpos (relevant): " +
          ", ".join(f"{k}={v:.4f}" for k, v in settled_qpos.items() if k in ("sh_rotation","sh_extension","sh_elv","elbow")))

    xml = build_xml(d3_local, d5_local, mocap_cloud=cloud, with_supination=False)
    tmp_xml = "/tmp/v17_mocap_overlay.xml"
    with open(tmp_xml, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(tmp_xml)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    # Copy settled-qpos values for matching joints in v16
    for jname, qval in settled_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = qval
    mujoco.mj_forward(model, data)
    # Now compute v16 wrist position with these joint angles applied, then add shoulder
    # translation to land v16 wrist at the cloud's mean rest wrist.
    v16_wrist_after_jset = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")].copy()
    sh_shift = cloud_wrist_mean - v16_wrist_after_jset
    print(f"  V16 wrist after applying settled rotations: {v16_wrist_after_jset*1000} mm")
    print(f"  sh_tx/ty/tz shift to align v16 wrist to cloud mean: {sh_shift*1000} mm")
    sh_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("sh_tx", "sh_ty", "sh_tz")]
    for i, jid in enumerate(sh_jnt_ids):
        data.qpos[model.jnt_qposadr[jid]] = sh_shift[i]
    mujoco.mj_forward(model, data)
    wrist_site = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "Wrist")]
    knC_site = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "KnuckleC")]
    knL_site = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "KnuckleL")]
    cloud_d3_mean = registered_full[:, 3, :].mean(0)
    cloud_d5_mean = registered_full[:, 4, :].mean(0)
    print(f"  Wrist site world: {wrist_site*1000} mm  (target: {cloud_wrist_mean*1000})")
    print(f"  KnuckleC site:    {knC_site*1000} mm  (cloud d3 mean: {cloud_d3_mean*1000})")
    print(f"  KnuckleL site:    {knL_site*1000} mm  (cloud d5 mean: {cloud_d5_mean*1000})")
    print(f"  Site-vs-cloud-mean residuals (mm): KnC={(knC_site - cloud_d3_mean)*1000}  KnL={(knL_site - cloud_d5_mean)*1000}")

    H, W = 720, 960
    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.sitegroup[0] = True

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wrist_pos = data.xpos[wrist_id].copy()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = wrist_pos + np.array([0.003, 0, 0])
    cam.distance = 0.025

    renders = {}
    views = [
        ( 90, -85, "v17_mocap_overlay_top.png",     "TOP-DOWN (dorsal view)"),
        ( 90,   0, "v17_mocap_overlay_distal.png",  "DISTAL (looking up the forearm)"),
        (  0,  60, "v17_mocap_overlay_oblique.png", "3/4 OBLIQUE"),
    ]
    for az, el, fname, caption in views:
        cam.azimuth = az
        cam.elevation = el
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        renders[fname] = (img, caption)
        iio.imwrite(fname, img)
        print(f"  -> {fname}")

    return renders


def make_review_composite(supination_path, mocap_renders, d3_local, d5_local):
    print("\n=== Composite review PNG ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.6, 2.0, 1.0])

    # Top row: supination strip
    ax_strip = fig.add_subplot(gs[0, :])
    ax_strip.imshow(mpimg.imread(supination_path))
    ax_strip.set_title("(1) Supination axis demo: hinge axis = +Y of wrist body (Green sphere). "
                       "Hand mesh rotates with supination if reparented to wrist body.",
                       fontsize=12)
    ax_strip.axis("off")

    # Middle row: 3 mocap-overlay views
    fnames = ["v17_mocap_overlay_top.png", "v17_mocap_overlay_distal.png", "v17_mocap_overlay_oblique.png"]
    for i, fname in enumerate(fnames):
        ax = fig.add_subplot(gs[1, i])
        img, caption = mocap_renders[fname]
        ax.imshow(img)
        ax.set_title(f"(2) Mocap d3/d5 cloud overlay — {caption}", fontsize=10)
        ax.axis("off")

    # Bottom: legend / numbers
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis("off")
    legend_text = (
        "WRIST-BODY FRAME (visible as small spheres on the wrist):\n"
        "    Red = +X axis    Green = +Y axis (proposed supination axis)    Blue = +Z axis     all 3 mm long\n"
        "\n"
        "MODEL SITES (proposed v17/v18 keypoint positions, attached to wrist body):\n"
        "    Yellow ball  = Wrist site  (origin of wrist body)\n"
        "    Cyan ball    = KnuckleC (d3) at data-derived offset\n"
        "    Magenta ball = KnuckleL (d5) at data-derived offset\n"
        "\n"
        "MOCAP CLOUD (200 frames spanning A36-1, registered to model frame, world-fixed sites):\n"
        "    Translucent cyan dots    = registered right_d3_mid 3D positions\n"
        "    Translucent magenta dots = registered right_d5_mid 3D positions\n"
        "\n"
        f"DATA-DERIVED OFFSETS (wrist-local mm):  KnuckleC = ({d3_local[0]*1000:+.2f}, {d3_local[1]*1000:+.2f}, {d3_local[2]*1000:+.2f})    "
        f"KnuckleL = ({d5_local[0]*1000:+.2f}, {d5_local[1]*1000:+.2f}, {d5_local[2]*1000:+.2f})\n"
        "\n"
        "Interpretation:\n"
        "  • If the bright cyan/magenta balls land in the densest region of the translucent cloud, the proposed offsets are good.\n"
        "  • If the cloud is offset from the balls, fit_offsets will move the marker sites toward the cloud.\n"
        "  • The cloud should rotate around the green axis when the mouse supinates — that's why we're adding the supination DOF.\n"
        "  • Hand meshes are still on the ulna body in this preview (v16 unchanged); v17 XML will reparent them to the wrist body."
    )
    ax_legend.text(0.01, 0.98, legend_text, fontsize=10, family="monospace",
                    verticalalignment="top", transform=ax_legend.transAxes)

    plt.suptitle("v17 supination DOF + d3/d5 keypoint placement REVIEW", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig("v17_proposal_REVIEW.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  -> v17_proposal_REVIEW.png")


def main():
    cloud, d3_local, d5_local, wrist_w, registered_rest = get_registered_d3_d5_cloud(n_frames=200)
    print(f"\nData-derived wrist-local offsets (mm):")
    print(f"  KnuckleC (d3_mid) = {d3_local*1000}")
    print(f"  KnuckleL (d5_mid) = {d5_local*1000}")

    render_supination_strip(d3_local, d5_local)
    mocap_renders = render_mocap_overlay(d3_local, d5_local, cloud, wrist_w, registered_rest)
    make_review_composite("v17_supination_strip.png", mocap_renders, d3_local, d5_local)
    print("\nDone.")


if __name__ == "__main__":
    main()
