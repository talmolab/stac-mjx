"""Final v17 site placement preview with d2_mid + d5_mid (true left-right paw spread).

Outputs:
  v17_final_supination.png        — 5-frame supination axis demo with hand reparented
  v17_final_overlay_top.png       — top-down mocap cloud overlay
  v17_final_overlay_distal.png    — distal (looking up forearm)
  v17_final_overlay_oblique.png   — 3/4 oblique
  v17_final_overlay_lateral.png   — lateral
  v17_final_REVIEW.png            — composite

Design:
  - Compute d2_mid / d5_mid offsets directly from v7 file's stored registered
    wrist + raw digit deltas (no model-mismatch issues).
  - Render with V16 model at qpos=0 + sh_tx/ty/tz shift so the wrist body lands
    exactly at the cloud's mean rest wrist position.
  - For supination strip, attach hand mesh geoms to the wrist body so they
    rotate with the supination DOF.
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
DIGITS_FILE = SESSION_DIR / f"{SID}_digits_points3d.h5"
POINTS3D_FILE = SESSION_DIR / f"{SID}_points3d.h5"


def lerp(arr):
    out = arr.copy(); T,K,D = out.shape
    for k in range(K):
        for d in range(D):
            x = out[:,k,d]; nan = np.isnan(x)
            if not nan.any() or nan.all(): continue
            idx = np.flatnonzero(nan); ok = np.flatnonzero(~nan)
            x[idx] = np.interp(idx, ok, x[ok]); out[:,k,d] = x
    return out


def smooth(p, w=15, po=3):
    out = p.copy()
    for k in range(p.shape[1]):
        for d in range(3):
            out[:,k,d] = savgol_filter(p[:,k,d], w, po)
    return out


def get_registered_5kp(d_med_name="right_d2_mid", d_lat_name="right_d5_mid"):
    """Return (T, 5, 3) registered: [S, E, W, KnuckleC=medial, KnuckleL=lateral]"""
    with h5py.File(V7_FILE) as f:
        v7 = f["tracks"][:, 0, :, :].astype(np.float64)
        R_v7 = np.array(f.attrs["registration_rotation"])
    with h5py.File(POINTS3D_FILE) as f:
        raw3 = f["tracks"][:, 0, :, :].astype(np.float64)
    with h5py.File(DIGITS_FILE) as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        re = f["reproj_err"][:]
        nd = [n.decode() if isinstance(n,bytes) else n for n in f.attrs["node_names"]]
    iC = nd.index(d_med_name); iL = nd.index(d_lat_name)
    vC = ~np.isnan(re[:,:,iC]).any(axis=0); vL = ~np.isnan(re[:,:,iL]).any(axis=0)
    valid = vC & vL

    raw5 = np.concatenate([raw3, digits[:,[iC, iL],:]], axis=1)
    raw5 = lerp(raw5)
    sm_mm = smooth(raw5)
    delta_C_cam = (sm_mm[:, 3, :] - sm_mm[:, 2, :]) / 1000.0
    delta_L_cam = (sm_mm[:, 4, :] - sm_mm[:, 2, :]) / 1000.0
    delta_C_world = (R_v7 @ delta_C_cam.T).T
    delta_L_world = (R_v7 @ delta_L_cam.T).T
    reg_w = v7[:, 2, :]
    reg_C = reg_w + delta_C_world
    reg_L = reg_w + delta_L_world
    registered = np.concatenate([v7, reg_C[:, None, :], reg_L[:, None, :]], axis=1)
    return registered, valid


def reparent_hand_meshes_xml(xml: str) -> tuple[str, str]:
    """Pull all Retopo_* hand mesh geoms from ulna body, return (xml_without_them, geoms_block_for_wrist).
    geoms_block_for_wrist has each mesh re-tagged with pos='0 -0.01092 0' to compensate for the
    wrist body's offset relative to ulna."""
    retopo_pat = re.compile(
        r'^[ \t]*<geom type="mesh" mesh="Retopo_[^"]+" class="hand_visual"/>\s*\n',
        re.MULTILINE,
    )
    lines = retopo_pat.findall(xml)
    xml = retopo_pat.sub("", xml)
    new_lines = []
    for line in lines:
        m = re.search(r'mesh="(Retopo_[^"]+)"', line)
        if m:
            new_lines.append(
                f'              <geom type="mesh" mesh="{m.group(1)}" '
                f'class="hand_visual" pos="0 -0.01092 0"/>'
            )
    return xml, "\n".join(new_lines)


def build_xml(d3_local, d5_local, with_supination=True, mocap_pts=None):
    with open(V16_XML) as f:
        xml = f.read()

    xml, hand_block = reparent_hand_meshes_xml(xml)

    SPHERE_R = 0.0008
    if with_supination:
        joint = ('              <joint name="forearm_supination" type="hinge" '
                 'pos="0 0 0" axis="0 1 0" range="-90 90" damping="0.01"/>\n')
    else:
        joint = ""
    sites = (
        f'              <site name="Wrist" type="sphere" size="{SPHERE_R}" rgba="1 1 0 1" pos="0 0 0"/>\n'
        f'              <site name="WristAxX" type="sphere" size="0.0003" rgba="1 0 0 1" pos="0.003 0 0"/>\n'
        f'              <site name="WristAxY" type="sphere" size="0.0003" rgba="0 0.8 0 1" pos="0 0.003 0"/>\n'
        f'              <site name="WristAxZ" type="sphere" size="0.0003" rgba="0 0 1 1" pos="0 0 0.003"/>\n'
        f'              <site name="KnuckleC" type="sphere" size="{SPHERE_R}" rgba="0 1 1 1" pos="{d3_local[0]:.6f} {d3_local[1]:.6f} {d3_local[2]:.6f}"/>\n'
        f'              <site name="KnuckleL" type="sphere" size="{SPHERE_R}" rgba="1 0 1 1" pos="{d5_local[0]:.6f} {d5_local[1]:.6f} {d5_local[2]:.6f}"/>\n'
    )
    pat = re.compile(
        r'<body name="wrist" pos="0 0\.01092 0">\s*<inertial[^/]+/>\s*</body>',
        re.MULTILINE | re.DOTALL,
    )
    new_block = (
        '<body name="wrist" pos="0 0.01092 0">\n'
        + joint
        + '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + sites
        + "\n" + hand_block + "\n"
        + '            </body>'
    )
    xml = pat.sub(new_block, xml)

    if mocap_pts:
        cloud_lines = [
            f'<site name="cloud_{tag}_{i:04d}" type="sphere" size="0.00025" rgba="{color}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
            for i, (p, tag, color) in enumerate(mocap_pts)
        ]
        joined = "\n      ".join(cloud_lines)
        xml = xml.replace("</worldbody>", f"      <!-- cloud -->\n      {joined}\n    </worldbody>")
    return xml


def get_v16_wrist_at_qpos0():
    m = mujoco.MjModel.from_xml_path(V16_XML)
    d = mujoco.MjData(m); d.qpos[:] = 0; mujoco.mj_forward(m, d)
    wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    return d.xpos[wrist_id].copy(), d.xmat[wrist_id].reshape(3,3).copy()


def render_supination_strip(d3_local, d5_local, sh_shift, output_path):
    print("\n=== Supination strip ===")
    xml = build_xml(d3_local, d5_local, with_supination=True)
    tmp = "/tmp/v17_sup.xml"
    with open(tmp, "w") as f: f.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    data = mujoco.MjData(model)
    sup_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "forearm_supination")
    sup_idx = model.jnt_qposadr[sup_jnt]

    H, W = 540, 720
    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.sitegroup[0] = True

    angles = [-60, -30, 0, 30, 60]
    frames = []
    sh_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("sh_tx", "sh_ty", "sh_tz")]
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")

    for ang in angles:
        data.qpos[:] = 0
        for i, jid in enumerate(sh_jnt_ids):
            data.qpos[model.jnt_qposadr[jid]] = sh_shift[i]
        data.qpos[sup_idx] = np.deg2rad(ang)
        mujoco.mj_forward(model, data)
        wrist_pos = data.xpos[wrist_id]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = wrist_pos + np.array([0.005, 0, 0])
        cam.distance = 0.025
        cam.azimuth = 90; cam.elevation = -30
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        frames.append((ang, img))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    for ax, (ang, img) in zip(axes, frames):
        ax.imshow(img)
        ax.set_title(f"forearm_supination = {ang:+d}°", fontsize=12)
        ax.axis("off")
    plt.suptitle("Proposed v17 supination axis (hinge on wrist body, axis = +Y of wrist-local = forearm long axis)\n"
                 f"Yellow=Wrist  |  Cyan=KnuckleC (d2_mid, MEDIAL +X)  |  Magenta=KnuckleL (d5_mid, LATERAL -X)\n"
                 "Hand mesh reparented onto wrist body so it rotates with supination",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> {output_path}")


def render_overlay(d3_local, d5_local, sh_shift, registered_rest, output_prefix):
    print("\n=== Mocap overlay ===")
    cloud_pts = []
    for t in range(registered_rest.shape[0]):
        cloud_pts.append((registered_rest[t, 3], "C", "0 0.65 0.85 0.5"))
        cloud_pts.append((registered_rest[t, 4], "L", "0.85 0 0.65 0.5"))

    xml = build_xml(d3_local, d5_local, with_supination=False, mocap_pts=cloud_pts)
    tmp = "/tmp/v17_overlay.xml"
    with open(tmp, "w") as f: f.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    data = mujoco.MjData(model); data.qpos[:] = 0
    sh_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("sh_tx", "sh_ty", "sh_tz")]
    for i, jid in enumerate(sh_jnt_ids):
        data.qpos[model.jnt_qposadr[jid]] = sh_shift[i]
    mujoco.mj_forward(model, data)

    knC = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "KnuckleC")]
    knL = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "KnuckleL")]
    cm = registered_rest[:, 3].mean(0); cl = registered_rest[:, 4].mean(0)
    print(f"  Sanity: KnC site - cloud mean = {(knC-cm)*1000} mm  (target 0)")
    print(f"  Sanity: KnL site - cloud mean = {(knL-cl)*1000} mm  (target 0)")

    H, W = 720, 960
    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.sitegroup[0] = True

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.xpos[wrist_id] + np.array([0.005, 0, 0])
    cam.distance = 0.030

    rendered = {}
    for az, el, key, cap in [
        (90, -85, "top", "TOP-DOWN (dorsal view)"),
        (90,   0, "distal", "DISTAL (down forearm)"),
        ( 0,  60, "oblique", "3/4 OBLIQUE"),
        ( 0,   0, "lateral", "LATERAL"),
    ]:
        cam.azimuth = az; cam.elevation = el
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        fname = f"{output_prefix}_{key}.png"
        iio.imwrite(fname, img)
        rendered[key] = (img, cap)
        print(f"  -> {fname}")
    return rendered


def make_review(d3_local, d5_local, supination_path, overlay_renders, output):
    print("\n=== Review composite ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.8, 0.9])
    ax_strip = fig.add_subplot(gs[0, :])
    ax_strip.imshow(mpimg.imread(supination_path))
    ax_strip.set_title("(1) Supination axis demo (5 angles): hand mesh rotates with the proposed forearm_supination joint",
                       fontsize=12)
    ax_strip.axis("off")

    keys = ["top", "distal", "oblique", "lateral"]
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[1, i])
        img, cap = overlay_renders[key]
        ax.imshow(img); ax.set_title(f"(2) {cap}", fontsize=10); ax.axis("off")

    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis("off")
    legend = (
        "PROPOSED v17 KEYPOINTS (option a):\n"
        "  KnuckleC = right_d2_mid (medial / thumb-side)\n"
        "  KnuckleL = right_d5_mid (lateral / pinky-side)\n"
        f"  Offsets in V16 wrist-local (mm): KnuckleC = ({d3_local[0]*1000:+.2f}, {d3_local[1]*1000:+.2f}, {d3_local[2]*1000:+.2f})    "
        f"KnuckleL = ({d5_local[0]*1000:+.2f}, {d5_local[1]*1000:+.2f}, {d5_local[2]*1000:+.2f})\n"
        f"  Lateral spread (cyan ↔ magenta in X): {abs(d3_local[0]-d5_local[0])*1000:.2f} mm — true left-right across the paw\n"
        "\n"
        "VIZ NOTES:\n"
        "  Yellow ball = Wrist site (origin of v17 wrist body)\n"
        "  Red/Green/Blue small balls = +X / +Y / +Z axes of wrist-local frame (3 mm long); +Y = supination axis\n"
        "  Big cyan/magenta balls = proposed model sites at data-derived offsets\n"
        "  Translucent dots = 200-frame rest cloud of registered d2_mid (cyan) and d5_mid (magenta)\n"
        "\n"
        "By construction, big bright balls sit at the cloud-mean position; if they don't visually overlap, the cloud has spread (motion variability)."
    )
    ax_legend.text(0.01, 0.98, legend, fontsize=10, family="monospace",
                    verticalalignment="top", transform=ax_legend.transAxes)
    plt.suptitle("v17 site placement REVIEW — d2_mid (medial) + d5_mid (lateral)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  -> {output}")


def main():
    registered, valid = get_registered_5kp("right_d2_mid", "right_d5_mid")
    print(f"5kp registered shape: {registered.shape}")

    vel = np.zeros(len(registered))
    vel[1:] = np.linalg.norm(np.diff(registered[:, 2, :], axis=0), axis=1)
    valid_idx = np.flatnonzero(valid)
    rest_idx = valid_idx[np.argsort(vel[valid_idx])[:200]]
    rest = registered[rest_idx]
    print(f"Rest cloud: {len(rest)} frames")

    cloud_W_mean = rest[:, 2].mean(0)
    cloud_C_mean = rest[:, 3].mean(0)
    cloud_L_mean = rest[:, 4].mean(0)
    print(f"  mean rest W   = {cloud_W_mean*1000} mm")
    print(f"  mean rest d2m = {cloud_C_mean*1000} mm")
    print(f"  mean rest d5m = {cloud_L_mean*1000} mm")

    v16_wp, v16_wR = get_v16_wrist_at_qpos0()
    sh_shift = cloud_W_mean - v16_wp
    print(f"V16 wrist (qpos=0): {v16_wp*1000} mm")
    print(f"sh_tx/ty/tz shift:  {sh_shift*1000} mm")

    d3_offset_world = cloud_C_mean - cloud_W_mean
    d5_offset_world = cloud_L_mean - cloud_W_mean
    d3_local = v16_wR.T @ d3_offset_world
    d5_local = v16_wR.T @ d5_offset_world
    print(f"\nd2_mid as KnuckleC, V16 wrist-local: {d3_local*1000} mm")
    print(f"d5_mid as KnuckleL, V16 wrist-local: {d5_local*1000} mm")
    print(f"Lateral spread (X): {abs(d3_local[0]-d5_local[0])*1000:.2f} mm")

    sup_path = "v17_final_supination.png"
    render_supination_strip(d3_local, d5_local, sh_shift, sup_path)
    overlays = render_overlay(d3_local, d5_local, sh_shift, rest, "v17_final_overlay")
    make_review(d3_local, d5_local, sup_path, overlays, "v17_final_REVIEW.png")

    print("\nDone. Files:")
    for p in sorted(Path(".").glob("v17_final_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
