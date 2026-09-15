"""Honest visualization: use v16-fitted qpos from A36-1 trial000_ik.h5 to pose the v16
model in its actual fitted configuration, then overlay the mocap d3_mid/d5_mid cloud
for the corresponding session frames. The proposed KnuckleC/L sites at data-derived
offsets should now sit exactly on the cloud center.
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
V16_IK = "/mnt/vast/niyo/stac-mjx/final_STACed_5sessions_v16/A36-1_2023-07-18_16-54-01_lightOff/A36-1_2023-07-18_16-54-01_lightOff_trial000_ik.h5"
SESSION_DIR = Path("/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video")
SID = "A36-1_2023-07-18_16-54-01_lightOff_video"
V7_FILE = SESSION_DIR / f"{SID}_mujoco_registered_forelimb_right_v7.h5"
POINTS3D_FILE = SESSION_DIR / f"{SID}_points3d.h5"
DIGITS_FILE = SESSION_DIR / f"{SID}_digits_points3d.h5"

MODEL_PATH = (
    "/mnt/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/"
    "mouse_forelimb_right.xml"
)


def gravity_settled_pose(model_path: str, settle_time=2.0):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m); d.qpos[:] = 0; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    n = int(settle_time / m.opt.timestep)
    for _ in range(n): mujoco.mj_step(m, d)
    return (d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus")].copy(),
            d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ulna")].copy(),
            d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist")].copy())


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


def register_5kp(start_frame, n_frames=100):
    """Reproduce v7 registration for S/E/W and extend to d3_mid + d5_mid for given session frames."""
    sh_w, el_w, wr_w = gravity_settled_pose(MODEL_PATH)
    H_LEN = float(np.linalg.norm(el_w - sh_w))
    F_LEN = float(np.linalg.norm(wr_w - el_w))
    with h5py.File(V7_FILE) as f:
        R_v7 = np.array(f.attrs["registration_rotation"])
        sh_ref = np.array(f.attrs["shoulder_reference_mm"])
    with h5py.File(POINTS3D_FILE) as f:
        raw3 = f["tracks"][:, 0, :, :].astype(np.float64)
    with h5py.File(DIGITS_FILE) as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        nd = [n.decode() if isinstance(n,bytes) else n for n in f.attrs["node_names"]]
    i3 = nd.index("right_d3_mid"); i5 = nd.index("right_d5_mid")
    kps5 = np.concatenate([raw3, digits[:,[i3,i5],:]], axis=1)
    kps5 = lerp(kps5)
    sm_mm = smooth(kps5)
    centered = (sm_mm - sh_ref[None,None,:]) / 1000.0
    sh_pf = centered[:,0,:]
    hv = centered[:,1,:] - sh_pf; fv = centered[:,2,:] - centered[:,1,:]
    hn = np.maximum(np.linalg.norm(hv,axis=1,keepdims=True),1e-10)
    fn = np.maximum(np.linalg.norm(fv,axis=1,keepdims=True),1e-10)
    new_e = sh_pf + (hv/hn)*H_LEN
    new_w = new_e + (fv/fn)*F_LEN
    new_d3 = new_w + (centered[:,3,:] - centered[:,2,:])
    new_d5 = new_w + (centered[:,4,:] - centered[:,2,:])
    res = np.stack([sh_pf, new_e, new_w, new_d3, new_d5], axis=1)
    aligned = np.einsum("fkd,dj->fkj", res, R_v7)
    registered = aligned + sh_w[None,None,:]
    return registered[start_frame:start_frame+n_frames], wr_w


def load_v16_fit():
    with h5py.File(V16_IK) as f:
        qpos = f["qpos"][:]
        names_qpos = [n.decode() for n in f["names_qpos"]]
    return qpos, names_qpos


def determine_v16_clip_start():
    """The v16 run script uses trial_start + start (from trials_edited.csv) as start_frame.
    For A36-1 trial 0: trial_start=3799, start=128 → start_frame=3927, n_frames=100."""
    return 3927, 100


def build_xml(d3_local, d5_local, mocap_pts=None):
    """Patch v16 XML: add Wrist/KnuckleC/KnuckleL sites + axis triad. mocap_pts is
    optional list of (x,y,z) world coords + label tag for sites in worldbody."""
    with open(V16_XML) as f:
        xml = f.read()
    SPHERE_R = 0.0006  # match cloud size for honest visual comparison
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
        '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + sites
        + '            </body>'
    )
    xml = pat.sub(new_block, xml)

    if mocap_pts:
        cloud_block = []
        for i, (p, tag, color) in enumerate(mocap_pts):
            cloud_block.append(
                f'<site name="cloud_{tag}_{i:03d}" type="sphere" size="0.0003" '
                f'rgba="{color}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
            )
        joined = "\n      ".join(cloud_block)
        xml = xml.replace("</worldbody>", f"      <!-- mocap cloud -->\n      {joined}\n    </worldbody>")
    return xml


def render(d3_local, d5_local, qpos_v16, names_qpos_v16, registered, output_prefix="v17_with_v16fit"):
    print(f"\n=== Rendering with v16 fitted qpos (mean over trial000) ===")
    # Pick mid-trial frame for the pose (frame 25 of 50)
    mid_frame = qpos_v16.shape[0] // 2
    qpos_use = qpos_v16[mid_frame]
    print(f"  Using v16 fitted qpos from frame {mid_frame}: " +
          ", ".join(f"{n}={v:+.4f}" for n,v in zip(names_qpos_v16, qpos_use)))

    # Build mocap point list: 5 keypoints (S, E, W, KnuckleC, KnuckleL) at the same frame
    # PLUS the full 100-frame cloud of d3_mid + d5_mid
    pts = []
    # The current frame's mocap points (large, opaque)
    sess_frame_in_clip = mid_frame  # registered is the clip slice
    for i, (tag, color) in enumerate([
        ("Sh_now",  "1 0.4 0.4 0.95"),
        ("El_now",  "1 0.7 0 0.95"),
        ("Wr_now",  "1 1 0.6 0.95"),
        ("d3_now",  "0 0.9 0.9 0.95"),
        ("d5_now",  "0.9 0 0.9 0.95"),
    ]):
        p = registered[sess_frame_in_clip, i]
        pts.append((p, tag, color))
    # Plus the full cloud (small, translucent)
    for t in range(registered.shape[0]):
        pts.append((registered[t, 3], "d3_cloud", "0 0.6 0.6 0.4"))
        pts.append((registered[t, 4], "d5_cloud", "0.6 0 0.6 0.4"))

    xml = build_xml(d3_local, d5_local, mocap_pts=pts)
    tmp = "/tmp/v17_with_v16fit.xml"
    with open(tmp, "w") as f:
        f.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    # Apply v16 fitted qpos
    for n, v in zip(names_qpos_v16, qpos_use):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = v
    mujoco.mj_forward(model, data)

    # Compare site positions to mocap for sanity check
    print(f"\n  Sanity check at frame {mid_frame}:")
    for site, kp_idx, tag in [("Wrist", 2, "Wrist"), ("KnuckleC", 3, "d3"), ("KnuckleL", 4, "d5")]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        sw = data.site_xpos[sid]
        mw = registered[sess_frame_in_clip, kp_idx]
        print(f"    {site:9s} model_world = {sw*1000} mm  mocap_{tag}_world = {mw*1000} mm  delta = {(sw-mw)*1000} mm")

    # Render multi-view
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

    views = [
        (90, -85, "top", "TOP-DOWN"),
        (90,   0, "distal", "DISTAL"),
        ( 0,  60, "oblique", "3/4 OBLIQUE"),
        ( 0,   0, "lateral", "LATERAL"),
    ]
    rendered = {}
    for az, el, key, caption in views:
        cam.azimuth = az; cam.elevation = el
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        fname = f"{output_prefix}_{key}.png"
        iio.imwrite(fname, img)
        rendered[key] = (img, caption)
        print(f"  -> {fname}")

    # Composite
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (key, (img, cap)) in zip(axes.ravel(), rendered.items()):
        ax.imshow(img)
        ax.set_title(cap, fontsize=12)
        ax.axis("off")
    plt.suptitle("v17 proposal with v16-FITTED qpos (A36-1 trial000 mid-frame)\n"
                  "Yellow=Wrist site | Cyan=KnuckleC (d3_mid) | Magenta=KnuckleL (d5_mid) | "
                  "Translucent dots = full d3/d5 cloud | Bright dots = mocap at this frame",
                  fontsize=11, y=0.99)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ALL.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> {output_prefix}_ALL.png")


def main():
    qpos_v16, names_v16 = load_v16_fit()
    print(f"v16 ik.h5 has {qpos_v16.shape[0]} frames, joints: {names_v16}")

    sf, n = determine_v16_clip_start()
    print(f"v16 trial000 was on session frames [{sf}, {sf+n})")

    registered, wr_w = register_5kp(sf, n_frames=n)
    print(f"Registered 5kp clip shape: {registered.shape}")
    print(f"Mean wrist over clip: {registered[:,2].mean(0)*1000} mm")
    print(f"Mean d3 over clip:    {registered[:,3].mean(0)*1000} mm")

    # Compute data-derived offsets in wrist-local frame using v16 fitted wrist orientation
    # at the corresponding clip frame. Use frame mid-trial.
    OLD = mujoco.MjModel.from_xml_path(MODEL_PATH)
    od = mujoco.MjData(OLD); od.qpos[:] = 0; od.qvel[:] = 0
    mujoco.mj_forward(OLD, od)
    n_steps = int(2.0 / OLD.opt.timestep)
    for _ in range(n_steps): mujoco.mj_step(OLD, od)
    wrist_R_old = od.xmat[mujoco.mj_name2id(OLD, mujoco.mjtObj.mjOBJ_BODY, "wrist")].reshape(3,3).copy()

    # Use the v16 fitted qpos to get the correct v16 wrist orientation at mid-clip frame
    # (for consistency with rendering)
    V16 = mujoco.MjModel.from_xml_path(V16_XML)
    vd = mujoco.MjData(V16); vd.qpos[:] = 0
    for n_, v in zip(names_v16, qpos_v16[qpos_v16.shape[0]//2]):
        jid = mujoco.mj_name2id(V16, mujoco.mjtObj.mjOBJ_JOINT, n_)
        if jid >= 0: vd.qpos[V16.jnt_qposadr[jid]] = v
    mujoco.mj_forward(V16, vd)
    wrist_R_v16 = vd.xmat[mujoco.mj_name2id(V16, mujoco.mjtObj.mjOBJ_BODY, "wrist")].reshape(3,3).copy()
    wrist_p_v16 = vd.xpos[mujoco.mj_name2id(V16, mujoco.mjtObj.mjOBJ_BODY, "wrist")].copy()

    # Compute offsets so that site_world (rendered) = mocap_world for the frame
    mid = qpos_v16.shape[0] // 2
    d3_world_offset = registered[mid, 3] - wrist_p_v16
    d5_world_offset = registered[mid, 4] - wrist_p_v16
    d3_local = wrist_R_v16.T @ d3_world_offset
    d5_local = wrist_R_v16.T @ d5_world_offset
    print(f"\nWrist body world (v16-fitted at mid frame): {wrist_p_v16*1000} mm")
    print(f"Mocap wrist world at mid frame:              {registered[mid,2]*1000} mm")
    print(f"  delta wrist body vs mocap wrist: {(wrist_p_v16-registered[mid,2])*1000} mm")
    print(f"\nd3_local (mocap d3 - wrist body, in v16 wrist-local): {d3_local*1000} mm")
    print(f"d5_local (mocap d5 - wrist body, in v16 wrist-local): {d5_local*1000} mm")

    render(d3_local, d5_local, qpos_v16, names_v16, registered)
    print("\nDone.")


if __name__ == "__main__":
    main()
