"""Clean visualization: compute everything in V16 model's wrist-local frame at qpos=0.
The sites will land EXACTLY on the cloud center by construction (no model mismatch).
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


def get_registered_cloud_v7_extended():
    """Use v7 file directly for S/E/W (proven correct) and extend with d3_mid/d5_mid by
    applying the SAME R + bone-rescaling logic to the digit data."""
    # Load v7 stored data (proven correct, exact)
    with h5py.File(V7_FILE) as f:
        v7_tracks = f["tracks"][:, 0, :, :].astype(np.float64)  # (T, 3, 3) m, registered S/E/W
        R_v7 = np.array(f.attrs["registration_rotation"])
        sh_ref_mm = np.array(f.attrs["shoulder_reference_mm"])
        F_LEN = float(f.attrs["target_forearm_len"])
        H_LEN = float(f.attrs["target_humerus_len"])
        sh_w = np.array(f.attrs["registration_translation"])  # MODEL_SHOULDER

    # Load raw 3kp + digits to compute knuckle deltas
    with h5py.File(SESSION_DIR / f"{SID}_points3d.h5") as f:
        raw3 = f["tracks"][:, 0, :, :].astype(np.float64)  # mm
    with h5py.File(DIGITS_FILE) as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        re = f["reproj_err"][:]
        nd = [n.decode() if isinstance(n,bytes) else n for n in f.attrs["node_names"]]
    i3 = nd.index("right_d2_mid"); i5 = nd.index("right_d5_mid")
    v_d3 = ~np.isnan(re[:,:,i3]).any(axis=0)
    v_d5 = ~np.isnan(re[:,:,i5]).any(axis=0)
    v_both = v_d3 & v_d5

    # Smooth raw d3/d5 same way as v7 recipe (savgol w=15 p=3)
    raw5 = np.concatenate([raw3, digits[:,[i3,i5],:]], axis=1)  # (T, 5, 3) raw mm
    raw5 = lerp(raw5)
    sm_mm = smooth(raw5)

    # Compute knuckle delta = (raw_dN - raw_wrist) per frame, in m, in CAMERA frame
    delta_d3_cam = (sm_mm[:, 3, :] - sm_mm[:, 2, :]) / 1000.0  # m
    delta_d5_cam = (sm_mm[:, 4, :] - sm_mm[:, 2, :]) / 1000.0

    # Rotate to model frame using v7's R
    delta_d3_world = (R_v7 @ delta_d3_cam.T).T  # (T, 3) m
    delta_d5_world = (R_v7 @ delta_d5_cam.T).T

    # Add to v7's registered wrist position
    reg_wrist = v7_tracks[:, 2, :]  # (T, 3) m
    reg_d3 = reg_wrist + delta_d3_world
    reg_d5 = reg_wrist + delta_d5_world

    # Stack 5kp registered
    registered = np.concatenate([v7_tracks, reg_d3[:, None, :], reg_d5[:, None, :]], axis=1)  # (T, 5, 3)
    return registered, v_both, sh_w


def main():
    registered, valid, _ = get_registered_cloud_v7_extended()
    print(f"Registered 5kp shape: {registered.shape}")

    # Pick rest frames (lowest velocity wrist + d3/d5 valid)
    vel = np.zeros(len(registered))
    vel[1:] = np.linalg.norm(np.diff(registered[:, 2, :], axis=0), axis=1)
    valid_idx = np.flatnonzero(valid)
    rest = valid_idx[np.argsort(vel[valid_idx])[:200]]
    print(f"Rest cloud: {len(rest)} frames")
    print(f"  mean S = {registered[rest, 0].mean(0)*1000} mm")
    print(f"  mean E = {registered[rest, 1].mean(0)*1000} mm")
    print(f"  mean W = {registered[rest, 2].mean(0)*1000} mm")
    print(f"  mean d3_mid = {registered[rest, 3].mean(0)*1000} mm")
    print(f"  mean d5_mid = {registered[rest, 4].mean(0)*1000} mm")

    # Now use V16 model at qpos=0 directly (NO settling, NO OLD model)
    model = mujoco.MjModel.from_xml_path(V16_XML)
    data = mujoco.MjData(model); data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    v16_wp = data.xpos[wrist_id].copy()
    v16_wR = data.xmat[wrist_id].reshape(3,3).copy()
    print(f"\nV16 model wrist body at qpos=0: pos = {v16_wp*1000} mm")

    # Compute sh_tx/ty/tz to translate v16 wrist to cloud_wrist_mean
    cloud_wrist_mean = registered[rest, 2].mean(0)
    sh_shift = cloud_wrist_mean - v16_wp
    print(f"Shift v16 to align wrist with cloud: {sh_shift*1000} mm")

    # Now compute d3_local in V16 wrist-local frame at qpos=0:
    # We want bright_KnuckleC site world = cloud_d3_mean (rest cloud center)
    # site_world = (v16_wp + sh_shift) + v16_wR @ d3_local
    # cloud_d3_mean = (v16_wp + sh_shift) + v16_wR @ d3_local
    # => d3_local = v16_wR.T @ (cloud_d3_mean - cloud_wrist_mean)  [since v16_wp + sh_shift = cloud_wrist_mean]
    d3_offset_world = registered[rest, 3].mean(0) - cloud_wrist_mean
    d5_offset_world = registered[rest, 4].mean(0) - cloud_wrist_mean
    d3_local = v16_wR.T @ d3_offset_world
    d5_local = v16_wR.T @ d5_offset_world
    print(f"\nd3_local (in V16 wrist-local): {d3_local*1000} mm  (norm {np.linalg.norm(d3_local)*1000:.2f})")
    print(f"d5_local (in V16 wrist-local): {d5_local*1000} mm  (norm {np.linalg.norm(d5_local)*1000:.2f})")

    # Build XML with sites + cloud
    cloud_pts = []
    for t in rest:
        cloud_pts.append((registered[t, 3], "d3", "0 0.65 0.85 0.5"))   # d3 cloud
        cloud_pts.append((registered[t, 4], "d5", "0.85 0 0.65 0.5"))   # d5 cloud

    SPHERE_R = 0.0006
    sites_xml = (
        f'              <site name="Wrist" type="sphere" size="{SPHERE_R}" rgba="1 1 0 1" pos="0 0 0"/>\n'
        f'              <site name="WristAxX" type="sphere" size="0.0003" rgba="1 0 0 1" pos="0.003 0 0"/>\n'
        f'              <site name="WristAxY" type="sphere" size="0.0003" rgba="0 0.8 0 1" pos="0 0.003 0"/>\n'
        f'              <site name="WristAxZ" type="sphere" size="0.0003" rgba="0 0 1 1" pos="0 0 0.003"/>\n'
        f'              <site name="KnuckleC" type="sphere" size="{SPHERE_R}" rgba="0 1 1 1" pos="{d3_local[0]:.6f} {d3_local[1]:.6f} {d3_local[2]:.6f}"/>\n'
        f'              <site name="KnuckleL" type="sphere" size="{SPHERE_R}" rgba="1 0 1 1" pos="{d5_local[0]:.6f} {d5_local[1]:.6f} {d5_local[2]:.6f}"/>\n'
    )
    with open(V16_XML) as f:
        xml = f.read()
    pat = re.compile(r'<body name="wrist" pos="0 0\.01092 0">\s*<inertial[^/]+/>\s*</body>',
                     re.MULTILINE | re.DOTALL)
    new_block = (
        '<body name="wrist" pos="0 0.01092 0">\n'
        '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + sites_xml + '            </body>'
    )
    xml = pat.sub(new_block, xml)
    cloud_block_lines = [
        f'<site name="cloud_{tag}_{i:04d}" type="sphere" size="0.00025" rgba="{color}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
        for i, (p, tag, color) in enumerate(cloud_pts)
    ]
    cloud_joined = "\n      ".join(cloud_block_lines)
    xml = xml.replace("</worldbody>", f"      <!-- cloud -->\n      {cloud_joined}\n    </worldbody>")

    tmp = "/tmp/viz_clean.xml"
    with open(tmp, "w") as f: f.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    data = mujoco.MjData(model); data.qpos[:] = 0
    # Apply shoulder shift
    for j_name, val in zip(("sh_tx","sh_ty","sh_tz"), sh_shift):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)

    # Sanity check
    knC = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "KnuckleC")]
    knL = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "KnuckleL")]
    cd3m = registered[rest, 3].mean(0); cd5m = registered[rest, 4].mean(0)
    print(f"\nSanity:")
    print(f"  KnuckleC site:    {knC*1000} mm")
    print(f"  cloud d3 mean:    {cd3m*1000} mm")
    print(f"  residual:         {(knC-cd3m)*1000} mm  (should be ~0)")
    print(f"  KnuckleL site:    {knL*1000} mm")
    print(f"  cloud d5 mean:    {cd5m*1000} mm")
    print(f"  residual:         {(knL-cd5m)*1000} mm")

    # Render
    H, W = 720, 960
    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.sitegroup[0] = True

    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.xpos[wrist_id] + np.array([0.003, 0, 0])
    cam.distance = 0.025

    rendered = {}
    for az, el, key, cap in [(90, -85, "top", "TOP-DOWN"),
                              (90,   0, "distal", "DISTAL"),
                              ( 0,  60, "oblique", "3/4 OBLIQUE"),
                              ( 0,   0, "lateral", "LATERAL")]:
        cam.azimuth = az; cam.elevation = el
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        fname = f"viz_clean_{key}.png"
        iio.imwrite(fname, img)
        rendered[key] = (img, cap)
        print(f"  -> {fname}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (key, (img, cap)) in zip(axes.ravel(), rendered.items()):
        ax.imshow(img); ax.set_title(cap, fontsize=12); ax.axis("off")
    plt.suptitle(
        "v17 Knuckle site placement (CLEAN: V16 model qpos=0 + sh shift to align wrist to rest mean)\n"
        f"KnuckleC (cyan) at +{d3_local[0]*1000:.2f},+{d3_local[1]*1000:.2f},+{d3_local[2]*1000:.2f} mm  | "
        f"KnuckleL (magenta) at +{d5_local[0]*1000:.2f},+{d5_local[1]*1000:.2f},+{d5_local[2]*1000:.2f} mm  | "
        "cloud = 200 rest frames",
        fontsize=10, y=0.99)
    plt.tight_layout()
    plt.savefig("viz_clean_ALL.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  -> viz_clean_ALL.png")


if __name__ == "__main__":
    main()
