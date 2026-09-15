"""Interactive v17 knuckle placement viewer using MuJoCo renders (not plotly Mesh3d).

http://localhost:8050

Sliders drive KnuckleC and KnuckleL site positions in V16 wrist-local frame.
On every slider change, the server rebuilds the XML with the current site offsets
and re-renders 4 viewpoints with MuJoCo. The result is shown as 4 PNG panels.

Save button writes current offsets to v17_knuckle_offsets_LATEST.json.
"""
from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import base64
import io
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import mujoco
import imageio.v3 as iio
from scipy.signal import savgol_filter
from dash import Dash, dcc, html, Input, Output, State

V16_XML = "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml"
SESSION_DIR = Path("/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video")
SID = "A36-1_2023-07-18_16-54-01_lightOff_video"
V7_FILE = SESSION_DIR / f"{SID}_mujoco_registered_forelimb_right_v7.h5"
DIGITS_FILE = SESSION_DIR / f"{SID}_digits_points3d.h5"
POINTS3D_FILE = SESSION_DIR / f"{SID}_points3d.h5"

H, W = 320, 420  # smaller renders for speed


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


def load_rest_cloud():
    """Return (rest_d2_mid, rest_d5_mid, sh_shift, v16_wp, v16_wR) in m + initial offsets."""
    with h5py.File(V7_FILE) as f:
        v7 = f["tracks"][:, 0, :, :].astype(np.float64)
        R_v7 = np.array(f.attrs["registration_rotation"])
    with h5py.File(POINTS3D_FILE) as f:
        raw3 = f["tracks"][:, 0, :, :].astype(np.float64)
    with h5py.File(DIGITS_FILE) as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        re = f["reproj_err"][:]
        nd = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    iC = nd.index("right_d2_mid"); iL = nd.index("right_d5_mid")
    vC = ~np.isnan(re[:, :, iC]).any(axis=0); vL = ~np.isnan(re[:, :, iL]).any(axis=0)
    valid = vC & vL

    raw5 = np.concatenate([raw3, digits[:, [iC, iL], :]], axis=1)
    raw5 = lerp(raw5); sm_mm = smooth(raw5)
    delta_C_cam = (sm_mm[:, 3, :] - sm_mm[:, 2, :]) / 1000.0
    delta_L_cam = (sm_mm[:, 4, :] - sm_mm[:, 2, :]) / 1000.0
    delta_C_world = (R_v7 @ delta_C_cam.T).T
    delta_L_world = (R_v7 @ delta_L_cam.T).T
    reg_w = v7[:, 2, :]
    reg_C = reg_w + delta_C_world
    reg_L = reg_w + delta_L_world

    vel = np.zeros(len(v7))
    vel[1:] = np.linalg.norm(np.diff(v7[:, 2, :], axis=0), axis=1)
    valid_idx = np.flatnonzero(valid)
    rest = valid_idx[np.argsort(vel[valid_idx])[:200]]

    cloud_W_mean = reg_w[rest].mean(0)
    cloud_C_mean = reg_C[rest].mean(0)
    cloud_L_mean = reg_L[rest].mean(0)

    # Get V16 wrist body world position + rotation at qpos=0
    m = mujoco.MjModel.from_xml_path(V16_XML)
    d = mujoco.MjData(m); d.qpos[:] = 0; mujoco.mj_forward(m, d)
    wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wp = d.xpos[wrist_id].copy()
    wR = d.xmat[wrist_id].reshape(3, 3).copy()
    sh_shift = cloud_W_mean - wp

    # Initial offsets from data (mean of rest cloud)
    init_C_local_mm = (wR.T @ (cloud_C_mean - cloud_W_mean)) * 1000
    init_L_local_mm = (wR.T @ (cloud_L_mean - cloud_W_mean)) * 1000

    return reg_C[rest], reg_L[rest], sh_shift, wp, wR, init_C_local_mm, init_L_local_mm


def build_xml(d3_local_m, d5_local_m):
    """Build a v16 XML with just the 3 marker sites + small wrist-axis triad."""
    with open(V16_XML) as f:
        xml = f.read()

    SPHERE_R = 0.0010
    sites = (
        f'              <site name="Wrist" type="sphere" size="{SPHERE_R}" rgba="1 1 0 1" pos="0 0 0"/>\n'
        f'              <site name="WristAxX" type="sphere" size="0.0003" rgba="1 0 0 1" pos="0.003 0 0"/>\n'
        f'              <site name="WristAxY" type="sphere" size="0.0003" rgba="0 0.8 0 1" pos="0 0.003 0"/>\n'
        f'              <site name="WristAxZ" type="sphere" size="0.0003" rgba="0 0 1 1" pos="0 0 0.003"/>\n'
        f'              <site name="KnuckleC" type="sphere" size="{SPHERE_R}" rgba="0 1 1 1" pos="{d3_local_m[0]:.6f} {d3_local_m[1]:.6f} {d3_local_m[2]:.6f}"/>\n'
        f'              <site name="KnuckleL" type="sphere" size="{SPHERE_R}" rgba="1 0 1 1" pos="{d5_local_m[0]:.6f} {d5_local_m[1]:.6f} {d5_local_m[2]:.6f}"/>\n'
    )
    pat = re.compile(
        r'<body name="wrist" pos="0 0\.01092 0">\s*<inertial[^/]+/>\s*</body>',
        re.MULTILINE | re.DOTALL,
    )
    new_block = (
        '<body name="wrist" pos="0 0.01092 0">\n'
        '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + sites + '            </body>'
    )
    xml = pat.sub(new_block, xml)
    return xml


def png_b64(img):
    buf = io.BytesIO()
    iio.imwrite(buf, img, extension=".png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_views(d3_local_mm, d5_local_mm, sh_shift):
    d3_m = np.array(d3_local_mm) / 1000
    d5_m = np.array(d5_local_mm) / 1000
    xml = build_xml(d3_m, d5_m)
    tmp = "/tmp/dash_v17_runtime.xml"
    with open(tmp, "w") as f:
        f.write(xml)
    model = mujoco.MjModel.from_xml_path(tmp)
    data = mujoco.MjData(model); data.qpos[:] = 0
    for j_name, val in zip(("sh_tx", "sh_ty", "sh_tz"), sh_shift):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=H, width=W)
    scene_opt = mujoco.MjvOption()
    scene_opt.sitegroup[0] = True

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wrist_pos = data.xpos[wrist_id].copy()
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = wrist_pos + np.array([0.005, 0, 0])
    cam.distance = 0.030

    images = {}
    for az, el, key in [(90, -85, "TOP"), (90, 0, "DISTAL"),
                         (0, 60, "OBLIQUE"), (0, 0, "LATERAL")]:
        cam.azimuth = az; cam.elevation = el
        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        img = renderer.render()
        images[key] = png_b64(img)
    renderer.close()
    return images


def main():
    print("Loading data + initial render...")
    cloud_C, cloud_L, sh_shift, wp, wR, init_C, init_L = load_rest_cloud()
    print(f"  cloud rest frames: {len(cloud_C)}  init KnuckleC mm = {init_C}  init KnuckleL mm = {init_L}")

    initial_imgs = render_views(init_C, init_L, sh_shift)

    app = Dash(__name__)

    def slider(slid, label, mn, mx, init, marks_step):
        # updatemode="mouseup" → only fires on release (avoids render backlog during drag)
        return html.Div([
            html.Label(label, id=f"{slid}-label"),
            dcc.Slider(id=slid, min=mn, max=mx, step=0.1, value=float(init),
                        marks={i: f"{i}" for i in range(mn, mx + 1, marks_step)},
                        updatemode="mouseup"),
        ])

    app.layout = html.Div([
        html.H2("v17 KnuckleC / KnuckleL placement (mujoco-rendered)"),
        html.P("Sliders update offsets in V16 wrist-local frame (mm). "
               "MuJoCo re-renders the 4 views below on each change. "
               "Yellow=Wrist | Cyan=KnuckleC (d2_mid medial) | Magenta=KnuckleL (d5_mid lateral) | "
               "Translucent dots = registered rest-frame mocap cloud."),

        html.Div([
            html.Div([
                html.Img(id="img-TOP", src=initial_imgs["TOP"], style={"width": "100%", "maxWidth": "420px"}),
                html.Div("TOP-DOWN", style={"textAlign": "center", "fontWeight": "bold", "fontSize": "12px"}),
            ], style={"width": "24%", "display": "inline-block", "padding": "2px"}),
            html.Div([
                html.Img(id="img-DISTAL", src=initial_imgs["DISTAL"], style={"width": "100%", "maxWidth": "420px"}),
                html.Div("DISTAL", style={"textAlign": "center", "fontWeight": "bold", "fontSize": "12px"}),
            ], style={"width": "24%", "display": "inline-block", "padding": "2px"}),
            html.Div([
                html.Img(id="img-OBLIQUE", src=initial_imgs["OBLIQUE"], style={"width": "100%", "maxWidth": "420px"}),
                html.Div("OBLIQUE", style={"textAlign": "center", "fontWeight": "bold", "fontSize": "12px"}),
            ], style={"width": "24%", "display": "inline-block", "padding": "2px"}),
            html.Div([
                html.Img(id="img-LATERAL", src=initial_imgs["LATERAL"], style={"width": "100%", "maxWidth": "420px"}),
                html.Div("LATERAL", style={"textAlign": "center", "fontWeight": "bold", "fontSize": "12px"}),
            ], style={"width": "24%", "display": "inline-block", "padding": "2px"}),
        ]),

        html.Hr(),
        html.Div([
            html.Div([
                html.H3("KnuckleC (cyan, medial)"),
                slider("knC-x", "X (lateral, +medial / -lateral)", -5, 5, init_C[0], 1),
                slider("knC-y", "Y (along forearm, +distal)", -2, 12, init_C[1], 2),
                slider("knC-z", "Z (dorsal, +up)", -5, 8, init_C[2], 2),
            ], style={"width": "48%", "display": "inline-block", "padding": "0 10px"}),
            html.Div([
                html.H3("KnuckleL (magenta, lateral)"),
                slider("knL-x", "X (lateral, +medial / -lateral)", -5, 5, init_L[0], 1),
                slider("knL-y", "Y (along forearm, +distal)", -2, 12, init_L[1], 2),
                slider("knL-z", "Z (dorsal, +up)", -5, 8, init_L[2], 2),
            ], style={"width": "48%", "display": "inline-block", "padding": "0 10px"}),
        ]),
        html.Hr(),
        html.Div([
            html.Button("💾 SAVE current offsets to JSON", id="save-btn", n_clicks=0,
                         style={"fontSize": "16px", "padding": "8px 16px", "marginRight": "10px",
                                "backgroundColor": "#4CAF50", "color": "white", "border": "none",
                                "borderRadius": "4px", "cursor": "pointer"}),
            html.Span(id="save-status", style={"fontFamily": "monospace", "color": "#333"}),
        ], style={"padding": "10px"}),
        html.Hr(),
        html.Div(id="output-summary",
                 style={"fontFamily": "monospace", "padding": "10px",
                        "backgroundColor": "#f4f4f4", "whiteSpace": "pre-wrap"}),
    ])

    # Single combined callback: slider release auto-renders mujoco views + updates labels
    @app.callback(
        [Output("img-TOP", "src"), Output("img-DISTAL", "src"),
         Output("img-OBLIQUE", "src"), Output("img-LATERAL", "src"),
         Output("output-summary", "children"),
         Output("knC-x-label", "children"), Output("knC-y-label", "children"), Output("knC-z-label", "children"),
         Output("knL-x-label", "children"), Output("knL-y-label", "children"), Output("knL-z-label", "children")],
        [Input("knC-x", "value"), Input("knC-y", "value"), Input("knC-z", "value"),
         Input("knL-x", "value"), Input("knL-y", "value"), Input("knL-z", "value")],
    )
    def update(cx, cy, cz, lx, ly, lz):
        imgs = render_views([cx, cy, cz], [lx, ly, lz], sh_shift)
        summary = (
            f"KnuckleC (wrist-local mm): X={cx:+.2f}  Y={cy:+.2f}  Z={cz:+.2f}\n"
            f"KnuckleL (wrist-local mm): X={lx:+.2f}  Y={ly:+.2f}  Z={lz:+.2f}\n"
            f"Lateral spread (X, mm): {abs(cx-lx):.2f}\n\n"
            f"v17 model XML site pos= attributes (in METERS):\n"
            f'  <site name="KnuckleC" pos="{cx/1000:.6f} {cy/1000:.6f} {cz/1000:.6f}"/>\n'
            f'  <site name="KnuckleL" pos="{lx/1000:.6f} {ly/1000:.6f} {lz/1000:.6f}"/>'
        )
        return (
            imgs["TOP"], imgs["DISTAL"], imgs["OBLIQUE"], imgs["LATERAL"],
            summary,
            f"X (lateral) = {cx:+.2f} mm",
            f"Y (along forearm) = {cy:+.2f} mm",
            f"Z (dorsal) = {cz:+.2f} mm",
            f"X (lateral) = {lx:+.2f} mm",
            f"Y (along forearm) = {ly:+.2f} mm",
            f"Z (dorsal) = {lz:+.2f} mm",
        )

    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        [State("knC-x", "value"), State("knC-y", "value"), State("knC-z", "value"),
         State("knL-x", "value"), State("knL-y", "value"), State("knL-z", "value")],
        prevent_initial_call=True,
    )
    def save(n_clicks, cx, cy, cz, lx, ly, lz):
        if not n_clicks:
            return ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"/mnt/vast/niyo/stac-mjx/v17_knuckle_offsets_{ts}.json")
        latest = Path("/mnt/vast/niyo/stac-mjx/v17_knuckle_offsets_LATEST.json")
        data = {
            "saved_at": ts,
            "session_used_for_cloud": SID,
            "frame_of_reference": "V16 wrist-local (mouse_forelimb_right_janelia_moving_shoulder_v2.xml at qpos=0)",
            "KnuckleC": {
                "source_landmark": "right_d2_mid",
                "local_mm": [cx, cy, cz],
                "local_m": [cx/1000, cy/1000, cz/1000],
                "site_pos_xml_attr": f'pos="{cx/1000:.6f} {cy/1000:.6f} {cz/1000:.6f}"',
            },
            "KnuckleL": {
                "source_landmark": "right_d5_mid",
                "local_mm": [lx, ly, lz],
                "local_m": [lx/1000, ly/1000, lz/1000],
                "site_pos_xml_attr": f'pos="{lx/1000:.6f} {ly/1000:.6f} {lz/1000:.6f}"',
            },
            "lateral_spread_mm": abs(cx - lx),
        }
        with open(path, "w") as f: json.dump(data, f, indent=2)
        with open(latest, "w") as f: json.dump(data, f, indent=2)
        return f"✓ Saved (#{n_clicks}) → {path.name} + v17_knuckle_offsets_LATEST.json"

    print("=" * 70)
    print("Open http://localhost:8050 — sliders update mujoco-rendered views live.")
    print("=" * 70)
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
