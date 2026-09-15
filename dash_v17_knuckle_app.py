"""Interactive browser viewer for placing v17 KnuckleC/KnuckleL sites.

Run:
    python dash_v17_knuckle_app.py
Then open http://localhost:8050 (or the host:port shown in the log).

Sliders adjust KnuckleC and KnuckleL offsets in V16 wrist-local frame (mm).
Drag them to position the cyan/magenta balls onto the hand mesh / mocap cloud.
The HTML shows you the current mm offsets for the v17 model XML.
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
import json
from datetime import datetime
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch, no_update

V16_XML = "/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml"
SESSION_DIR = Path("/mnt/vast/eric/azim-emg-data/A36-1_2023-07-18_16-54-01_lightOff_video")
SID = "A36-1_2023-07-18_16-54-01_lightOff_video"
V7_FILE = SESSION_DIR / f"{SID}_mujoco_registered_forelimb_right_v7.h5"
DIGITS_FILE = SESSION_DIR / f"{SID}_digits_points3d.h5"
POINTS3D_FILE = SESSION_DIR / f"{SID}_points3d.h5"


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


def load_data(med_node="right_d2_mid", lat_node="right_d5_mid"):
    with h5py.File(V7_FILE) as f:
        v7 = f["tracks"][:, 0, :, :].astype(np.float64)
        R_v7 = np.array(f.attrs["registration_rotation"])
    with h5py.File(POINTS3D_FILE) as f:
        raw3 = f["tracks"][:, 0, :, :].astype(np.float64)
    with h5py.File(DIGITS_FILE) as f:
        digits = f["tracks"][:, 0, :, :].astype(np.float64)
        re = f["reproj_err"][:]
        nd = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]

    # Compute registered cloud for each digit landmark we might want
    interesting_landmarks = [f"right_d{d}_{k}" for d in range(2, 6) for k in ["knuckle", "mid", "tip"]]
    landmark_clouds = {}
    raw_w_smooth = smooth(lerp(raw3))[:, 2, :]  # (T, 3) m smoothed wrist
    for lm in interesting_landmarks:
        if lm not in nd: continue
        idx = nd.index(lm)
        raw_lm = digits[:, idx, :].copy()
        valid = ~np.isnan(raw_lm).any(axis=1)
        # Smooth via lerp + savgol
        raw5 = np.concatenate([raw3, digits[:, [idx], :]], axis=1)
        raw5 = lerp(raw5); sm_mm = smooth(raw5)
        delta_cam = (sm_mm[:, 3, :] - sm_mm[:, 2, :]) / 1000.0  # m
        delta_world = (R_v7 @ delta_cam.T).T
        registered = v7[:, 2, :] + delta_world  # (T, 3) m
        landmark_clouds[lm] = (registered, valid)

    return v7, landmark_clouds, R_v7


def get_v16_state():
    """V16 model wrist body world position and rotation at qpos=0, plus mesh data
    (vertices + triangle faces) per geom for proper Mesh3d rendering."""
    m = mujoco.MjModel.from_xml_path(V16_XML)
    d = mujoco.MjData(m); d.qpos[:] = 0; mujoco.mj_forward(m, d)
    ulna_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ulna")
    humerus_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus")
    wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wp = d.xpos[wrist_id].copy()
    wR = d.xmat[wrist_id].reshape(3, 3).copy()

    bone_meshes = {"ulna_right_mesh", "radius_right_mesh", "humerus_right_mesh"}
    hand_meshes_world = []
    bone_meshes_world = []
    MAX_FACES_PER_MESH_HAND = 200   # decimate aggressively — 102 meshes × 200 = 20K faces
    MAX_FACES_PER_MESH_BONE = 4000  # forearm bones, want them more detailed
    for gi in range(m.ngeom):
        bid = m.geom_bodyid[gi]
        if bid not in (ulna_id, humerus_id): continue
        mesh_id = m.geom_dataid[gi]
        if mesh_id < 0: continue
        mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if not mesh_name: continue
        vstart = m.mesh_vertadr[mesh_id]; vcount = m.mesh_vertnum[mesh_id]
        verts = m.mesh_vert[vstart:vstart + vcount].copy()
        fstart = m.mesh_faceadr[mesh_id]; fcount = m.mesh_facenum[mesh_id]
        faces = m.mesh_face[fstart:fstart + fcount].copy()

        # Decimate by stride
        is_hand = "Retopo" in mesh_name
        max_f = MAX_FACES_PER_MESH_HAND if is_hand else MAX_FACES_PER_MESH_BONE
        if len(faces) > max_f:
            stride = max(1, len(faces) // max_f)
            faces = faces[::stride]
            # Re-index: keep only used vertices
            used = np.unique(faces.flatten())
            old_to_new = -np.ones(len(verts), dtype=np.int64)
            old_to_new[used] = np.arange(len(used))
            verts = verts[used]
            faces = old_to_new[faces]

        gp = m.geom_pos[gi]; gq = m.geom_quat[gi]
        gR = np.zeros(9); mujoco.mju_quat2Mat(gR, gq); gR = gR.reshape(3, 3)
        mp = m.mesh_pos[mesh_id]; mq = m.mesh_quat[mesh_id]
        mR = np.zeros(9); mujoco.mju_quat2Mat(mR, mq); mR = mR.reshape(3, 3)
        bR = d.xmat[bid].reshape(3, 3); bp = d.xpos[bid]
        v_world = (bR @ ((gR @ (mR @ verts.T + mp[:, None])).T + gp).T).T + bp

        if is_hand:
            hand_meshes_world.append((v_world, faces))
        elif mesh_name in bone_meshes:
            bone_meshes_world.append((v_world, faces))

    n_hv = sum(len(v) for v,_ in hand_meshes_world)
    n_hf = sum(len(f) for _,f in hand_meshes_world)
    n_bv = sum(len(v) for v,_ in bone_meshes_world)
    n_bf = sum(len(f) for _,f in bone_meshes_world)
    print(f"  After decimation: hand {n_hv} verts / {n_hf} faces, bones {n_bv} verts / {n_bf} faces")
    return wp, wR, hand_meshes_world, bone_meshes_world


def main():
    print("Loading data...")
    v7, clouds, R_v7 = load_data()
    print(f"  Loaded {len(clouds)} landmark clouds")

    print("Loading V16 model...")
    wp, wR, hand_meshes_world, bone_meshes_world = get_v16_state()
    print(f"  Wrist body: {wp*1000} mm")
    print(f"  Hand mesh count: {len(hand_meshes_world)}")
    print(f"  Bone mesh count: {len(bone_meshes_world)}")

    # For default landmarks (d2_mid, d5_mid), compute rest-frame cloud for sh shift
    d2m, valid_d2m = clouds["right_d2_mid"]
    d5m, valid_d5m = clouds["right_d5_mid"]
    valid_both = valid_d2m & valid_d5m
    # Wrist velocity
    vel = np.zeros(len(v7))
    vel[1:] = np.linalg.norm(np.diff(v7[:, 2, :], axis=0), axis=1)
    valid_idx = np.flatnonzero(valid_both)
    rest_idx = valid_idx[np.argsort(vel[valid_idx])[:200]]
    cloud_W_mean = v7[rest_idx, 2, :].mean(0)
    sh_shift = cloud_W_mean - wp
    print(f"  sh_shift (mm): {sh_shift*1000}")

    # Apply shift to all model points (so model wrist body lands at cloud_W_mean)
    wrist_pos_shifted = wp + sh_shift  # = cloud_W_mean

    # Combine all hand-mesh triangles into ONE Mesh3d trace (faster rendering),
    # offsetting face indices for each subsequent mesh.
    def combine(meshes_world):
        if not meshes_world:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)
        all_v = []; all_f = []; offset = 0
        for v, f in meshes_world:
            all_v.append(v + sh_shift)
            all_f.append(f + offset)
            offset += v.shape[0]
        return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)
    hand_v, hand_f = combine(hand_meshes_world)
    bone_v, bone_f = combine(bone_meshes_world)

    # Build the Dash app
    app = Dash(__name__)

    # Default initial slider values: data-derived offsets for d2_mid, d5_mid
    d_C0_world = clouds["right_d2_mid"][0][rest_idx].mean(0) - cloud_W_mean
    d_L0_world = clouds["right_d5_mid"][0][rest_idx].mean(0) - cloud_W_mean
    d_C0_local = wR.T @ d_C0_world * 1000  # mm
    d_L0_local = wR.T @ d_L0_world * 1000

    # Pre-build cloud traces for each landmark choice
    # Convert clouds to mm for plotting
    def make_cloud_trace(name, color):
        positions, valid = clouds[name]
        # Use rest indices defined globally for d2+d5 — works decent for other pairs
        valid_idx_local = np.flatnonzero(valid)
        if len(valid_idx_local) < 200:
            return None
        rest = valid_idx_local[np.argsort(vel[valid_idx_local])[:200]]
        pts = positions[rest] * 1000  # mm
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.4),
            name=f"{name} cloud",
            visible=(name in ("right_d2_mid", "right_d5_mid")),
        )

    # Just the two clouds we actually want to align to. Toggle others via legend if needed.
    cloud_traces = []
    cloud_colors = {
        "right_d2_mid": "rgba(0,220,220,0.5)",
        "right_d5_mid": "rgba(220,0,220,0.5)",
    }
    for name, color in cloud_colors.items():
        if name not in clouds: continue
        trace = make_cloud_trace(name, color)
        if trace is not None:
            cloud_traces.append(trace)

    # Hand mesh as triangulated Mesh3d (looks like the mujoco render, not point cloud)
    hand_trace = go.Mesh3d(
        x=hand_v[:, 0]*1000, y=hand_v[:, 1]*1000, z=hand_v[:, 2]*1000,
        i=hand_f[:, 0], j=hand_f[:, 1], k=hand_f[:, 2],
        color="lightgray", opacity=0.55, name="Hand bones (Retopo meshes)",
        showlegend=True, lighting=dict(ambient=0.5, diffuse=0.7),
        flatshading=True,
    )
    # Forearm bones — slightly darker
    bone_trace = go.Mesh3d(
        x=bone_v[:, 0]*1000, y=bone_v[:, 1]*1000, z=bone_v[:, 2]*1000,
        i=bone_f[:, 0], j=bone_f[:, 1], k=bone_f[:, 2],
        color="dimgray", opacity=0.55, name="Forearm (ulna/radius/humerus)",
        showlegend=True, lighting=dict(ambient=0.5, diffuse=0.7),
        flatshading=True,
    )

    # Wrist site (yellow)
    wrist_trace = go.Scatter3d(
        x=[wrist_pos_shifted[0]*1000], y=[wrist_pos_shifted[1]*1000], z=[wrist_pos_shifted[2]*1000],
        mode="markers", marker=dict(size=8, color="gold"),
        name="Wrist site (model body)",
    )
    # Wrist axis triad: 3mm in +X, +Y, +Z (wrist-local) → world
    axis_traces = []
    for i, (name, col) in enumerate([("X", "red"), ("Y", "green"), ("Z", "blue")]):
        axis = np.zeros(3); axis[i] = 0.003
        endpoint_world = wrist_pos_shifted + wR @ axis
        axis_traces.append(go.Scatter3d(
            x=[wrist_pos_shifted[0]*1000, endpoint_world[0]*1000],
            y=[wrist_pos_shifted[1]*1000, endpoint_world[1]*1000],
            z=[wrist_pos_shifted[2]*1000, endpoint_world[2]*1000],
            mode="lines+markers", line=dict(color=col, width=4),
            marker=dict(size=4, color=col),
            name=f"+{name} (wrist-local)",
        ))

    # Knuckle balls — initial positions
    knC_world = wrist_pos_shifted + wR @ (d_C0_local / 1000)
    knL_world = wrist_pos_shifted + wR @ (d_L0_local / 1000)
    knC_trace = go.Scatter3d(
        x=[knC_world[0]*1000], y=[knC_world[1]*1000], z=[knC_world[2]*1000],
        mode="markers", marker=dict(size=12, color="cyan"),
        name="KnuckleC (cyan)",
    )
    knL_trace = go.Scatter3d(
        x=[knL_world[0]*1000], y=[knL_world[1]*1000], z=[knL_world[2]*1000],
        mode="markers", marker=dict(size=12, color="magenta"),
        name="KnuckleL (magenta)",
    )

    fig = go.Figure(data=[hand_trace, bone_trace, wrist_trace, *axis_traces, *cloud_traces, knC_trace, knL_trace])
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        height=800, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    SLIDER_STYLE = {"width": "100%"}
    app.layout = html.Div([
        html.H2("v17 KnuckleC / KnuckleL site placement"),
        html.P("Drag sliders to move the cyan / magenta balls in V16 wrist-local frame. "
               "Match them to the hand mesh / mocap cloud."),
        dcc.Graph(id="scene", figure=fig, style={"height": "800px"}),
        html.Hr(),
        html.Div([
            html.Div([
                html.H3("KnuckleC (cyan)"),
                html.Label(id="knC-x-label"),
                dcc.Slider(id="knC-x", min=-5, max=5, step=0.1, value=float(d_C0_local[0]),
                            marks={i: f"{i}" for i in range(-5, 6)}),
                html.Label(id="knC-y-label"),
                dcc.Slider(id="knC-y", min=-2, max=12, step=0.1, value=float(d_C0_local[1]),
                            marks={i: f"{i}" for i in range(-2, 13, 2)}),
                html.Label(id="knC-z-label"),
                dcc.Slider(id="knC-z", min=-5, max=8, step=0.1, value=float(d_C0_local[2]),
                            marks={i: f"{i}" for i in range(-5, 9, 2)}),
            ], style={"width": "48%", "display": "inline-block", "padding": "0 10px"}),
            html.Div([
                html.H3("KnuckleL (magenta)"),
                html.Label(id="knL-x-label"),
                dcc.Slider(id="knL-x", min=-5, max=5, step=0.1, value=float(d_L0_local[0]),
                            marks={i: f"{i}" for i in range(-5, 6)}),
                html.Label(id="knL-y-label"),
                dcc.Slider(id="knL-y", min=-2, max=12, step=0.1, value=float(d_L0_local[1]),
                            marks={i: f"{i}" for i in range(-2, 13, 2)}),
                html.Label(id="knL-z-label"),
                dcc.Slider(id="knL-z", min=-5, max=8, step=0.1, value=float(d_L0_local[2]),
                            marks={i: f"{i}" for i in range(-5, 9, 2)}),
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
        html.Div(id="output-summary", style={"fontFamily": "monospace", "padding": "10px",
                                              "backgroundColor": "#f4f4f4", "whiteSpace": "pre-wrap"}),
        # Hidden state with constants used in the callback
        dcc.Store(id="state", data={
            "wrist_pos": wrist_pos_shifted.tolist(),
            "wrist_R": wR.tolist(),
            "n_traces": len(fig.data),
        }),
    ])

    @app.callback(
        [Output("scene", "figure"),
         Output("output-summary", "children"),
         Output("knC-x-label", "children"), Output("knC-y-label", "children"), Output("knC-z-label", "children"),
         Output("knL-x-label", "children"), Output("knL-y-label", "children"), Output("knL-z-label", "children")],
        [Input("knC-x", "value"), Input("knC-y", "value"), Input("knC-z", "value"),
         Input("knL-x", "value"), Input("knL-y", "value"), Input("knL-z", "value")],
        [State("state", "data")],
    )
    def update_balls(cx, cy, cz, lx, ly, lz, state):
        wp = np.array(state["wrist_pos"])
        wR = np.array(state["wrist_R"])
        n_traces = state["n_traces"]
        local_C = np.array([cx, cy, cz]) / 1000
        local_L = np.array([lx, ly, lz]) / 1000
        world_C = (wp + wR @ local_C) * 1000
        world_L = (wp + wR @ local_L) * 1000

        # Use Patch for live trace updates (Dash detects this as a change)
        patched = Patch()
        # Last 2 traces are KnuckleC (-2) and KnuckleL (-1)
        patched["data"][n_traces - 2]["x"] = [float(world_C[0])]
        patched["data"][n_traces - 2]["y"] = [float(world_C[1])]
        patched["data"][n_traces - 2]["z"] = [float(world_C[2])]
        patched["data"][n_traces - 1]["x"] = [float(world_L[0])]
        patched["data"][n_traces - 1]["y"] = [float(world_L[1])]
        patched["data"][n_traces - 1]["z"] = [float(world_L[2])]

        summary = (
            f"KnuckleC (wrist-local mm): X={cx:+.2f}  Y={cy:+.2f}  Z={cz:+.2f}    "
            f"world: ({world_C[0]:.2f}, {world_C[1]:.2f}, {world_C[2]:.2f})\n"
            f"KnuckleL (wrist-local mm): X={lx:+.2f}  Y={ly:+.2f}  Z={lz:+.2f}    "
            f"world: ({world_L[0]:.2f}, {world_L[1]:.2f}, {world_L[2]:.2f})\n\n"
            f"For v17 model XML — site pos= attributes (in METERS, wrist-local):\n"
            f'  <site name="KnuckleC" pos="{cx/1000:.6f} {cy/1000:.6f} {cz/1000:.6f}"/>\n'
            f'  <site name="KnuckleL" pos="{lx/1000:.6f} {ly/1000:.6f} {lz/1000:.6f}"/>\n\n'
            f"Lateral spread (X axis, mm): {abs(cx-lx):.2f}\n"
            f"Distal spread (Y axis, mm): KnC={cy:+.2f}, KnL={ly:+.2f}"
        )
        return (
            patched, summary,
            f"X = {cx:+.2f} mm", f"Y = {cy:+.2f} mm (forearm long axis +Y = supination axis)", f"Z = {cz:+.2f} mm",
            f"X = {lx:+.2f} mm", f"Y = {ly:+.2f} mm", f"Z = {lz:+.2f} mm",
        )

    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        [State("knC-x", "value"), State("knC-y", "value"), State("knC-z", "value"),
         State("knL-x", "value"), State("knL-y", "value"), State("knL-z", "value")],
        prevent_initial_call=True,
    )
    def save_offsets(n_clicks, cx, cy, cz, lx, ly, lz):
        if not n_clicks:
            return ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"/mnt/vast/niyo/stac-mjx/v17_knuckle_offsets_{ts}.json")
        latest_path = Path("/mnt/vast/niyo/stac-mjx/v17_knuckle_offsets_LATEST.json")
        data = {
            "saved_at": ts,
            "session_used": SID,
            "frame_of_reference": "V16 wrist-local (mouse_forelimb_right_janelia_moving_shoulder_v2.xml)",
            "KnuckleC": {
                "source_landmark": "right_d2_mid",
                "local_mm": [cx, cy, cz],
                "local_m": [cx/1000, cy/1000, cz/1000],
                "site_pos_xml": f'pos="{cx/1000:.6f} {cy/1000:.6f} {cz/1000:.6f}"',
            },
            "KnuckleL": {
                "source_landmark": "right_d5_mid",
                "local_mm": [lx, ly, lz],
                "local_m": [lx/1000, ly/1000, lz/1000],
                "site_pos_xml": f'pos="{lx/1000:.6f} {ly/1000:.6f} {lz/1000:.6f}"',
            },
            "lateral_spread_mm": abs(cx - lx),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        with open(latest_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"✓ Saved (#{n_clicks}) to {path.name} and v17_knuckle_offsets_LATEST.json"

    print("\n" + "=" * 70)
    print("Starting Dash app on http://0.0.0.0:8050")
    print("Open in browser. Drag sliders to position knuckle balls.")
    print("=" * 70)
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
