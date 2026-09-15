"""Render the 5 spliced mocap keypoints (Shoulder/Elbow/Wrist + d2_mid + d5_mid)
reprojected onto each camera's video frames for the v17 smoke-test reach window.

Output:
  v17_spliced_mocap_A36-1_trial4072.mp4  (3 cams side-by-side)

Each panel shows:
  - Raw 2D SLEAP detection: solid colored dot (anipose-projected if mocap-reproj differs)
  - Reprojected (3D-back-to-2D): white-bordered ring at the same color
  - Skeleton edges: shoulder->elbow->wrist->[d2_mid, d5_mid]
"""
from __future__ import annotations
from pathlib import Path
import cv2
import h5py
import numpy as np

ANIMAL = "A36-1"
SESSION = "A36-1_2023-07-18_16-54-01_lightOff_video"
ROOT = Path("/mnt/vast/eric/azim-emg-data")
SESSION_DIR = ROOT / SESSION
CAMS = ("CamA", "CamB", "CamC")

CLIP_START = 25781   # match the v17 STAC clip for A36-1 (best d2+d5 validity)
N_FRAMES = 100
PLAY_FPS = 25       # slow-mo (source is 200fps)

# Color per landmark (BGR for cv2 — but we use RGB throughout, conversion at write)
COLORS = {
    "Shoulder": (255, 80, 80),     # red
    "Elbow":    (255, 180, 80),    # orange
    "Wrist":    (255, 255, 80),    # yellow
    "KnuckleC": (80, 255, 255),    # cyan (= d2_mid)
    "KnuckleL": (255, 80, 255),    # magenta (= d5_mid)
}
EDGES = [
    ("Shoulder", "Elbow"),
    ("Elbow", "Wrist"),
    ("Wrist", "KnuckleC"),
    ("Wrist", "KnuckleL"),
]


def brighten(img, gamma=0.6, lo_pct=2, hi_pct=99):
    g = img.astype(np.float32) / 255.0
    lo = np.percentile(g, lo_pct); hi = np.percentile(g, hi_pct)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.power(g, gamma) * 255).astype(np.uint8)


def banner(img, text, h=24, color=(220, 240, 220)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def main():
    # Load 3kp (S, E, W) from points3d.h5
    p3d_h5 = SESSION_DIR / f"{SESSION}_points3d.h5"
    with h5py.File(p3d_h5) as f:
        p3_2d = f["points_2d"][:, CLIP_START:CLIP_START + N_FRAMES]   # (3 cams, T, 3 nodes, 2)
        p3_re = f["reprojections"][:, CLIP_START:CLIP_START + N_FRAMES]
        n3 = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    # Load digits → d2_mid, d5_mid
    dig_h5 = SESSION_DIR / f"{SESSION}_digits_points3d.h5"
    with h5py.File(dig_h5) as f:
        d_2d = f["points_2d"][:, CLIP_START:CLIP_START + N_FRAMES]  # (3, T, 17, 2)
        d_re = f["reprojections"][:, CLIP_START:CLIP_START + N_FRAMES]
        nd = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["node_names"]]
    print(f"3kp h5 nodes: {n3}, shape {p3_2d.shape}")
    print(f"digits h5 nodes: {len(nd)}, shape {d_2d.shape}")

    # Build 5kp index lookup
    iSh3 = n3.index("right_shoulder"); iEl3 = n3.index("right_elbow"); iWr3 = n3.index("right_wrist")
    iC = nd.index("right_d2_mid"); iL = nd.index("right_d5_mid")

    # Per-cam, per-frame: stack 5kp 2D (raw + reproj)
    # Result: (3 cams, N_FRAMES, 5 kps, 2)
    raw5 = np.full((3, N_FRAMES, 5, 2), np.nan)
    re5 = np.full((3, N_FRAMES, 5, 2), np.nan)
    for c in range(3):
        raw5[c, :, 0] = p3_2d[c, :, iSh3]
        raw5[c, :, 1] = p3_2d[c, :, iEl3]
        raw5[c, :, 2] = p3_2d[c, :, iWr3]
        raw5[c, :, 3] = d_2d[c, :, iC]
        raw5[c, :, 4] = d_2d[c, :, iL]
        re5[c, :, 0] = p3_re[c, :, iSh3]
        re5[c, :, 1] = p3_re[c, :, iEl3]
        re5[c, :, 2] = p3_re[c, :, iWr3]
        re5[c, :, 3] = d_re[c, :, iC]
        re5[c, :, 4] = d_re[c, :, iL]

    # Open videos
    caps = {}
    for cam in CAMS:
        path = SESSION_DIR / cam / f"{SESSION}_{cam}_resample.mp4"
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, CLIP_START)
        caps[cam] = cap

    # Output writer
    H = 512; W = 512
    title_h = 28
    legend_h = 36
    final_w = W * 3
    final_h = title_h + H + legend_h
    out_path = Path(f"/mnt/vast/niyo/stac-mjx/refined_STACed_data_v17/A36-1/v17_spliced_mocap_{ANIMAL}_trial{CLIP_START}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, PLAY_FPS, (final_w, final_h))

    # Pre-build legend strip
    legend = np.zeros((legend_h, final_w, 3), dtype=np.uint8)
    x = 10
    for kp_name, col in COLORS.items():
        cv2.circle(legend, (x + 8, legend_h // 2), 6, col, -1)
        cv2.circle(legend, (x + 35, legend_h // 2), 6, col, 2)  # ring (reproj)
        cv2.putText(legend, kp_name, (x + 50, legend_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        x += 200
    cv2.putText(legend, "dot=raw 2D, ring=3D->2D reproj", (x, legend_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    kp_names = ["Shoulder", "Elbow", "Wrist", "KnuckleC", "KnuckleL"]
    for t in range(N_FRAMES):
        panels = []
        for c, cam in enumerate(CAMS):
            ok, frame = caps[cam].read()
            if not ok:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[:2] != (H, W):
                    frame = cv2.resize(frame, (W, H))
            frame = brighten(frame)
            # Draw skeleton edges (raw)
            for a, b in EDGES:
                ai = kp_names.index(a); bi = kp_names.index(b)
                pa = raw5[c, t, ai]; pb = raw5[c, t, bi]
                if not np.isnan(pa).any() and not np.isnan(pb).any():
                    cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                             COLORS[a], 1, cv2.LINE_AA)
            # Draw raw (filled dots) and reproj (rings)
            for ki, kname in enumerate(kp_names):
                col = COLORS[kname]
                pr = raw5[c, t, ki]; pe = re5[c, t, ki]
                if not np.isnan(pr).any():
                    cv2.circle(frame, (int(pr[0]), int(pr[1])), 5, col, -1, cv2.LINE_AA)
                if not np.isnan(pe).any():
                    cv2.circle(frame, (int(pe[0]), int(pe[1])), 7, col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(pe[0]), int(pe[1])), 7, (255, 255, 255), 1, cv2.LINE_AA)
            err = np.linalg.norm(re5[c, t] - raw5[c, t], axis=-1)
            mean_err = float(np.nanmean(err)) if (~np.isnan(err)).any() else float("nan")
            frame = banner(frame, f"{cam}  f={CLIP_START + t}  reproj_err={mean_err:.1f}px", h=24)
            panels.append(frame)
        row = np.concatenate(panels, axis=1)
        title = f"{ANIMAL} v17 spliced mocap (S/E/W + d2_mid + d5_mid)  frame {t + 1}/{N_FRAMES}  abs={CLIP_START + t}"
        title_band = banner(np.zeros((title_h, final_w, 3), dtype=np.uint8), title, h=title_h)
        full = np.concatenate([title_band, row, legend], axis=0)
        writer.write(cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

    writer.release()
    for cap in caps.values():
        cap.release()
    print(f"\nDone -> {out_path}")
    print(f"Size: {out_path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
