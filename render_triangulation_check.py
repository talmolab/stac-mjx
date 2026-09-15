#!/usr/bin/env python
"""Per-trial "triangulation check" videos (no MuJoCo/STAC model involved --
just raw 2D detections vs. their calibration-projected 3D triangulation),
matching the style of the reference video Eric already had:
CFL_36_20231220_trial_0401_triangulation_check.mp4 (green=original SLEAP
2D prediction, red=reprojection of the triangulated 3D point back into that
camera). Regenerated here via the same method for ALL 15 CFL_35/36/37 trials,
in two variants per trial so the calibration refinement's effect is directly
visible:

  <trial>_triangulation_check.mp4          -- original calibration.toml +
                                               points3d.h5
  <trial>_triangulation_check_refined.mp4  -- calibration_refined.toml +
                                               points3d_refined.h5

Both variants overlay the SAME green 2D detections (they don't change --
only the 3D triangulation used to compute the red reprojection changes), so
the red dots visibly snapping onto the green ones in the refined video is the
direct visual signature of the calibration fix.

Both calib_path/points3d_path pairs are used self-consistently: original
calibration.toml + points3d.h5 for the "original" variant,
calibration_refined.toml + its own points3d_refined.h5 for the "refined"
variant. (An earlier version of this session tried reprojecting a
gauge-corrected version of the refined points through the *original*
calibration.toml instead, reasoning that calibration_refined.toml's own
extrinsics live in an arbitrary drifted gauge -- bundle_adjust_iter has no
absolute anchor, so the whole reconstruction's rotation/translation/scale is
unconstrained even though reprojection error, which is gauge-invariant,
genuinely improves. That reasoning is still correct for why gauge drift
happens, but "fixing" it this way made the video look *worse*, not better --
reprojecting through the original calibration necessarily reintroduces
whatever accuracy problems the original calibration had (e.g. Cam01's bad
extrinsics), which is exactly what the refinement is trying to fix in the
first place. Reverted per direct visual feedback: this video's job is to show
2D reprojection accuracy, which is what the refined calibration's own
self-consistent triangulation demonstrates correctly.)
"""
from pathlib import Path

import cv2
import numpy as np
import h5py
import sleap_io as sio
from aniposelib.cameras import CameraGroup

TRAINING_ROOT = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData")
CAM_DIRS = ["Cam00", "Cam01", "Cam02"]

KP_NAMES = ["shoulder", "elbow", "Wrist", "d1_k", "d1_t", "d2_k", "d2_m", "d2_t",
            "d3_k", "d3_m", "d3_t", "d4_k", "d4_m", "d4_t", "d5_k", "d5_m", "d5_t",
            "js_base", "js_ball"]
IDX = {n: i for i, n in enumerate(KP_NAMES)}

# Same topology as register_v21_mocap.py's TREE_EDGES, plus the js_base-js_ball
# rung (not part of the mouse skeleton rescale tree, but a real rigid segment
# worth drawing here since this is a pure-visualization check, not a rescale).
SKELETON_EDGES = [
    ("shoulder", "elbow"), ("elbow", "Wrist"),
    ("Wrist", "d1_k"), ("Wrist", "d2_k"), ("Wrist", "d3_k"), ("Wrist", "d4_k"), ("Wrist", "d5_k"),
    ("d1_k", "d1_t"),
    ("d2_k", "d2_m"), ("d2_m", "d2_t"),
    ("d3_k", "d3_m"), ("d3_m", "d3_t"),
    ("d4_k", "d4_m"), ("d4_m", "d4_t"),
    ("d5_k", "d5_m"), ("d5_m", "d5_t"),
    ("js_base", "js_ball"),
]

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def load_2d_points(trial_dir: Path, trial_name: str, cam_names):
    arrs = []
    for cam in cam_names:
        slp_path = trial_dir / cam / f"{trial_name}_{cam}_merged.slp"
        labels = sio.load_slp(str(slp_path))
        labels_sorted = sorted(labels, key=lambda lf: lf.frame_idx)
        arr = np.stack([lf.instances[0].numpy() for lf in labels_sorted])  # (T,19,2)
        arrs.append(arr)
    return np.stack(arrs)  # (n_cams,T,19,2)


def draw_overlay(frame, p2d_green, p2d_red, cam_label, frame_idx, legend):
    frame = frame.copy()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def draw_skeleton(pts, color):
        for a, b in SKELETON_EDGES:
            pa, pb = pts[IDX[a]], pts[IDX[b]]
            if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                continue
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1, cv2.LINE_AA)
        for pt in pts:
            if np.any(np.isnan(pt)):
                continue
            if color is GREEN:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, 1, cv2.LINE_AA)

    draw_skeleton(p2d_green, GREEN)
    draw_skeleton(p2d_red, RED)

    cv2.putText(frame, cam_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
    if legend:
        h = frame.shape[0]
        cv2.putText(frame, f"frame {frame_idx}  (green=original prediction, red=reprojection)",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame


def render_trial_variant(trial_dir: Path, trial_name: str, calib_path: Path,
                          points3d_path: Path, output_path: Path):
    cgroup = CameraGroup.load(str(calib_path))
    cam_names = [c.get_name() for c in cgroup.cameras]

    p2ds = load_2d_points(trial_dir, trial_name, cam_names)  # (3,T,19,2)
    n_cams, n_frames, n_kp, _ = p2ds.shape

    with h5py.File(points3d_path, "r") as f:
        p3ds = f["tracks"][:][:, 0, :, :]  # (T,19,3)

    # reproject every camera's view of the 3D points
    reproj = np.full((n_cams, n_frames, n_kp, 2), np.nan)
    for ci, cam in enumerate(cgroup.cameras):
        pts = p3ds.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        proj = np.full((pts.shape[0], 2), np.nan)
        if valid.any():
            proj[valid] = cam.project(pts[valid]).reshape(-1, 2)
        reproj[ci] = proj.reshape(n_frames, n_kp, 2)

    caps = [cv2.VideoCapture(str(trial_dir / cam / f"{trial_name}_{cam}.mp4")) for cam in cam_names]
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w * n_cams, h))

    for fi in range(n_frames):
        tiles = []
        for ci, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            tile = draw_overlay(frame, p2ds[ci, fi], reproj[ci, fi], cam_names[ci], fi, legend=(ci == 0))
            tiles.append(tile)
        combined = np.concatenate(tiles, axis=1)
        writer.write(combined)

    writer.release()
    for cap in caps:
        cap.release()
    return n_frames


def main():
    trials = [d.name for d in sorted(TRAINING_ROOT.iterdir())
              if d.is_dir() and d.name.startswith(("CFL_35", "CFL_36", "CFL_37"))]

    for trial_name in trials:
        trial_dir = TRAINING_ROOT / trial_name
        print(f"\n{'='*70}\n{trial_name}\n{'='*70}")

        try:
            n = render_trial_variant(
                trial_dir, trial_name,
                calib_path=trial_dir / "calibration.toml",
                points3d_path=trial_dir / f"{trial_name}_points3d.h5",
                output_path=trial_dir / f"{trial_name}_triangulation_check.mp4",
            )
            print(f"  original:  {n} frames -> {trial_name}_triangulation_check.mp4")
        except Exception as e:
            import traceback
            print(f"  original FAILED: {e}")
            traceback.print_exc()

        try:
            n = render_trial_variant(
                trial_dir, trial_name,
                calib_path=trial_dir / "calibration_refined.toml",
                points3d_path=trial_dir / f"{trial_name}_points3d_refined.h5",  # its own, self-consistent triangulation
                output_path=trial_dir / f"{trial_name}_triangulation_check_refined.mp4",
            )
            print(f"  refined:   {n} frames -> {trial_name}_triangulation_check_refined.mp4")
        except Exception as e:
            import traceback
            print(f"  refined FAILED: {e}")
            traceback.print_exc()

    print("\nAll triangulation check videos done.")


if __name__ == "__main__":
    main()
