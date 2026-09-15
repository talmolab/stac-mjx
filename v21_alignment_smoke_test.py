#!/usr/bin/env python
"""Phase D scaling/alignment smoke test (CHECKPOINT).

For 1 representative trial per CFL_35/36/37 session: draw the REGISTERED
mocap keypoints (converted back to calibration frame via
RegistrationTransform.mujoco_to_calibration, reusing multicam.py directly)
onto a resting real video frame from each of the 3 cameras, using the
existing _draw_mocap_keypoints helper from stac_mjx/reproject_stac_model.py.
No MuJoCo model rendering yet (no STAC fit has run) -- this only checks that
the registered 3D points land in the right place on real footage: wrist near
the wrist, knuckles near knuckles, joystick ball near the visible joystick.

Also renders a coordinate-frame-independent 3D comparison: the model's
rest-pose skeleton vs. the registered mocap's median-resting pose.

Outputs (per trial): v21_registration_check/<trial>_alignment_check.png,
                      v21_registration_check/<trial>_skeleton_3d.png

STOP: do not proceed to the STAC smoke test until these are visually approved.
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")

from pathlib import Path
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from stac_mjx.multicam import load_calibration, load_registration_transform
from stac_mjx.reproject_stac_model import _draw_mocap_keypoints
from register_v21_mocap import (
    KP_NAMES, IDX, TREE_EDGES, ARM_HAND_KP, model_rest_targets, model_joystick_targets,
    TRAINING_ROOT, MODEL_PATH,
)

OUT_DIR = Path("/mnt/vast/niyo/stac-mjx/v21_registration_check")
OUT_DIR.mkdir(exist_ok=True)

PICKS = [
    "CFL_35_20240128_trial_0001",
    "CFL_36_20231220_trial_0001",
    "CFL_37_20240423_trial_0001",
]

# plt.cm.tab20(linspace(0,1,N)) does NOT map cleanly to tab20's 20 discrete
# swatches for N != 20 (it resamples/interpolates the colormap) -- use the
# actual discrete .colors list indexed by integer position instead, so each
# keypoint gets an unambiguous, correctly-labeled color.
_tab20 = plt.cm.tab20.colors  # 20 discrete (r,g,b) tuples, 0-1 range
KP_COLORS_BGR = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in _tab20[:len(KP_NAMES)]]


def alignment_check_image(trial_name: str):
    trial_dir = TRAINING_ROOT / trial_name
    calib_path = trial_dir / "calibration.toml"
    reg_path = trial_dir / f"{trial_name}_registered_v21.h5"
    if not reg_path.exists():
        raise FileNotFoundError(f"run register_v21_mocap.py first: missing {reg_path}")

    cameras = load_calibration(str(calib_path))
    reg = load_registration_transform(str(reg_path))

    with h5py.File(reg_path, "r") as f:
        tracks_mj = f["tracks"][:, 0, :, :]  # (T,19,3) meters, v21 world frame
        rest_idx = f.attrs["resting_frame_indices"]

    frame_idx = int(rest_idx[0])  # first resting frame
    kp_mj = tracks_mj[frame_idx]  # (19,3)
    shoulder_calib = reg.shoulder_reference_mm  # fixed shoulder ref (mm, calib frame)
    kp_calib = reg.mujoco_to_calibration(kp_mj, shoulder_calib)  # (19,3) mm

    panels = []
    for cam in cameras:
        cam_dir = None
        for variant in [cam.name, cam.name.lower(), cam.name.upper()]:
            cand = trial_dir / variant
            if cand.is_dir():
                cam_dir = cand
                break
        if cam_dir is None:
            raise FileNotFoundError(f"no camera dir for {cam.name} in {trial_dir}")
        mp4 = list(cam_dir.glob("*.mp4"))[0]
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_idx} from {mp4}")
        frame = cam.undistort_frame(frame)
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = _draw_mocap_keypoints(
            frame, kp_calib, cam, KP_COLORS_BGR, radius=4, marker_style="circle",
            label=f"{cam.name} frame{frame_idx}",
        )

        # js_base/js_ball explicitly, unambiguously: big filled markers + direct
        # text labels (not just relying on the shared tab20 legend color).
        js_base_2d = cam.project_points(kp_calib[IDX["js_base"]:IDX["js_base"] + 1])[0]
        js_ball_2d = cam.project_points(kp_calib[IDX["js_ball"]:IDX["js_ball"] + 1])[0]
        bx, by = int(js_base_2d[0]), int(js_base_2d[1])
        gx, gy = int(js_ball_2d[0]), int(js_ball_2d[1])
        cv2.drawMarker(frame, (bx, by), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 2)  # orange = js_base
        cv2.putText(frame, "js_base", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
        cv2.drawMarker(frame, (gx, gy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)  # magenta = js_ball
        cv2.putText(frame, "js_ball", (gx + 10, gy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        panels.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    grid = np.concatenate(panels, axis=1)
    out_path = OUT_DIR / f"{trial_name}_alignment_check.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  wrote {out_path}")
    return kp_mj, rest_idx


def skeleton_3d_plot(trial_name: str, targets: dict):
    reg_path = TRAINING_ROOT / trial_name / f"{trial_name}_registered_v21.h5"
    with h5py.File(reg_path, "r") as f:
        tracks_mj = f["tracks"][:, 0, :, :]
        rest_idx = f.attrs["resting_frame_indices"]
    median_resting = np.median(tracks_mj[rest_idx], axis=0)  # (19,3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def plot_skeleton(get_point, color, label, marker):
        for p, c in TREE_EDGES:
            pp, cp = get_point(p), get_point(c)
            ax.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]], color=color, linewidth=1.5)
        pts = np.array([get_point(k) for k in ARM_HAND_KP])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, label=label, marker=marker, s=25)

    plot_skeleton(lambda k: targets[k], "tab:blue", "model rest pose", "o")
    plot_skeleton(lambda k: median_resting[IDX[k]], "tab:red", "registered mocap (median resting)", "x")

    # Joystick: both sides, so misalignment is visible directly (not just via
    # video reprojection). Model side = actual FK on the v21 XML's
    # joystick_base/joystick bodies; mocap side = registered median-resting
    # js_base/js_ball (rotate+translate only, no rescale -- see register_trial).
    model_js = model_joystick_targets(MODEL_PATH)
    mocap_js_base = median_resting[IDX["js_base"]]
    mocap_js_ball = median_resting[IDX["js_ball"]]
    ax.scatter(*model_js["js_base"], color="tab:orange", label="model js_base", marker="s", s=80)
    ax.scatter(*model_js["js_ball"], color="tab:purple", label="model js_ball", marker="^", s=80)
    ax.scatter(*mocap_js_base, color="tab:orange", label="mocap js_base (registered)", marker="x", s=100)
    ax.scatter(*mocap_js_ball, color="tab:purple", label="mocap js_ball (registered)", marker="+", s=120)
    ax.plot([model_js["js_base"][0], mocap_js_base[0]],
            [model_js["js_base"][1], mocap_js_base[1]],
            [model_js["js_base"][2], mocap_js_base[2]], "k:", linewidth=1)
    ax.plot([model_js["js_ball"][0], mocap_js_ball[0]],
            [model_js["js_ball"][1], mocap_js_ball[1]],
            [model_js["js_ball"][2], mocap_js_ball[2]], "k:", linewidth=1)
    js_base_err = np.linalg.norm(model_js["js_base"] - mocap_js_base)
    js_ball_err = np.linalg.norm(model_js["js_ball"] - mocap_js_ball)
    print(f"  js_base model<->mocap distance: {js_base_err*1000:.3f} mm")
    print(f"  js_ball model<->mocap distance: {js_ball_err*1000:.3f} mm")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{trial_name}: model rest pose vs. registered mocap")
    ax.legend()
    out_path = OUT_DIR / f"{trial_name}_skeleton_3d.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    targets = model_rest_targets(MODEL_PATH)
    for trial_name in PICKS:
        print(f"[{trial_name}]")
        alignment_check_image(trial_name)
        skeleton_3d_plot(trial_name, targets)
    print("\nSTOP: inspect the images in v21_registration_check/ before proceeding "
          "to the STAC smoke test.")


if __name__ == "__main__":
    main()
