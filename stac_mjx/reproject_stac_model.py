"""Reproject STAC'd MuJoCo model onto multi-camera video frames.

Renders the MuJoCo arm model from each calibration camera viewpoint,
composites it onto the original video frames, and outputs a multi-view
grid video. Also overlays mocap keypoints and environmental markers.
"""

import os

# setdefault, not a hard override: lets a caller that already needs a
# different GL backend (e.g. osmesa, where egl isn't available) set
# MUJOCO_GL before importing this module without it being clobbered here.
os.environ.setdefault("MUJOCO_GL", "egl")

import csv
import numpy as np
import cv2
import h5py
import mujoco
import imageio
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from dm_control import mjcf
from scipy.spatial.transform import Rotation as RotLib

from stac_mjx import io
from stac_mjx.multicam import (
    load_calibration,
    load_registration_transform,
    load_raw_shoulder_positions,
    load_env_points3d,
    CameraParams,
    RegistrationTransform,
)


# Background color for chroma keying (pure green)
_BG_COLOR = np.array([0, 255, 0], dtype=np.uint8)
_BG_THRESH = 40  # tolerance for green screen keying


def _setup_mujoco_model(cfg, kp_names, cam_params_list=None):
    """Create MuJoCo model with body sites and calibration cameras.

    Args:
        cfg: STAC config.
        kp_names: Keypoint names.
        cam_params_list: List of dicts with 'name', 'pos', 'xyaxes', 'fovy'.

    Returns:
        (mj_model, mj_data) ready for FK and rendering.
    """
    xml_path = cfg.model.MJCF_PATH
    root = mjcf.from_path(xml_path)

    # Add body sites for keypoints (same as Stac.__init__)
    for key, v in cfg.model.KEYPOINT_MODEL_PAIRS.items():
        parent = root.worldbody.find("body", v)
        pos = cfg.model.KEYPOINT_INITIAL_OFFSETS[key]
        parent.add(
            "site", name=key, type="sphere", size=[0.001],
            rgba="0 0 0 0.8", pos=pos, group=3,
        )

    # Add calibration cameras
    if cam_params_list:
        for cp in cam_params_list:
            root.worldbody.add(
                "camera",
                name=f"calib_{cp['name']}",
                pos=cp["pos"].tolist(),
                xyaxes=cp["xyaxes"].tolist(),
                fovy=float(cp["fovy"]),
            )

    physics = mjcf.Physics.from_mjcf_model(root)
    mj_model = deepcopy(physics.model.ptr)
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def _set_green_background(mj_model, offwidth=512, offheight=512):
    """Configure the MuJoCo model for green-screen rendering.

    offwidth/offheight must be >= the actual render_size requested by the
    caller (mujoco.Renderer's framebuffer is capped by these), so they're
    parameterized rather than hardcoded to 512x512 -- a caller requesting a
    larger render_size (e.g. to match a non-512x512 source video's aspect
    ratio) would otherwise hit "Image width > framebuffer width".
    """
    # Set RGBA background to green for chroma keying
    # MuJoCo stores rgba_fog as the background color when fog is off
    # We modify the skybox texture and fog color
    mj_model.vis.rgba.fog[:] = [0, 1, 0, 1]

    # Disable skybox, grid floor, and headlight effects for clean keying
    mj_model.vis.global_.offwidth = offwidth
    mj_model.vis.global_.offheight = offheight


def _make_scene_option():
    """Create scene options showing only the arm mesh, no sites/joints."""
    opt = mujoco.MjvOption()
    # Show mesh geoms (group 0), hide others
    opt.geomgroup[0] = 1  # bone meshes
    opt.geomgroup[1] = 0  # ground plane
    opt.geomgroup[2] = 0  # sites visual
    # Hide all decorations
    opt.sitegroup[0] = 0
    opt.sitegroup[1] = 0
    opt.sitegroup[2] = 0
    opt.sitegroup[3] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    opt.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    opt.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
    return opt


def _compute_calibration_cam_xyaxes(
    cam_params: CameraParams,
    reg: RegistrationTransform,
    shoulder_calib: np.ndarray,
) -> dict:
    """Compute MuJoCo camera pos, xyaxes, fovy from calibration.

    Uses xyaxes (right + up vectors) instead of quaternion for
    unambiguous camera orientation.

    Args:
        cam_params: Calibration camera parameters.
        reg: Registration transform (calib → MuJoCo).
        shoulder_calib: (3,) shoulder position in calibration frame (mm).

    Returns:
        dict with 'pos' (3,), 'xyaxes' (6,), 'fovy' float.
    """
    R_cv = cam_params.R
    t_cv = cam_params.tvec

    # Effective world-to-camera rotation in MuJoCo frame
    # R_total maps MuJoCo→calibration (column-vector convention),
    # so R_cv @ R_total gives MuJoCo→camera
    R_eff = R_cv @ reg.R_total
    t_eff = R_cv @ shoulder_calib + t_cv

    # Camera position in MuJoCo frame. Must apply the SAME per-trial uniform
    # scale_factor used during registration (register_v21_mocap.py's
    # shared-rotation_v4 recipe) -- this term is R_total.T @ (cam_pos_calib -
    # shoulder_calib) / 1000, i.e. the same "centered, unit-converted" vector
    # that registration multiplies by scale_factor before rotating+
    # translating. Missing this placed the virtual camera at the wrong
    # distance from the model, differently per trial (each trial has its own
    # scale_factor) -- looked like the model changing size between trials
    # (a pure perspective/depth artifact, not an actual model-size change,
    # since the MJCF's own geometry never varies) and threw off the
    # joystick's apparent position along with everything else in the shot.
    cam_pos_mj = (-R_eff.T @ t_eff / 1000.0) * reg.scale_factor + reg.model_shoulder

    # Camera axes in MuJoCo world frame
    # R_eff.T maps camera → MuJoCo world
    cam_right = R_eff.T @ np.array([1, 0, 0.0])   # OpenCV +X = right
    cam_down = R_eff.T @ np.array([0, 1, 0.0])     # OpenCV +Y = down

    # MuJoCo xyaxes: [right_xyz, up_xyz], up = -down
    xyaxes = np.concatenate([cam_right, -cam_down])

    # FOV from intrinsics
    fy = cam_params.K[1, 1]
    height = cam_params.size[1]
    fovy = 2.0 * np.degrees(np.arctan(height / (2.0 * fy)))

    return {"pos": cam_pos_mj, "xyaxes": xyaxes, "fovy": fovy}


def _render_model_frame(
    mj_model, mj_data, renderer, qpos, camera, scene_option
) -> np.ndarray:
    """Render the MuJoCo model for one frame, return RGBA with alpha mask.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        renderer: MuJoCo renderer.
        qpos: Joint angles for this frame.
        camera: Camera index (int) or MjvCamera object.
        scene_option: Scene rendering options.

    Returns:
        (H, W, 4) RGBA image with alpha channel from green screen keying.
    """
    mj_data.qpos[:] = qpos
    mujoco.mj_fwdPosition(mj_model, mj_data)

    renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
    pixels = renderer.render()  # (H, W, 3) uint8

    # Create alpha mask: everything that's NOT green background
    diff = np.abs(pixels.astype(np.int16) - _BG_COLOR.astype(np.int16))
    mask = np.any(diff > _BG_THRESH, axis=2).astype(np.uint8) * 255

    # Slight dilation to clean edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    rgba = np.dstack([pixels, mask])
    return rgba


def _composite(bg: np.ndarray, fg_rgba: np.ndarray, alpha_scale: float = 0.7) -> np.ndarray:
    """Alpha-composite foreground RGBA onto background RGB.

    Args:
        bg: (H, W, 3) background image (uint8).
        fg_rgba: (H, W, 4) foreground with alpha channel (uint8).
        alpha_scale: overall opacity of the overlay (0-1).

    Returns:
        (H, W, 3) composited image (uint8).
    """
    alpha = (fg_rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha_scale
    fg = fg_rgba[:, :, :3].astype(np.float32)
    bg_f = bg.astype(np.float32)
    out = bg_f * (1 - alpha) + fg * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_env_markers(
    frame: np.ndarray,
    env_points: np.ndarray,
    env_names: list,
    cam: CameraParams,
    color=(255, 255, 255),
    thickness=1,
) -> np.ndarray:
    """Draw environmental markers (bars, water spout) on a frame.

    Args:
        frame: (H, W, 3) image to draw on (modified in place).
        env_points: (N, 3) 3D points in calibration frame (mm).
        env_names: list of point names.
        cam: Camera parameters for projection.
        color: BGR color for drawing.
        thickness: line thickness.

    Returns:
        frame with markers drawn.
    """
    pts_2d = cam.project_points(env_points).astype(np.int32)
    name_to_idx = {n: i for i, n in enumerate(env_names)}

    # Draw bar rectangles
    for prefix in ["BarL", "BarR"]:
        corners = ["TL", "TR", "BR", "BL"]
        idxs = [name_to_idx.get(f"{prefix}_{c}") for c in corners]
        if all(i is not None for i in idxs):
            poly = np.array([pts_2d[i] for i in idxs], dtype=np.int32)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)

    # Draw water spout
    tip_idx = name_to_idx.get("water_spout_tip")
    mm_idx = name_to_idx.get("water_spout_1mm")
    if tip_idx is not None and mm_idx is not None:
        cv2.line(frame, tuple(pts_2d[tip_idx]), tuple(pts_2d[mm_idx]), color=color, thickness=thickness)
        cv2.circle(frame, tuple(pts_2d[tip_idx]), 3, (0, 255, 255), -1)

    return frame


def _draw_mocap_keypoints(
    frame: np.ndarray,
    kp_3d: np.ndarray,
    cam: CameraParams,
    colors: list,
    radius: int = 4,
    marker_style: str = "circle",
    label: str = "",
) -> np.ndarray:
    """Draw keypoints projected onto a frame.

    Args:
        frame: (H, W, 3) image.
        kp_3d: (N, 3) keypoint positions in calibration frame (mm).
        cam: Camera parameters.
        colors: list of BGR colors per keypoint.
        radius: circle radius.
        marker_style: "circle" for open circle, "cross" for X, "filled" for solid dot.
        label: optional text label to draw near first keypoint.

    Returns:
        frame with keypoints drawn.
    """
    pts_2d = cam.project_points(kp_3d).astype(np.int32)
    for i, (pt, color) in enumerate(zip(pts_2d, colors)):
        x, y = tuple(pt)
        if marker_style == "circle":
            cv2.circle(frame, (x, y), radius, color, 2)
        elif marker_style == "cross":
            d = radius
            cv2.line(frame, (x - d, y - d), (x + d, y + d), color, 2)
            cv2.line(frame, (x - d, y + d), (x + d, y - d), color, 2)
        elif marker_style == "filled":
            cv2.circle(frame, (x, y), radius, color, -1)
    if label and len(pts_2d) > 0:
        lx, ly = pts_2d[0]
        cv2.putText(frame, label, (lx + 8, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return frame


def load_trial_csv(csv_path: str) -> dict:
    """Load trial start/end frame info from CSV.

    Returns:
        dict mapping trial_number -> {'trial_start': int, 'start': int, 'end': int}
    """
    trials = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial = int(row["trial"])
            start = int(row["start"])
            end = int(row["end"].rstrip(","))
            trial_start = int(row["trial_start"])
            if start != 0 or end != 0:
                trials[trial] = {
                    "trial_start": trial_start,
                    "start": start,
                    "end": end,
                }
    return trials


def render_reprojection_video(
    stac_h5_path: str,
    calibration_path: str,
    video_dir: str,
    env_points3d_path: "str | None",
    registered_h5_path: str,
    raw_points3d_path: str,
    clip_csv_path: "str | None",
    trial_number: "int | None",
    output_path: str,
    fps: int = 30,
    render_size: tuple = (512, 512),
):
    """Render a multi-view reprojection video for one trial.

    Args:
        stac_h5_path: Path to STAC'd IK H5 file.
        calibration_path: Path to calibration.toml.
        video_dir: Directory with CamA/, CamB/, CamC/ subdirs containing mp4s.
        env_points3d_path: Path to env points3d.h5 (bars + water spout), or
            None to skip drawing environmental rig markers entirely (not
            every dataset tracks them).
        registered_h5_path: Path to *_mujoco_registered_forelimb_right_v4.h5.
        raw_points3d_path: Path to session points3d.h5 (raw triangulated keypoints).
        clip_csv_path: Path to trial CSV with frame indices, or None if the
            video file IS the trial (one fixed-length clip per file).
        trial_number: Which trial to render, or None (with clip_csv_path=None)
            to render the whole clip starting at video frame 0.
        output_path: Output MP4 path.
        fps: Output video FPS.
        render_size: (width, height) for MuJoCo rendering per camera.
    """
    print(f"Loading data for trial {trial_number}...")

    # Load STAC data
    cfg, stac_data = io.load_stac_data(stac_h5_path)
    qposes = stac_data.qpos
    kp_data = stac_data.kp_data  # (n_frames, n_kp*3) in MuJoCo frame - raw mocap input
    marker_sites = stac_data.marker_sites  # (n_frames, n_kp, 3) - FK-computed site positions
    offsets = stac_data.offsets
    n_frames = qposes.shape[0]
    n_kp = len(stac_data.kp_names)

    # Load calibration
    cameras = load_calibration(calibration_path)
    print(f"  Loaded {len(cameras)} cameras: {[c.name for c in cameras]}")

    # Load registration transform
    reg = load_registration_transform(registered_h5_path)

    # Load raw shoulder positions (calibration frame, mm)
    raw_shoulders = load_raw_shoulder_positions(raw_points3d_path)

    # Load env markers (optional -- pass env_points3d_path=None to skip; not
    # every dataset tracks fixed rig landmarks like bars/water spout)
    if env_points3d_path is not None:
        env_data = load_env_points3d(env_points3d_path)
        env_points = env_data["points"]  # (N, 3) in calibration frame (mm)
        env_names = env_data["names"]
    else:
        env_points = None
        env_names = None

    # Load trial info. clip_csv_path/trial_number are only needed to look up a
    # start-frame offset within a longer multi-trial session recording; if the
    # video file itself IS the trial (e.g. the js_luc3d_hand CFL_* dataset,
    # one fixed-length clip per trial dir), pass both as None to skip the CSV
    # lookup entirely and start from frame 0 of the video.
    if clip_csv_path is None or trial_number is None:
        video_start_frame = 0
        print(f"  No clip CSV (trial_number=None): using whole clip, "
              f"video frames 0 to {n_frames - 1}")
    else:
        trials = load_trial_csv(clip_csv_path)
        if trial_number not in trials:
            raise ValueError(f"Trial {trial_number} not found in CSV")
        trial_info = trials[trial_number]
        video_start_frame = trial_info["trial_start"] + trial_info["start"]
        print(f"  Trial {trial_number}: video frames {video_start_frame} to {video_start_frame + n_frames - 1}")

    # Open video captures
    video_files = {}
    for cam in cameras:
        # Try multiple case variants for camera directory name
        cam_dir = None
        # Handle e.g. "camA" → "CamA", "CAMA", etc.
        title_case = cam.name[0].upper() + cam.name[1:] if cam.name else cam.name
        for variant in [cam.name, cam.name.lower(), cam.name.upper(), title_case]:
            candidate = Path(video_dir) / variant
            if candidate.is_dir():
                cam_dir = candidate
                break
        if cam_dir is None:
            raise FileNotFoundError(f"No camera directory found for {cam.name} in {video_dir}")
        mp4s = list(cam_dir.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError(f"No MP4 found in {cam_dir}")
        video_files[cam.name] = str(mp4s[0])

    caps = {}
    for cam in cameras:
        cap = cv2.VideoCapture(video_files[cam.name])
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)
        caps[cam.name] = cap

    # Compute calibration cameras in MuJoCo frame
    # Use fixed shoulder reference if available, otherwise per-frame
    shoulder_frame = reg.shoulder_reference_mm if reg.shoulder_reference_mm is not None else raw_shoulders[video_start_frame]
    mj_cam_params = []
    for cam_params in cameras:
        cp = _compute_calibration_cam_xyaxes(cam_params, reg, shoulder_frame)
        cp["name"] = cam_params.name
        mj_cam_params.append(cp)

    # Set up MuJoCo model with calibration cameras
    mj_model, mj_data = _setup_mujoco_model(cfg, stac_data.kp_names, mj_cam_params)
    w, h = render_size
    _set_green_background(mj_model, offwidth=max(512, w), offheight=max(512, h))
    scene_option = _make_scene_option()

    # Map camera names to MuJoCo camera indices
    cam_name_to_idx = {}
    for cam_params in cameras:
        mj_cam_name = f"calib_{cam_params.name}"
        for i in range(mj_model.ncam):
            if mj_cam_name in mj_model.camera(i).name:
                cam_name_to_idx[cam_params.name] = i
                break

    # Create renderers
    renderers = {}
    for cam_params in cameras:
        renderers[cam_params.name] = mujoco.Renderer(mj_model, height=h, width=w)

    # Keypoint colors (BGR for OpenCV)
    kp_colors = []
    for name in stac_data.kp_names:
        rgba_str = cfg.model.KEYPOINT_COLOR_PAIRS[name]
        rgba = [float(x) for x in rgba_str.split()]
        kp_colors.append((int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)))

    # Render frames
    print(f"  Rendering {n_frames} frames across {len(cameras)} cameras...")

    with imageio.get_writer(output_path, fps=fps) as video:
        for frame_idx in tqdm(range(n_frames)):
            panels = []
            abs_frame = video_start_frame + frame_idx
            shoulder_calib = raw_shoulders[abs_frame]
            # Use fixed shoulder reference if available (v6+ registration),
            # otherwise use per-frame raw shoulder (v4 registration)
            inv_shoulder = reg.shoulder_reference_mm if reg.shoulder_reference_mm is not None else shoulder_calib

            # Get mocap keypoints (raw input) in calibration frame
            kp_mj = kp_data[frame_idx].reshape(n_kp, 3)
            kp_calib = reg.mujoco_to_calibration(kp_mj, inv_shoulder)

            # Get STAC marker sites (FK fitted positions) in calibration frame
            ms_mj = marker_sites[frame_idx]
            ms_calib = reg.mujoco_to_calibration(ms_mj, inv_shoulder)

            for cam_params in cameras:
                # Read and undistort video frame
                ret, frame = caps[cam_params.name].read()
                if not ret:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    frame = cam_params.undistort_frame(frame)
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))

                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Render MuJoCo model from this camera's viewpoint
                cam_idx = cam_name_to_idx[cam_params.name]
                model_rgba = _render_model_frame(
                    mj_model, mj_data, renderers[cam_params.name],
                    qposes[frame_idx], cam_idx, scene_option,
                )

                # Composite model onto video frame
                frame = _composite(frame, model_rgba, alpha_scale=0.5)

                # Draw env markers (if provided)
                if env_points is not None:
                    frame = _draw_env_markers(frame, env_points, env_names, cam_params)

                # Draw raw mocap keypoints (open circles)
                frame = _draw_mocap_keypoints(
                    frame, kp_calib, cam_params, kp_colors,
                    radius=5, marker_style="circle", label="mocap",
                )

                # Draw STAC marker sites (X marks)
                frame = _draw_mocap_keypoints(
                    frame, ms_calib, cam_params, kp_colors,
                    radius=5, marker_style="cross", label="stac",
                )

                # Convert BGR to RGB for imageio
                panels.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Tile panels horizontally
            grid_frame = np.concatenate(panels, axis=1)
            video.append_data(grid_frame)

    # Cleanup
    for cap in caps.values():
        cap.release()
    for r in renderers.values():
        r.close()

    print(f"  Saved to {output_path}")


def main():
    """Render reprojection video for a single trial (command-line entry point)."""
    import argparse

    parser = argparse.ArgumentParser(description="Reproject STAC model onto multi-camera video")
    parser.add_argument("--stac-h5", required=True, help="Path to STAC'd IK H5 file")
    parser.add_argument("--calibration", required=True, help="Path to calibration.toml")
    parser.add_argument("--video-dir", required=True, help="Directory with CamA/CamB/CamC mp4s")
    parser.add_argument("--env-points3d", required=True, help="Path to env points3d.h5")
    parser.add_argument("--registered-h5", required=True, help="Path to registered H5 (v4)")
    parser.add_argument("--raw-points3d", required=True, help="Path to session points3d.h5")
    parser.add_argument("--clip-csv", required=True, help="Path to trial CSV")
    parser.add_argument("--trial", type=int, required=True, help="Trial number")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    render_reprojection_video(
        stac_h5_path=args.stac_h5,
        calibration_path=args.calibration,
        video_dir=args.video_dir,
        env_points3d_path=args.env_points3d,
        registered_h5_path=args.registered_h5,
        raw_points3d_path=args.raw_points3d,
        clip_csv_path=args.clip_csv,
        trial_number=args.trial,
        output_path=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
