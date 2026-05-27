#!/usr/bin/env python3
"""Interactive MuJoCo viewer for replaying STAC fit results.

Provides both a Python API (:func:`replay`) and a CLI entry point
(``stac-viewer``) for animating STAC output H5 files in MuJoCo's
passive viewer with optional keypoint / marker-site overlays.

Examples::

    # CLI
    stac-viewer fit_offsets.h5
    stac-viewer ik_only.h5 --xml models/rodent.xml --fps 60

    # Python
    from stac_mjx.replay_fit import replay
    replay("fit_offsets.h5")
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Union

import h5py
import mujoco
import numpy as np
import yaml


_COMMON_ROOT_BODIES = ("torso", "reference_base", "thorax", "trunk", "body")


def _resolve_xml_path(mjcf_path: str, h5_path: Union[str, Path]) -> Path:
    """Find the MJCF XML on disk, searching several locations."""
    p = Path(mjcf_path)
    if p.is_absolute() and p.exists():
        return p

    h5_dir = Path(h5_path).resolve().parent
    candidates = [
        h5_dir / p,
        h5_dir / p.name,
        Path.cwd() / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        f"Could not find MJCF model '{mjcf_path}'.\n"
        "Searched:\n" + "\n".join(f"  {c}" for c in candidates) + "\n"
        "Use --xml to specify the path explicitly."
    )


def _detect_root_body(model: mujoco.MjModel, hint: Optional[str] = None) -> int:
    """Return the body id to track with the camera."""
    if hint is not None:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hint)
        if bid >= 0:
            return bid
        raise ValueError(
            f"Root body '{hint}' not found in model. "
            f"Available bodies: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]}"
        )
    for name in _COMMON_ROOT_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    return 1  # first non-world body


def _load_h5_for_replay(h5_path: Union[str, Path]) -> dict:
    """Load STAC output H5 with minimal dependencies (h5py + numpy)."""
    with h5py.File(h5_path, "r") as f:
        result: dict = {}

        if "config" in f:
            config_yaml = f["config"][()].decode("utf-8")
            result["config"] = yaml.safe_load(config_yaml)

        if "qpos" not in f:
            raise ValueError(
                f"{h5_path} does not contain 'qpos'. "
                "This does not look like a STAC output file."
            )
        result["qpos"] = f["qpos"][:]

        for key in ("kp_data", "marker_sites"):
            if key in f:
                result[key] = f[key][:]

    return result


def _normalize_shapes(data: dict) -> dict:
    """Flatten batched (n_clips, n_per_clip, ...) arrays to (n_frames, ...)."""
    qpos = data["qpos"]
    if qpos.ndim == 3:
        data["qpos"] = qpos.reshape(-1, qpos.shape[-1])
        for key in ("kp_data", "marker_sites"):
            if key in data:
                arr = data[key]
                data[key] = arr.reshape(-1, *arr.shape[2:])

    if "kp_data" in data and data["kp_data"].ndim == 2:
        kp = data["kp_data"]
        data["kp_data"] = kp.reshape(kp.shape[0], -1, 3)

    if "marker_sites" in data and data["marker_sites"].ndim == 2:
        ms = data["marker_sites"]
        data["marker_sites"] = ms.reshape(ms.shape[0], -1, 3)

    return data


def _auto_sphere_size(model: mujoco.MjModel) -> float:
    """Heuristic: sphere radius ~ 1/200 of the model's bounding extent."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    extents = data.xpos[1:].max(axis=0) - data.xpos[1:].min(axis=0)
    extent = max(extents.max(), 0.01)
    return float(extent / 200.0)


def replay(
    h5_path: Union[str, Path],
    xml_path: Optional[Union[str, Path]] = None,
    fps: Optional[float] = None,
    scale: Optional[float] = None,
    show_markers: bool = True,
    sphere_size: Optional[float] = None,
    root_body: Optional[str] = None,
) -> None:
    """Launch interactive MuJoCo viewer to replay a STAC fit.

    Args:
        h5_path: Path to a STAC output H5 file (from ``fit_offsets`` or
            ``ik_only``).
        xml_path: Path to the MJCF XML model. If *None*, auto-resolved
            from the config embedded in the H5.
        fps: Playback frames per second. If *None*, uses ``RENDER_FPS``
            from the embedded config (default 30).
        scale: Geometry scale factor. If *None*, uses ``SCALE_FACTOR``
            from the embedded config (default 1.0).
        show_markers: Overlay input keypoints (red) and fitted marker
            sites (green) as debug spheres.
        sphere_size: Radius of debug spheres in metres. If *None*,
            auto-scaled from the model's bounding extent.
        root_body: Name of the body the camera tracks. If *None*,
            tries common names (``torso``, ``reference_base``, ...)
            then falls back to the first non-world body.
    """
    h5_path = Path(h5_path).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    data = _load_h5_for_replay(h5_path)
    config = data.get("config", {})
    model_cfg = config.get("model", {})

    if xml_path is None:
        mjcf = model_cfg.get("MJCF_PATH")
        if mjcf is None:
            raise ValueError(
                "No MJCF_PATH in the H5 config and --xml not provided."
            )
        xml_path = _resolve_xml_path(mjcf, h5_path)
    else:
        xml_path = Path(xml_path).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"XML model not found: {xml_path}")

    if scale is None:
        scale = float(model_cfg.get("SCALE_FACTOR", 1.0))
    if fps is None:
        fps = float(model_cfg.get("RENDER_FPS", 30))

    data = _normalize_shapes(data)
    qpos = data["qpos"]
    kp_data = data.get("kp_data")
    marker_sites = data.get("marker_sites")
    has_markers = show_markers and kp_data is not None and marker_sites is not None

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if scale != 1.0:
        model.body_pos[:] *= scale
        model.geom_pos[:] *= scale
        model.geom_size[:] *= scale
        model.site_pos[:] *= scale
        if model.nmesh > 0:
            model.mesh_vert[:] *= scale
        print(f"  Model geometry scaled by {scale}")
    mj_data = mujoco.MjData(model)

    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"qpos width {qpos.shape[1]} != model.nq {model.nq}. "
            f"The H5 was probably fit against a different MJCF."
        )

    if sphere_size is None:
        sphere_size = _auto_sphere_size(model)

    root_body_id = _detect_root_body(model, root_body)
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)

    n_frames = qpos.shape[0]
    frame_dt = 1.0 / fps

    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_step = 0.001

    paused = False
    speed_presets = (0.1, 0.25, 0.5, 1.0, 2.0)
    speed_idx = speed_presets.index(1.0)
    frame_idx = 0

    print(f"Replaying {n_frames} frames from {h5_path.name}")
    print(f"  XML: {xml_path}")
    print(f"  qpos: {qpos.shape}, model.nq: {model.nq}")
    if has_markers:
        print(f"  kp_data: {kp_data.shape}, marker_sites: {marker_sites.shape}")
    print(f"  FPS: {fps}, scale: {scale}, tracking body: '{root_name}' (id {root_body_id})")
    print(f"  sphere_size: {sphere_size:.4f} m")
    print()
    print("Controls:")
    print("  SPACE        pause / play")
    print("  LEFT/RIGHT   step -1 / +1 frame")
    print(f"  UP/DOWN      speed: {list(speed_presets)} (start 1.0x)")
    if floor_gid >= 0:
        print(f"  - / =        nudge floor z by +/- 1 mm")
    if has_markers:
        print("  Markers:  red = input keypoints,  green = fitted marker sites")
    print()

    last_cam_print = 0.0

    def key_callback(keycode: int) -> None:
        nonlocal paused, frame_idx, speed_idx
        if keycode == 32:  # SPACE
            paused = not paused
            print(f"\n[{'PAUSED' if paused else 'PLAYING'}] frame {frame_idx}")
        elif keycode == 263:  # LEFT
            frame_idx = (frame_idx - 1) % n_frames
            print(f"\n[STEP -] frame {frame_idx}")
        elif keycode == 262:  # RIGHT
            frame_idx = (frame_idx + 1) % n_frames
            print(f"\n[STEP +] frame {frame_idx}")
        elif keycode == 265:  # UP
            speed_idx = min(speed_idx + 1, len(speed_presets) - 1)
            print(f"\n[SPEED] {speed_presets[speed_idx]}x")
        elif keycode == 264:  # DOWN
            speed_idx = max(speed_idx - 1, 0)
            print(f"\n[SPEED] {speed_presets[speed_idx]}x")
        elif keycode in (45, 61) and floor_gid >= 0:  # '-' / '='
            model.geom_pos[floor_gid, 2] += floor_step if keycode == 61 else -floor_step
            wz = float(model.geom_pos[floor_gid, 2])
            print(f"\n[FLOOR] z={wz:+.5f}")

    import mujoco.viewer  # noqa: delayed so headless imports don't break

    with mujoco.viewer.launch_passive(model, mj_data, key_callback=key_callback) as viewer:
        viewer.cam.distance = sphere_size * 30
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 90.0
        if has_markers:
            viewer.user_scn.ngeom = 0

        while viewer.is_running():
            t0 = time.time()

            mj_data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(model, mj_data)
            viewer.cam.lookat[:] = mj_data.xpos[root_body_id]

            now = time.time()
            if now - last_cam_print > 0.25:
                lk = viewer.cam.lookat
                print(
                    f"\rframe {frame_idx:4d}/{n_frames}  "
                    f"lookat=({lk[0]:+.4f},{lk[1]:+.4f},{lk[2]:+.4f})  "
                    f"dist={viewer.cam.distance:.4f}  "
                    f"az={viewer.cam.azimuth:6.1f}  el={viewer.cam.elevation:+5.1f}",
                    end="", flush=True,
                )
                last_cam_print = now

            if has_markers:
                viewer.user_scn.ngeom = 0
                sz = np.array([sphere_size, 0, 0])
                identity = np.eye(3).flatten()
                red = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)
                green = np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)

                for p in kp_data[frame_idx]:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=sz,
                        pos=p.astype(np.float64),
                        mat=identity,
                        rgba=red,
                    )
                    viewer.user_scn.ngeom += 1

                for p in marker_sites[frame_idx]:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=sz,
                        pos=p.astype(np.float64),
                        mat=identity,
                        rgba=green,
                    )
                    viewer.user_scn.ngeom += 1

            viewer.sync()

            if not paused:
                frame_idx = (frame_idx + 1) % n_frames
            dt = (frame_dt / speed_presets[speed_idx]) - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def main() -> None:
    """CLI entry point for ``stac-viewer``."""
    parser = argparse.ArgumentParser(
        prog="stac-viewer",
        description="Interactive MuJoCo viewer for STAC fit results.",
    )
    parser.add_argument(
        "h5",
        help="Path to a STAC output H5 file (fit_offsets or ik_only).",
    )
    parser.add_argument(
        "--xml",
        default=None,
        help="Path to MJCF XML model. Auto-resolved from H5 config if omitted.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Playback fps (default: from H5 config, or 30).",
    )
    parser.add_argument(
        "--scale", type=float, default=None,
        help="Geometry scale factor (default: from H5 config, or 1.0).",
    )
    parser.add_argument(
        "--no-markers", action="store_true",
        help="Don't overlay input keypoints / marker sites.",
    )
    parser.add_argument(
        "--sphere-size", type=float, default=None,
        help="Radius (m) of debug spheres (default: auto-scaled from model).",
    )
    parser.add_argument(
        "--root-body", default=None,
        help="Body name for camera tracking (default: auto-detect).",
    )
    args = parser.parse_args()

    replay(
        h5_path=args.h5,
        xml_path=args.xml,
        fps=args.fps,
        scale=args.scale,
        show_markers=not args.no_markers,
        sphere_size=args.sphere_size,
        root_body=args.root_body,
    )


if __name__ == "__main__":
    main()
