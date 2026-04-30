"""MuJoCo-MJX visualization utilities."""

import numpy as np
from pathlib import Path
from stac_mjx.stac import Stac
from stac_mjx import io
from omegaconf import DictConfig


def viz_stac(
    data_path: Path | str,
    n_frames: int,
    save_path: Path | str,
    start_frame: int = 0,
    camera: int | str = 0,
    height: int = 1200,
    width: int = 1920,
    base_path: Path | None = None,
    show_marker_error: bool = False,
) -> tuple[DictConfig, list[np.ndarray]]:
    """Render forward kinematics from STAC output data.

    Args:
        data_path: Path to STAC output HDF5 file.
        n_frames: Number of frames to render.
        save_path: Output video file path (.mp4).
        start_frame: First frame to render.
        camera: MuJoCo camera name or index.
        height: Render height in pixels.
        width: Render width in pixels.
        base_path: Base path for resolving config paths. Defaults to cwd.
        show_marker_error: Whether to show marker-keypoint distance.

    Returns:
        Tuple of (config, list of rendered RGB frames).
    """
    cfg, d = io.load_stac_data(data_path)
    qposes = d.qpos
    kp_data = d.kp_data
    kp_names = d.kp_names
    offsets = d.offsets

    if base_path is None:
        base_path = Path.cwd()

    xml_path = base_path / cfg.model.MJCF_PATH

    stac = Stac(xml_path, cfg, kp_names)

    return cfg, stac.render(
        qposes,
        kp_data,
        offsets,
        n_frames,
        save_path,
        start_frame,
        camera,
        height,
        width,
        show_marker_error,
    )
