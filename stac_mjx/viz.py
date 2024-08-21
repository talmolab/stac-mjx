"""A collection mujoco-mjx vizualization utilities."""

import pickle
import numpy as np
from pathlib import Path
from stac_mjx.controller import STAC
from omegaconf import DictConfig
from typing import Union, Dict


def viz_stac(
    data_path: Union[Path, str],
    stac_cfg: DictConfig,
    model_cfg: Dict,
    n_frames: int,
    save_path: Union[Path, str],
    start_frame: int = 0,
    camera: Union[int, str] = 0,
    height: int = 1200,
    width: int = 1920,
    base_path: Path = Path.cwd(),
):
    """Render forward kinematics from keypoint positions.

    Args:
        data_path (Union[Path, str]): Path to stac output pickle file
        stac_cfg (DictConfig): stac_cfg file
        model_cfg (Dict): model_cfg file
        n_frames (int): number of frames to render
        save_path (Union[Path, str]): Path to save rendered video (.mp4)
        start_frame (int, optional): Starting rendering frame. Defaults to 0.
        camera (Union[int, str], optional): Camera name (or number), defined in MJCF. Defaults to 0.
        height (int, optional): Rendering pixel height. Defaults to 1200.
        width (int, optional): Rendering pixel width. Defaults to 1920.
        base_path (Path, optional): Base path for path strings in configs. Defaults to Path.cwd().

    Returns:
        (List): List of frames
    """

    xml_path = base_path / model_cfg["MJCF_PATH"]

    # Load data
    with open(data_path, "rb") as file:
        d = pickle.load(file)
        qposes = np.array(d["qpos"])
        kp_data = np.array(d["kp_data"])
        kp_names = d["kp_names"]
        offsets = d["offsets"]

    # initialize STAC to create mj_model with scaling and marker body sites according to config
    # Set the learned offsets for body sites manually
    stac = STAC(xml_path, stac_cfg, model_cfg, kp_names)
    return stac.render(
        qposes,
        kp_data,
        offsets,
        n_frames,
        save_path,
        start_frame,
        camera,
        height,
        width,
    )
