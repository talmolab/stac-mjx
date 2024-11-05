"""A collection mujoco-mjx vizualization utilities."""

import pickle
import numpy as np
from pathlib import Path
from stac_mjx.stac import Stac
from omegaconf import DictConfig
from typing import Union, Dict

# FLY_MODEL
# import stac_mjx.io_dict_to_hdf5 as ioh5


def viz_stac(
    data_path: Union[Path, str],
    cfg: DictConfig,
    n_frames: int,
    save_path: Union[Path, str],
    start_frame: int = 0,
    camera: Union[int, str] = 0,
    height: int = 1200,
    width: int = 1920,
    base_path=None,
    show_marker_error=False,
):
    """Render forward kinematics from keypoint positions.

    Args:
        data_path (Union[Path, str]): Path to stac output pickle file
        cfg (DictConfig): configs
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
    if base_path is None:
        base_path = Path.cwd()

    xml_path = base_path / cfg.model.MJCF_PATH
    print("xml_path ", xml_path)

    # Load data
    with open(data_path, "rb") as file:
        d = pickle.load(file)
        qposes = np.array(d["qpos"])
        kp_data = np.array(d["kp_data"])
        kp_names = d["kp_names"]
        offsets = d["offsets"]

    #FLY_MODEL
    if data_path.suffix == ".h5":
        data = ioh5.load(data_path)
        qposes = np.array(data["qpos"])
        kp_data = np.array(data["kp_data"])
        kp_names = data["kp_names"]
        offsets = data["offsets"]
    else:
        with open(data_path, "rb") as file:
            d = pickle.load(file)
            qposes = np.array(d["qpos"])
            kp_data = np.array(d["kp_data"])
            kp_names = d["kp_names"]
            offsets = d["offsets"]

    # initialize stac to create mj_model with scaling and marker body sites according to config
    # Set the learned offsets for body sites manually
    stac = Stac(xml_path, cfg, kp_names)
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
        show_marker_error,
    )
