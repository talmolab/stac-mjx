"""Utility functions to load data from .mat .yaml and .h5 files."""

import os
import numpy as np
from jax import numpy as jnp
import yaml
import scipy.io as spio
import pickle
from typing import Union
from pynwb import NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation
import h5py
from pathlib import Path
from omegaconf import DictConfig
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict, field
from typing import List

from stac_mjx.config import Config, ModelConfig, MujocoConfig, StacConfig  # re-export


@dataclass
class StacData:
    """Data structure for STAC output."""

    qpos: np.ndarray  # Root position and quaternion, and joint angles
    xpos: np.ndarray  # Body positions
    xquat: np.ndarray  # Body quaternions
    marker_sites: np.ndarray  # Marker site positions
    offsets: np.ndarray  # Marker site offsets
    kp_data: np.ndarray  # Keypoint data
    names_qpos: List[str]  # Names of qpos
    names_xpos: List[str]  # Names of xpos
    kp_names: List[str]  # Names of keypoints

    # Optional
    qvel: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Inferred joint velocities

    def as_dict(self) -> dict:
        """Convert the dataclass instance to a dictionary."""
        return asdict(self)


def load_mocap(cfg: DictConfig, base_path: Union[Path, None] = None):
    """Load mocap data based on file type.

    Loads mocap file based on filetype, and returns the data flattened
    for immediate consumption by stac_mjx algorithm.

    Args:
        cfg (DictConfig): Configs.
        base_path (Union[Path, None], optional): Base path for file paths in configs. Defaults to None.

    Returns:
        Mocap data flattened into an np array of shape [#frames, keypointXYZ],
        where 'keypointXYZ' represents the flattened 3D keypoint components.
        The data is also scaled by multiplication with "MOCAP_SCALE_FACTOR", e.g.
        if the mocap data is in mm and the model is in meters, this should be
        0.001.

    Raises:
        ValueError if an unsupported filetype is encountered.
        ValueError if ordered list of keypoint names is missing or
        does not match number of keypoints.
    """
    if base_path is None:
        base_path = Path.cwd()

    file_path = base_path / cfg.stac.data_path
    # using pathlib
    if file_path.suffix == ".mat":
        label3d_path = cfg.model.get("KP_NAMES_LABEL3D_PATH", None)
        data, kp_names = load_dannce(str(file_path), names_filename=label3d_path)
    elif file_path.suffix == ".nwb":
        data, kp_names = load_nwb(file_path)
    elif file_path.suffix == ".h5":
        data, kp_names = load_h5(file_path)
    else:
        raise ValueError(
            "Unsupported file extension. Please provide a .nwb or .mat file."
        )

    kp_names = kp_names or cfg.model.KP_NAMES

    if kp_names is None:
        raise ValueError(
            "Keypoint names not provided. Please provide an ordered list of keypoint names \
            corresponding to the keypoint data order."
        )

    if len(kp_names) != data.shape[2]:
        raise ValueError(
            f"Number of keypoint names ({len(kp_names)}) is not the same as the number of keypoints in data ({data.shape[1]})"
        )

    model_inds = [
        kp_names.index(src) for src, dst in cfg.model.KEYPOINT_MODEL_PAIRS.items()
    ]

    sorted_kp_names = [kp_names[i] for i in model_inds]

    # Scale mocap data to match model
    data = data * cfg.model.MOCAP_SCALE_FACTOR
    # Sort in kp_names order
    data = jnp.array(data[:, :, model_inds])
    # Flatten data from [#num frames, #keypoints, xyz]
    # into [#num frames, #keypointsXYZ]
    data = jnp.transpose(data, (0, 2, 1))
    data = jnp.reshape(data, (data.shape[0], -1))

    return data, sorted_kp_names


def load_dannce(filename, names_filename=None):
    """Load mocap data from .mat file.

    .mat file is presumed to be constructed by dannce:
    (https://github.com/spoonsso/dannce). In particular this means it relies on
    the data being in millimeters [num frames, num keypoints, xyz], and that we
    use the data stored in the "pred" key.
    """
    node_names = None
    if names_filename is not None:
        mat = spio.loadmat(names_filename)
        node_names = [item[0] for sublist in mat["joint_names"] for item in sublist]

    data = _check_keys(spio.loadmat(filename, struct_as_record=False, squeeze_me=True))[
        "pred"
    ]
    return data, node_names


def load_nwb(filename):
    """Load mocap data from .nwb file.

    Data is presumed [num frames, num keypoints, xyz].
    """
    data = []
    with NWBHDF5IO(filename, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        pose_est = nwbfile.processing["behavior"]["PoseEstimation"]
        node_names = pose_est.nodes[:].tolist()
        data = np.stack(
            [pose_est[node_name].data[:] for node_name in node_names], axis=-1
        )

    return data, node_names


def load_h5(filename):
    """Load .h5 file formatted as [frames, xyz, keypoints].

    Args:
        filename (str): Path to the .h5 file.

    Returns:
        dict: Dictionary containing the data from the .h5 file.
    """
    # TODO tracks is a hardcoded dataset name
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]

    data = np.array(data["tracks"])
    data = np.squeeze(data, axis=1)
    data = np.transpose(data, (0, 2, 1))
    return data, None


def _check_keys(dict):
    """Check if entries in dictionary are mat-objects.

    Mat-objects are changed to nested dictionaries.
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """A recursive function which constructs from matobjects nested dictionaries."""
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def save_data_to_h5(
    config: Config,
    kp_names: list,
    names_qpos: list,
    names_xpos: list,
    kp_data: np.ndarray,
    marker_sites: np.ndarray,
    offsets: np.ndarray,
    qpos: np.ndarray,
    xpos: np.ndarray,
    xquat: np.ndarray,
    qvel: np.ndarray,
    file_path: str,
):
    """Save configuration and STAC data to an HDF5 file.

    Args:
        config (Config): Configuration dataclass.
        kp_names (list): List of keypoint names.
        names_qpos (list): List of qpos names.
        names_xpos (list): List of xpos names.
        kp_data (np.ndarray): Keypoint data array.
        marker_sites (np.ndarray): Marker sites array.
        offsets (np.ndarray): Offsets array.
        qpos (np.ndarray): Qpos array.
        xpos (np.ndarray): Xpos array.
        xquat (np.ndarray): Xquat array.
        qvel (np.ndarray): Qvel array.
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, "w") as f:
        # Save config as a YAML string
        config_yaml = OmegaConf.to_yaml(OmegaConf.structured(config))
        f.create_dataset("config", data=np.bytes_(config_yaml))

        # Save stac output data
        f.create_dataset("kp_names", data=np.array(kp_names, dtype="S"))
        f.create_dataset("names_qpos", data=np.array(names_qpos, dtype="S"))
        f.create_dataset("names_xpos", data=np.array(names_xpos, dtype="S"))
        f.create_dataset("kp_data", data=kp_data, compression="gzip")
        f.create_dataset("marker_sites", data=marker_sites, compression="gzip")
        f.create_dataset("offsets", data=offsets, compression="gzip")
        f.create_dataset("qpos", data=qpos, compression="gzip")
        f.create_dataset("qvel", data=qvel, compression="gzip")
        f.create_dataset("xpos", data=xpos, compression="gzip")
        f.create_dataset("xquat", data=xquat, compression="gzip")


def load_stac_data(file_path) -> tuple[Config, StacData]:
    """Load configuration and STAC data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        tuple: A tuple containing the Config and StacData dataclasses.
    """
    with h5py.File(file_path, "r") as f:
        # Load config from YAML string
        config_yaml = f["config"][()].decode("utf-8")
        config = OmegaConf.create(config_yaml)
        config = OmegaConf.structured(Config(**config))

        # Load additional values
        kp_names = [name.decode("utf-8") for name in f["kp_names"]]
        names_qpos = [name.decode("utf-8") for name in f["names_qpos"]]
        names_xpos = [name.decode("utf-8") for name in f["names_xpos"]]
        kp_data = f["kp_data"][()]
        marker_sites = f["marker_sites"][()]
        offsets = f["offsets"][()]
        qpos = f["qpos"][()]
        qvel = f["qvel"][()]
        xpos = f["xpos"][()]
        xquat = f["xquat"][()]

        stac_data = StacData(
            kp_names=kp_names,
            names_qpos=names_qpos,
            names_xpos=names_xpos,
            kp_data=kp_data,
            marker_sites=marker_sites,
            offsets=offsets,
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
        )

    return config, stac_data
