"""Utility functions to load data from .mat .yaml and .h5 files."""

import numpy as np
import jax.numpy as jp
import scipy.io as spio
from pynwb import NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation
import h5py
from pathlib import Path
from omegaconf import DictConfig
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict, field

from stac_mjx.config import Config


@dataclass
class StacData:
    """Data structure for STAC output."""

    qpos: np.ndarray  # Root position and quaternion, and joint angles
    xpos: np.ndarray  # Body positions
    xquat: np.ndarray  # Body quaternions
    marker_sites: np.ndarray  # Marker site positions
    offsets: np.ndarray  # Marker site offsets
    kp_data: np.ndarray  # Keypoint data
    names_qpos: list[str]  # Names of qpos
    names_xpos: list[str]  # Names of xpos
    kp_names: list[str]  # Names of keypoints
    qvel: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # Inferred joint velocities, OPTIONAL

    def as_dict(self) -> dict:
        """Convert the dataclass instance to a dictionary."""
        return asdict(self)


def load_data(
    cfg: DictConfig, base_path: Path | None = None
) -> tuple[jp.ndarray, list[str]]:
    """Load mocap data, flatten, and scale for STAC consumption.

    Loads mocap file based on filetype, sorts keypoints to match model order,
    scales by MOCAP_SCALE_FACTOR, and flattens from [frames, keypoints, xyz]
    to [frames, keypoints*xyz].

    Args:
        cfg: STAC configuration.
        base_path: Base path for resolving relative file paths. Defaults to cwd.

    Returns:
        Tuple of (flattened mocap array, sorted keypoint names).

    Raises:
        ValueError: If unsupported filetype or keypoint names missing/mismatched.
    """
    if base_path is None:
        base_path = Path.cwd()

    file_path = base_path / cfg.stac.data_path
    if file_path.suffix == ".mat":
        label3d_path = cfg.model.get("KP_NAMES_LABEL3D_PATH", None)
        data, kp_names = load_dannce(str(file_path), names_filename=label3d_path)
    elif file_path.suffix == ".nwb":
        data, kp_names = load_nwb(file_path)
    elif file_path.suffix == ".h5":
        data, kp_names = load_h5(file_path)
    else:
        raise ValueError(
            "Unsupported file extension. Please provide a .mat, .nwb, or .h5 file."
        )

    kp_names = kp_names or cfg.model.KP_NAMES

    if kp_names is None:
        raise ValueError(
            "Keypoint names not provided. Please provide an ordered list of keypoint names "
            "corresponding to the keypoint data order."
        )

    if len(kp_names) != data.shape[2]:
        raise ValueError(
            f"Number of keypoint names ({len(kp_names)}) is not the same as the number of keypoints in data ({data.shape[2]})"
        )

    model_inds = [
        kp_names.index(src) for src, dst in cfg.model.KEYPOINT_MODEL_PAIRS.items()
    ]

    sorted_kp_names = [kp_names[i] for i in model_inds]

    data = data * cfg.model.MOCAP_SCALE_FACTOR
    data = jp.array(data[:, :, model_inds])
    data = jp.transpose(data, (0, 2, 1))
    data = jp.reshape(data, (data.shape[0], -1))

    return data, sorted_kp_names


def load_dannce(
    filename: str | Path, names_filename: str | Path | None = None
) -> tuple[np.ndarray, list[str] | None]:
    """Load mocap data from a DANNCE .mat file.

    Expects data in millimeters with shape [frames, keypoints, xyz],
    stored under the "pred" key.

    Args:
        filename: Path to the .mat file.
        names_filename: Optional path to a .mat file containing joint names.

    Returns:
        Tuple of (mocap data array, keypoint names or None).
    """
    node_names = None
    if names_filename is not None:
        mat = spio.loadmat(names_filename)
        node_names = [item[0] for sublist in mat["joint_names"] for item in sublist]

    data = _check_keys(spio.loadmat(filename, struct_as_record=False, squeeze_me=True))[
        "pred"
    ]
    return data, node_names


def load_nwb(filename: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load mocap data from a .nwb file.

    Expects data with shape [frames, keypoints, xyz].

    Args:
        filename: Path to the .nwb file.

    Returns:
        Tuple of (mocap data array, keypoint names).
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


def load_h5(filename: str | Path) -> tuple[np.ndarray, None]:
    """Load mocap data from a .h5 file.

    Expects data with shape [frames, xyz, keypoints]. Transposes to
    [frames, keypoints, xyz] on output.

    Args:
        filename: Path to the .h5 file.

    Returns:
        Tuple of (mocap data array, None) since .h5 files lack keypoint names.
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


def _check_keys(d: dict) -> dict:
    """Convert mat-objects in a dictionary to nested dictionaries."""
    for key in d:
        if isinstance(d[key], spio.matlab.mat_struct):
            d[key] = _todict(d[key])
    return d


def _todict(matobj: spio.matlab.mat_struct) -> dict:
    """Recursively convert a mat_struct to a nested dictionary."""
    result = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            result[strg] = _todict(elem)
        else:
            result[strg] = elem
    return result


def save_data_to_h5(
    config: Config,
    kp_names: list[str],
    names_qpos: list[str],
    names_xpos: list[str],
    kp_data: np.ndarray,
    marker_sites: np.ndarray,
    offsets: np.ndarray,
    qpos: np.ndarray,
    xpos: np.ndarray,
    xquat: np.ndarray,
    qvel: np.ndarray,
    file_path: str | Path,
) -> None:
    """Save configuration and STAC data to an HDF5 file.

    Args:
        config: STAC configuration dataclass.
        kp_names: Ordered keypoint names.
        names_qpos: Generalized coordinate names.
        names_xpos: Body position names.
        kp_data: Keypoint data array.
        marker_sites: Marker site positions.
        offsets: Marker site offsets.
        qpos: Generalized coordinates.
        xpos: Body positions.
        xquat: Body quaternions.
        qvel: Generalized velocities.
        file_path: Output HDF5 file path.
    """
    with h5py.File(file_path, "w") as f:
        config_yaml = OmegaConf.to_yaml(OmegaConf.structured(config))
        f.create_dataset("config", data=np.bytes_(config_yaml))

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


def load_stac_data(file_path: str | Path) -> tuple[Config, StacData]:
    """Load configuration and STAC data from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Tuple of (Config, StacData).
    """
    with h5py.File(file_path, "r") as f:
        config_yaml = f["config"][()].decode("utf-8")
        config = OmegaConf.create(config_yaml)
        config = OmegaConf.structured(Config(**config))

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
