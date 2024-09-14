"""Utility functions to load data from .mat .yaml and .h5 files."""

import os
import numpy as np
from jax import numpy as jnp
import yaml
import scipy.io as spio
import pickle
from typing import Text
from pynwb import NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation
import h5py
from pathlib import Path
from typing import Dict
from jax.lib import xla_bridge
import os


def enable_xla_flags():
    """Enables XLA Flags for faster runtime on Nvidia GPUs."""
    if xla_bridge.get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
        )


def load_data(file_path: Path, params: Dict, label3d_path=None):
    """Main mocap data file loader interface.

    Loads mocap file based on filetype, and returns the data flattened
    for immediate consumption by stac_mjx algorithm.

    Args:
        file_path: path to be loaded, which should have a supported
        file type suffix, either .mat or .nwb, and presumed to be organized
        as [num frames, num keypoints, xyz].
        params:

    Returns:
        Mocap data flattened into an np array of shape [#frames, keypointXYZ],
        where 'keypointXYZ' represents the flattened 3D keypoint components.
        The data is also scaled by multiplication with "MOCAP_SCALE_FACTOR", e.g.
        if the mocap data is in mm and the model is in meters, this should be
        0.001.

    Raises:
        ValueError if an unsupported filetype is encountered.
    """
    # using pathlib
    if file_path.suffix == ".mat":
        data, kp_names = load_dannce(str(file_path), names_filename=label3d_path)
    elif file_path.suffix == ".nwb":
        data, kp_names = load_nwb(file_path)
    elif file_path.suffix == ".h5":
        data, kp_names = load_h5(file_path)
    else:
        raise ValueError(
            "Unsupported file extension. Please provide a .nwb or .mat file."
        )

    kp_names = kp_names or params["KP_NAMES"]

    if kp_names is None:
        raise ValueError(
            "Keypoint names not provided. Please provide an ordered list of keypoint names \
            corresponding to the keypoint data order."
        )

    if len(kp_names) != data.shape[2]:
        raise ValueError(
            f"Number of keypoint names ({len(kp_names)}) is not the same as the number of keypoints in data ({data.shape[1]})"
        )

    model_inds = np.array(
        [kp_names.index(src) for src, dst in params["KEYPOINT_MODEL_PAIRS"].items()]
    )
    sorted_kp_names = [kp_names[i] for i in model_inds]

    # Scale mocap data to match model
    data = data * params["MOCAP_SCALE_FACTOR"]
    # Sort in kp_names order
    data = jnp.array(data[:, :, model_inds])
    # Flatten data from [#num frames, #keypoints, xyz]
    # into [#num frames, #keypointsXYZ]
    data = jnp.transpose(data, (0, 2, 1))
    data = jnp.reshape(data, (data.shape[0], -1))

    return data, sorted_kp_names


def load_dannce(filename, names_filename=None):
    """Loads mocap data from .mat file.

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
    """Loads mocap data from .nwb file.

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
    # TODO add track information
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]

    data = np.array(data["tracks"])
    data = np.squeeze(data, axis=1)
    data = np.transpose(data, (0, 2, 1))
    return data, None


def _check_keys(dict):
    """Checks if entries in dictionary are mat-objects.

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


def _load_params(param_path):
    """Load parameters for the animal.

    :param param_path: Path to .yaml file specifying animal parameters.
    """
    with open(param_path, "r") as infile:
        try:
            params = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def save(fit_data, save_path: Text):
    """Save data.

    Save data as .p

    Args:
        fit_data (numpy array): Data to write out.
        save_path (Text): Path to save data. Defaults to None.
    """
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _, file_extension = os.path.splitext(save_path)
    if file_extension == ".p":
        with open(save_path, "wb") as output_file:
            pickle.dump(fit_data, output_file, protocol=2)
    else:
        with open(save_path + ".p", "wb") as output_file:
            pickle.dump(fit_data, output_file, protocol=2)
