"""Utility functions to load data from .mat .yaml and .h5 files."""

import numpy as np
from jax import numpy as jnp
import h5py
import os
import yaml
import scipy.io as spio
import pickle
from typing import Text
from pynwb import NWBFile, NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation

# Constants 
MM_TO_M = 0.001

def daance_to_stac_mjx(data):
    """
    Rearrange the data daance .mat mocap ordering
    to internal order, and scale from millimeters to
    meters. 

    Args: mocap data arranged in dannce ordering, and
    stored in mm.

    Out: data rearranged into internal format, scaled to 
    meters.
    """

    kp_names = params["KP_NAMES"]
    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)

    # Dannce data is stored in mm
    data = data * MM_TO_M
    data = jnp.array(data[:, :, stac_keypoint_order])
    data = jnp.transpose(data, (0, 2, 1))
    data = jnp.reshape(data, (data.shape[0], -1))

    return data


def load_dannce_mat(data_path):
    """
    loads in mocap data from .mat file constructed by dannce:
    (https://github.com/spoonsso/dannce). in particular this means
    it relies on the data being in millimeters, and that we use the data
    stored in the "pred" key. 
    """

    data = spio.loadmat(data_path, struct_as_record=False, squeeze_me=True)
    data =  _check_keys(data)["pred"][:]

    return daance_to_stac_mjx(data)


def load_dannce_nwb(filename):
    """
    loads data in from .nwb file. Presumed organized and scaled 
    equivalent to a dannce .mat file, that has been converted to
    """
    data = []
    with NWBHDF5IO(filename, mode='r', load_namespaces=True) as io:
        nwbfile = io.read()
        pose_est = nwbfile.processing['behavior']['PoseEstimation']
        #print(pose_est.shape)

        node_names = pose_est.nodes[:].tolist()

        for node_name in node_names:
            data.append(pose_est[node_name].data[:])

        data = np.stack(data, axis=-1)

    return daance_to_stac_mjx(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
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


def init_params(cfg):
    global params
    params = cfg


# TODO put this in the STAC class
def save(fit_data, save_path: Text):
    """Save data.

    Args:
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
