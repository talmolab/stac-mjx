"""Utility functions to load data from .mat .yaml and .h5 files."""

import numpy as np
import h5py
import os
import yaml
import scipy.io as spio
import pickle
from typing import Text
from pynwb import NWBFile, NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    print("Pred - mat", data["pred"].shape)
    print(data["pred"][0,0,:])
    return _check_keys(data)

def loadnwb(filename):
    """
    loads data in from .nwb file
    """
    trx = []
    with NWBHDF5IO(filename, mode='r', load_namespaces=True) as io:
        read_nwbfile = io.read()
        read_pe = read_nwbfile.processing['behavior']['PoseEstimation']

        node_names = read_pe.nodes[:].tolist()

        for node_name in node_names:
            trx.append(read_pe[node_name].data[:])

        trx = np.stack(trx, axis=-1)

    print("nwb shape", trx.shape)
    print(trx[0,0,:])
    
    return trx


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
