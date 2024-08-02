from stac_mjx import utils
import os

"""
Test data loaders for supported file types.
"""

def test_load_nwb():
    """
    Test loading data from .nwb file.
    """
    params = utils._load_params("configs/rodent.yaml")
    data = utils.load_data("tests/data/test_mocap_1000_frames.nwb", params)
    assert data.shape == (1000, 69)


def test_load_mat_no_lablel3d():
    """
    Test loading data from .mat file.
    """
    params = utils._load_params("configs/rodent.yaml")
    data = utils.load_data("tests/data/test_mocap_1000_frames.mat", params)
    assert data.shape == (1000, 69)


def test_load_mat_w_lablel3d():
    """
    Test loading data from a .mat file w/ labels file
    """

    animal_params = utils._load_params("tests/data/test_rodent.yaml")
    assert(animal_params != None)

    data = utils.load_data("tests/data/test_mocap_1000_frames.mat", animal_params)
    assert data.shape == (1000, 69)
     



    