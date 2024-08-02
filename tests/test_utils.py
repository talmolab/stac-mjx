from stac_mjx import utils
import os

"""
Test data loaders for supported file types.
"""


def test_load_nwb(rodent_yaml_file, mocap_mat_file):
    """
    Test loading data from .nwb file.
    """
    params = utils._load_params(rodent_yaml_file)
    data = utils.load_data(mocap_mat_file, params)
    assert data.shape == (1000, 69)


def test_load_mat_no_lablel3d(rodent_yaml_file, mocap_mat_file):
    """
    Test loading data from .mat file.
    """
    params = utils._load_params(rodent_yaml_file)
    data = utils.load_data(mocap_mat_file, params)
    assert data.shape == (1000, 69)


def test_load_mat_w_lablel3d(rodent_load3d_yaml_file, mocap_mat_file):
    """
    Test loading data from a .mat file w/ labels file
    """
    animal_params = utils._load_params(rodent_load3d_yaml_file)
    assert animal_params != None

    data = utils.load_data(mocap_mat_file, animal_params)
    assert data.shape == (1000, 69)
