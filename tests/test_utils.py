from stac_mjx import utils
import os

"""
Test data loaders for supported file types.
"""


def test_load_nwb(rodent_config, mocap_mat):
    """
    Test loading data from .nwb file.
    """
    params = utils._load_params(rodent_config)
    assert params is not None

    data = utils.load_data(mocap_mat, params)
    assert data.shape == (1000, 69)


def test_load_mat_no_lablel3d(rodent_config, mocap_mat):
    """
    Test loading data from .mat file.
    """
    params = utils._load_params(rodent_config)
    assert params is not None

    data = utils.load_data(mocap_mat, params)
    assert data.shape == (1000, 69)


def test_load_mat_w_lablel3d(rodent_config_label3d, mocap_mat):
    """
    Test loading data from a .mat file w/ labels file
    """
    params = utils._load_params(rodent_config_label3d)
    assert params is not None

    data = utils.load_data(mocap_mat, params)
    assert data.shape == (1000, 69)
