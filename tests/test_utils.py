from stac_mjx import utils
from pathlib import Path
import pytest

"""
Test data loaders for supported file types.
"""

_BASE_PATH = Path.cwd()


def test_load_nwb(rodent_config, mocap_nwb):
    """
    Test loading data from .nwb file.
    """
    params = utils._load_params(_BASE_PATH / rodent_config)
    assert params is not None

    data, sorted_kp_names = utils.load_data(_BASE_PATH / mocap_nwb, params)
    assert data.shape == (1000, 69)


def test_load_mat_no_label3d(rodent_config, mocap_mat):
    """
    Test loading data from .mat file.
    """
    params = utils._load_params(_BASE_PATH / rodent_config)
    assert params is not None

    data, sorted_kp_names = utils.load_data(_BASE_PATH / mocap_mat, params)
    assert data.shape == (1000, 69)


def test_load_mat_w_label3d(rodent_config_label3d, mocap_mat):
    """
    Test loading data from a .mat file w/ labels file
    """
    params = utils._load_params(_BASE_PATH / rodent_config_label3d)
    assert params is not None

    data, sorted_kp_names = utils.load_data(
        _BASE_PATH / mocap_mat,
        params,
        _BASE_PATH / params.get("KP_NAMES_LABEL3D_PATH", None),
    )
    assert data.shape == (1000, 69)


def test_load_mat_no_kp_names(rodent_config_no_kp_names, mocap_mat):
    params = utils._load_params(_BASE_PATH / rodent_config_no_kp_names)
    assert params is not None

    with pytest.raises(ValueError):
        data, sorted_kp_names = utils.load_data(
            _BASE_PATH / mocap_mat,
            params,
        )


def test_load_mat_less_kp_names(rodent_config_less_kp_names, mocap_mat):
    params = utils._load_params(_BASE_PATH / rodent_config_less_kp_names)
    assert params is not None

    with pytest.raises(ValueError):
        data, sorted_kp_names = utils.load_data(
            _BASE_PATH / mocap_mat,
            params,
        )
