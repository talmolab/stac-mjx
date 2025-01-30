from stac_mjx import io
from pathlib import Path
import pytest
import hydra

"""
Test data loaders for supported file types.
"""


def load_config_with_overrides(
    config_dir, stac_data_path_override=None, model_override=None
):
    overrides = []
    if stac_data_path_override:
        overrides.append(f"stac.data_path={stac_data_path_override}")
    if model_override:
        overrides.append(f"model={model_override}")

    # Initialize Hydra and set the config path
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose the configuration by specifying the config name
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def test_load_nwb(config, mocap_nwb):
    """
    Test loading data from .nwb file.
    """
    # params = utils._load_params(_BASE_PATH / rodent_config)
    # assert params is not None
    cfg = load_config_with_overrides(config, stac_data_path_override=mocap_nwb)
    data, sorted_kp_names = io.load_mocap(cfg)
    assert data.shape == (1000, 69)
    assert len(sorted_kp_names) == 23


def test_load_mat_no_label3d(config, mocap_mat):
    """
    Test loading data from .mat file.
    """
    cfg = load_config_with_overrides(config, stac_data_path_override=mocap_mat)
    data, sorted_kp_names = io.load_mocap(cfg)
    assert data.shape == (1000, 69)
    assert len(sorted_kp_names) == 23


def test_load_mat_w_label3d(config, rodent_config_label3d, mocap_mat):
    """
    Test loading data from a .mat file w/ labels file
    """
    cfg = load_config_with_overrides(
        config, stac_data_path_override=mocap_mat, model_override=rodent_config_label3d
    )
    data, sorted_kp_names = io.load_mocap(cfg)
    assert data.shape == (1000, 69)
    assert len(sorted_kp_names) == 23


def test_load_mat_no_kp_names(config, rodent_config_no_kp_names, mocap_mat):
    cfg = load_config_with_overrides(
        config,
        stac_data_path_override=mocap_mat,
        model_override=rodent_config_no_kp_names,
    )

    with pytest.raises(ValueError):
        data, sorted_kp_names = io.load_mocap(cfg)


def test_load_mat_less_kp_names(config, rodent_config_less_kp_names, mocap_mat):
    cfg = load_config_with_overrides(
        config,
        stac_data_path_override=mocap_mat,
        model_override=rodent_config_less_kp_names,
    )

    with pytest.raises(ValueError):
        data, sorted_kp_names = io.load_mocap(cfg)


def test_load_h5(config, mouse_config, mocap_h5):
    """
    Test loading data from a .h5 file
    """

    cfg = load_config_with_overrides(
        config,
        stac_data_path_override=mocap_h5,
        model_override=mouse_config,
    )

    data, sorted_kp_names = io.load_mocap(cfg)

    assert data.shape == (3600, 102)
    assert len(sorted_kp_names) == 34
