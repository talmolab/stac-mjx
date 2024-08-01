import pytest
from unittest.mock import Mock
from omegaconf import OmegaConf, DictConfig
from stac_mjx import utils
from stac_mjx import main


@pytest.fixture
def mock_omegaconf_load(monkeypatch):
    mock = Mock(
        side_effect=[
            OmegaConf.create({"model": "config"}),
            OmegaConf.create({"stac": "config"}),
        ]
    )
    monkeypatch.setattr(OmegaConf, "load", mock)
    return mock


@pytest.fixture
def mock_init_params(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(utils, "init_params", mock)
    return mock


def test_load_configs(mock_omegaconf_load, mock_init_params):
    # Define test input paths
    stac_config_path = "/path/to/stac.yaml"
    model_config_path = "/path/to/model.yaml"

    # Call the function
    result = main.load_configs(stac_config_path, model_config_path)

    # Assert that OmegaConf.load was called twice with correct arguments
    mock_omegaconf_load.assert_any_call(model_config_path)
    mock_omegaconf_load.assert_any_call(stac_config_path)

    # Assert that init_params was called with the correct argument
    mock_init_params.assert_called_once_with({"model": "config"})

    # Assert that the result is a DictConfig
    assert isinstance(result, DictConfig)

    # Assert that the result contains the expected data
    assert result == OmegaConf.create({"stac": "config"})
