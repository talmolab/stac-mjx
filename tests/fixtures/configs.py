import pytest
from unittest.mock import Mock
from omegaconf import OmegaConf
from stac_mjx import utils


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
