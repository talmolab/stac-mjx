from omegaconf import DictConfig
from stac_mjx import main
from stac_mjx import utils
from pathlib import Path
from typing import Dict
import pytest


def test_load_configs(config):
    # Check that utils.params is not defined before loading
    with pytest.raises(AttributeError):
        utils.params

    # Call the function
    cfg = main.load_configs(config)

    # Assert that the configs are the correct type
    assert isinstance(cfg, DictConfig)

    # Assert that the resulting configs contain the expected data
    assert cfg.stac.fit_path == "fit.p"
    assert cfg.model.N_FRAMES_PER_CLIP == 360
