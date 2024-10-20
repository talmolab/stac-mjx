from omegaconf import DictConfig
from stac_mjx import main
from stac_mjx import io
from pathlib import Path
from typing import Dict
import pytest


def test_load_configs(config):
    # Check that utils.params is not defined before loading
    with pytest.raises(AttributeError):
        io.params

    # Call the function
    cfg = main.load_configs(config)

    # Assert that the configs are the correct type
    assert isinstance(cfg, DictConfig)

    # Assert that the resulting configs contain the expected data
    assert cfg.stac.fit_offsets_path == "fit.p"
    assert cfg.model.N_FRAMES_PER_CLIP == 360
