from omegaconf import OmegaConf, DictConfig
from stac_mjx import main
from stac_mjx import utils
from pathlib import Path
from typing import Dict
import pytest

_BASE_PATH = Path.cwd()


def test_load_configs(stac_config, rodent_config):
    # Check that utils.params is not defined before loading
    with pytest.raises(AttributeError):
        utils.params

    # Call the function
    cfg = main.load_configs(_BASE_PATH / stac_config, _BASE_PATH / rodent_config)

    # Assert that the configs are the correct type
    assert isinstance(cfg, DictConfig)
    assert isinstance(utils.params, Dict)

    # Assert that the resulting configs contain the expected data
    assert cfg.paths.fit_path == "fit.p"
    assert utils.params["N_FRAMES_PER_CLIP"] == 360
