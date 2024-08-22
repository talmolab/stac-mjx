from pathlib import Path

from stac_mjx import main
from stac_mjx import utils
from stac_mjx.controller import STAC
from mujoco import _structs

_BASE_PATH = Path.cwd()


def test_init_stac(mocap_nwb, stac_config, rodent_config):
    stac_cfg, model_cfg = main.load_configs(
        _BASE_PATH / stac_config, _BASE_PATH / rodent_config
    )

    kp_data, sorted_kp_names = utils.load_data(_BASE_PATH / mocap_nwb, model_cfg)

    xml_path = _BASE_PATH / model_cfg["MJCF_PATH"]

    stac = STAC(xml_path, stac_cfg, model_cfg, sorted_kp_names)

    assert stac.stac_cfg == stac_cfg
    assert stac.model_cfg == model_cfg
    assert stac._kp_names == sorted_kp_names
    assert isinstance(stac._mj_model, _structs.MjModel)
