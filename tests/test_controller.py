from pathlib import Path

from stac_mjx import main
from stac_mjx import utils
from stac_mjx.controller import STAC
from mujoco import _structs

_BASE_PATH = Path.cwd()


def test_init_stac(mocap_nwb, config):
    cfg = main.load_configs(config)
    xml_path = _BASE_PATH / cfg.model.MJCF_PATH
    kp_data, sorted_kp_names = utils.load_data(cfg)

    stac = STAC(xml_path, cfg, sorted_kp_names)

    assert stac.cfg == cfg
    assert stac._kp_names == sorted_kp_names
    assert isinstance(stac._mj_model, _structs.MjModel)
