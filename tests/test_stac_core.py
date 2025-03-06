import pytest
import stac_mjx
from dm_control import mjcf
import dm_control
from pathlib import Path
import mujoco
from stac_mjx.utils import mjx_load
import stac_mjx
# import jax


TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent

@pytest.fixture
def arm_model():
    return str(PROJECT_ROOT / "stac-mjx" / "models" / "arm_model_v3_torque.xml")

@pytest.fixture
def base_path():
    return Path("/root/vast/joshua/stac-mjx")

@pytest.fixture
def config(base_path):
    return stac_mjx.load_configs(base_path / "configs")

def test_stac_obj(arm_model, base_path, config):

    # get cfg, kp_data, and sorted_kp_names:
    cfg = stac_mjx.load_configs(base_path / "configs")
    kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
    stac = stac_mjx.stac.Stac(arm_model, cfg, sorted_kp_names)

    assert sorted_kp_names == ['Shoulder', 'Elbow', 'Wrist']
    assert stac.stac_core_obj is None
    assert isinstance(stac._root, dm_control.mjcf.element.RootElement)
    assert isinstance(stac._mj_model, mujoco._structs.MjModel)

def test_stac_core_compilations(base_path, config):
    stac_mjx.enable_xla_flags()

    kp_data, sorted_kp_names = stac_mjx.load_mocap(config, base_path)
    stac, _, _ = stac_mjx.run_stac(
        config,
        kp_data, 
        sorted_kp_names, 
        base_path=base_path
        )

    assert stac.




    # root = mjcf.from_path(arm_model)
    # model = mjcf.Physics.from_mjcf_model(root).model.ptr

    # mjx_model, mjx_data = mjx_load(model)
