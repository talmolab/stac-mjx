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
    kp_data, sorted_kp_names = stac_mjx.load_mocap(config, base_path)
    stac = stac_mjx.stac.Stac(arm_model, config, sorted_kp_names)

    assert sorted_kp_names == ['Shoulder', 'Elbow', 'Wrist']
    assert isinstance(stac._root, dm_control.mjcf.element.RootElement)
    assert isinstance(stac._mj_model, mujoco._structs.MjModel)

def test_stac_core_compilations(base_path, config):
    stac_mjx.enable_xla_flags()

    assert stac_mjx.stac_core._m_opt._cache_size() == 0
    assert stac_mjx.stac_core._q_opt._cache_size() == 0
    assert stac_mjx.stac_core.m_loss._cache_size() == 0

    kp_data, sorted_kp_names = stac_mjx.load_mocap(config, base_path)
    _, _ = stac_mjx.run_stac(
        config,
        kp_data, 
        sorted_kp_names, 
        base_path=base_path
        )

    assert stac_mjx.stac_core._m_opt._cache_size() == 2
    assert stac_mjx.stac_core._q_opt._cache_size() == 2
    assert stac_mjx.stac_core.m_loss._cache_size() == 0