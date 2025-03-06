import pytest
import stac_mjx
from dm_control import mjcf
import dm_control
from pathlib import Path
import mujoco
from stac_mjx.utils import mjx_load
import stac_mjx
import optax
import jaxopt
# import jax


# TEST_DIR = Path(__file__).parent
# PROJECT_ROOT = TEST_DIR.parent.parent


@pytest.fixture
def base_path():
    return Path("/root/vast/joshua/stac-mjx")

@pytest.fixture
def config(base_path):
    return stac_mjx.load_configs(base_path / "configs")


def test_stac_core_obj(config):

    stac_core_obj = stac_mjx.stac_core.StacCore(1e-10)

    # ensure that object type is what it is
    assert isinstance(stac_core_obj, stac_mjx.stac_core.StacCore)

    # assert the object instance variables
    assert isinstance(stac_core_obj.opt, optax._src.base.GradientTransformationExtraArgs)
    assert isinstance(stac_core_obj.q_solver, jaxopt._src.projected_gradient.ProjectedGradient)
    assert isinstance(stac_core_obj.m_solver, jaxopt._src.optax_wrapper.OptaxSolver)

    # assert the tolerance values are correct
    staccore1 = stac_mjx.stac_core.StacCore(tol=config.model.FTOL)
    assert staccore1.q_solver.tol == config.model.FTOL
    assert stac_core_obj.q_solver.tol == 1e-10


def test_stac_core_compilations(base_path, config):
    stac_mjx.enable_xla_flags()

    kp_data, sorted_kp_names = stac_mjx.load_mocap(config, base_path)
    
    assert stac_mjx.stac_core.m_opt._cache_size() == 0
    assert stac_mjx.stac_core.q_opt._cache_size() == 0

    _, _ = stac_mjx.run_stac(
        config,
        kp_data, 
        sorted_kp_names, 
        base_path=base_path
        )

    assert stac_mjx.stac_core.m_loss._cache_size() == 0
    assert stac_mjx.stac_core.q_opt._cache_size() == 2
    assert stac_mjx.stac_core.m_opt._cache_size() == 2