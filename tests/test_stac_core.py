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


@pytest.fixture
def TEST_DIR():
    return Path(__file__).parent


@pytest.fixture
def PROJECT_DIR():
    return Path(__file__).parent.parent


def test_stac_core_obj(TEST_DIR):

    config = stac_mjx.load_configs(TEST_DIR / "configs")
    stac_core_obj = stac_mjx.stac_core.StacCore(1e-10)

    # ensure that object type is what it is
    assert isinstance(stac_core_obj, stac_mjx.stac_core.StacCore)

    # assert the object instance variables
    assert isinstance(
        stac_core_obj.opt, optax._src.base.GradientTransformationExtraArgs
    )
    assert isinstance(
        stac_core_obj.q_tol, jaxopt._src.projected_gradient.ProjectedGradient
    )
    assert isinstance(stac_core_obj.m_solver, jaxopt._src.optax_wrapper.OptaxSolver)

    # assert the tolerance values are correct
    staccore1 = stac_mjx.stac_core.StacCore(tol=config.model.FTOL)
    assert staccore1.q_tol.tol == config.model.FTOL
    assert stac_core_obj.q_tol.tol == 1e-10


def test_stac_core_compilations(TEST_DIR, PROJECT_DIR):
    # tests cache sizes of jit compiled functions
    # run time is long without gpu so commented out
    # tests after run_stac

    config = stac_mjx.load_configs(TEST_DIR / "configs")
    stac_mjx.enable_xla_flags()

    assert stac_mjx.stac_core.m_loss._cache_size() == 0
    assert stac_mjx.stac_core._m_opt._cache_size() == 0
    assert stac_mjx.stac_core._q_opt._cache_size() == 0

    kp_data, sorted_kp_names = stac_mjx.load_mocap(config, base_path=PROJECT_DIR)
    # _, _ = stac_mjx.run_stac(config, kp_data, sorted_kp_names, base_path=PROJECT_DIR)

    # assert stac_mjx.stac_core.m_loss._cache_size() == 0
    # assert stac_mjx.stac_core._q_opt._cache_size() == 2
    # assert stac_mjx.stac_core._m_opt._cache_size() == 2
