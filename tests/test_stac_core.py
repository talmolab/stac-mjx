import pytest
import stac_mjx
from pathlib import Path
import stac_mjx
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

    assert isinstance(stac_core_obj, stac_mjx.stac_core.StacCore)

    assert isinstance(
        stac_core_obj.q_solver, jaxopt._src.projected_gradient.ProjectedGradient
    )

    staccore1 = stac_mjx.stac_core.StacCore(tol=config.model.FTOL)
    assert staccore1.q_solver.tol == config.model.FTOL
    assert stac_core_obj.q_solver.tol == 1e-10


def test_stac_core_compilations(TEST_DIR, PROJECT_DIR):
    config = stac_mjx.load_configs(TEST_DIR / "configs")
    stac_mjx.enable_xla_flags()

    assert stac_mjx.stac_core._q_opt._cache_size() == 0
    assert stac_mjx.stac_core._m_opt._cache_size() == 0

    kp_data, sorted_kp_names = stac_mjx.load_data(config, base_path=PROJECT_DIR)
