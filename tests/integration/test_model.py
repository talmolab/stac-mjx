from pathlib import Path
import pytest

import mujoco
from stac_mjx.utils import mjx_load

# Define path roots
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent


# Common model loading/testing code (Feel free to add more!)
def load_model(path: str):
    spec = mujoco.MjSpec.from_file(str(path))
    model = spec.compile()

    mjx_model, mjx_data = mjx_load(model)

    assert mjx_model is not None
    assert mjx_data is not None
    assert mjx_model.nq > 0, "Model should have degrees of freedom"
    assert mjx_model.nv > 0, "Model should have velocities"
    assert len(mjx_data.qpos) == mjx_model.nq, "Data should match model dimensions"


# Fixtures
@pytest.fixture
def rodent_model():
    return str(PROJECT_ROOT / "models" / "rodent.xml")


@pytest.fixture
def mouse_model():
    return str(PROJECT_ROOT / "models" / "mouse_with_meshes.xml")


# Tests
def test_rodent_load(rodent_model):
    load_model(rodent_model)


def test_mouse_load(mouse_model):
    load_model(mouse_model)
