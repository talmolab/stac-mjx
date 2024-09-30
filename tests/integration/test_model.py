import pytest

from dm_control import mjcf
from mujoco import mjx
from stac_mjx.operations import mjx_load


def load_model(path: str):
    root = mjcf.from_path(path)
    model = mjcf.Physics.from_mjcf_model(root).model.ptr

    mjx_model, mjx_data = mjx_load(model)

    assert mjx_model is not None
    assert mjx_data is not None


# Fixtures
@pytest.fixture
def rodent_model():
    return "models/rodent.xml"


@pytest.fixture()
def mouse_model():
    return "models/mouse_with_meshes.xml"


# Tests
def test_rodent_load(rodent_model):
    load_model(rodent_model)


def test_mouse_load(mouse_model):
    load_model(mouse_model)
