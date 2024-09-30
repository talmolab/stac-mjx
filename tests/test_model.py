# TODO check which ones are required
from dm_control import mjcf

# import mujoco
from mujoco import mjx


def load_model(path: str):
    root = mjcf.from_path(path)
    physics = mjcf.Physics.from_mjcf_model(root)

    # Create mjx model and data
    mjx_model = mjx.put_model(physics.model.ptr)
    assert mjx_model is not None

    mjx_data = mjx.make_data(mjx_model)
    assert mjx_data is not None


def test_rodent_load():
    load_model("models/rodent.xml")


def test_mouse_load():
    load_model("models/mouse_with_meshes.xml")
