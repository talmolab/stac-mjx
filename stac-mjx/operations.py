from jax import numpy as jnp
from jax import jit
from mujoco.mjx._src import smooth
import numpy as np
import utils


@jit
def kinematics(mjx_model, mjx_data):
    return smooth.kinematics(mjx_model, mjx_data)


@jit
def com_pos(mjx_model, mjx_data):
    return smooth.com_pos(mjx_model, mjx_data)


def get_site_xpos(mjx_data):
    """Returns MjxData.site_xpos of keypoint body sites

    Args:
        mjx_data (_type_): _description_
        site_index_map (_type_): _description_

    Returns:
        jax.Array: _description_
    """
    return mjx_data.site_xpos[jnp.array(list(utils.params["site_index_map"].values()))]


def get_site_pos(mjx_model):
    """Gets MjxModel.site_pos of keypoint body sites

    Args:
        mjx_data (_type_): _description_
        site_index_map (_type_): _description_

    Returns:
        jax.Array: _description_
    """
    return mjx_model.site_pos[jnp.array(list(utils.params["site_index_map"].values()))]


def set_site_pos(mjx_model, offsets):
    """Sets MjxModel.sites_pos to offsets and returns the new mjx_model

    Args:
        mjx_data (_type_): _description_
        site_index_map (_type_): _description_

    Returns:
        _type_: _description_
    """
    indices = np.fromiter(utils.params["site_index_map"].values(), dtype=int)
    new_site_pos = mjx_model.site_pos.at[indices].set(offsets)
    mjx_model = mjx_model.replace(site_pos=new_site_pos)
    return mjx_model


def make_qs(q0, qs_to_opt, q):
    """Creates new set of qs combining initial and new qs for part optimization based on qs_to_opt

    Args:
        q0 (_type_): _description_
        qs_to_opt (_type_): _description_
        q (_type_): _description_

    Returns:
        jnp.Array: _description_
    """
    return jnp.copy((1 - qs_to_opt) * q0 + qs_to_opt * jnp.copy(q))


def replace_qs(mjx_model, mjx_data, q_opt_param):
    if q_opt_param is None:
        print("optimization failed, continuing")

    else:
        mjx_data = mjx_data.replace(qpos=q_opt_param)
        mjx_data = kinematics(mjx_model, mjx_data)

    return mjx_data
