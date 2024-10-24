"""This module contains utility functions for Stac."""

import os

from jax import numpy as jp
from jax import jit
from jax.lib import xla_bridge
from mujoco import mjx
from mujoco.mjx._src import smooth
import numpy as np
from stac_mjx import io


def enable_xla_flags():
    """Enables XLA Flags for faster runtime on Nvidia GPUs."""
    if xla_bridge.get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
        )


def mjx_load(mj_model):
    """Load mujoco model into mjx."""
    # Create mjx model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    return mjx_model, mjx_data


@jit
def kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """Jit compiled forward kinematics.

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.kinematics(mjx_model, mjx_data)


@jit
def com_pos(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """Jit compiled com_pos calculation.

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.com_pos(mjx_model, mjx_data)


def get_site_xpos(mjx_data: mjx.Data, site_idxs: jp.ndarray):
    """Get MjxData.site_xpos of keypoint body sites.

    Args:
        mjx_data (mjx.Data):

    Returns:
        jax.Array: MjxData.site_xpos of keypoint body sites, ie
        Cartesian coords of body sites.
    """
    return mjx_data.site_xpos[site_idxs]


def get_site_pos(mjx_model: mjx.Model, site_idxs: jp.ndarray):
    """Get MjxModel.site_pos of keypoint body sites.

    Args:
        mjx_data (mjx.Data):

    Returns:
        jax.Array: MjxModel.site_pos of keypoint body sites, ie
        local position offset rel. to body.
    """
    return mjx_model.site_pos[site_idxs]


def set_site_pos(mjx_model: mjx.Model, offsets, site_idxs: jp.ndarray):
    """Set MjxModel.sites_pos to offsets and returns the new mjx_model.

    Args:
        mjx_model (mjx.Model):
        offsets (jax.Array):

    Returns:
        mjx_model: Resulting mjx.Model
    """
    new_site_pos = mjx_model.site_pos.at[site_idxs].set(offsets)
    mjx_model = mjx_model.replace(site_pos=new_site_pos)
    return mjx_model


def make_qs(q0, qs_to_opt, q):
    """Create new set of qs combining initial and new qs for part optimization based on qs_to_opt.

    Args:
        q0 (jax.Array): initial joint angles
        qs_to_opt (jax.Array): joint angles that were optimized
        q (jax.Array): new joint angles

    Returns:
        jp.Array: resulting set of joint angles
    """
    return jp.copy((1 - qs_to_opt) * q0 + qs_to_opt * jp.copy(q))


def replace_qs(mjx_model: mjx.Model, mjx_data: mjx.Data, q):
    """Replace joint angles in mjx.Data with new ones and performs forward kinematics.

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):
        q (jax.Array): new joint angles

    Returns:
        mjx.Data: resulting mjx Data
    """
    if q is None:
        print("optimization failed, continuing")

    else:
        mjx_data = mjx_data.replace(qpos=q)
        mjx_data = kinematics(mjx_model, mjx_data)

    return mjx_data
