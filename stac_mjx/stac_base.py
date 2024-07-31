"""Implementation of stac for animal motion capture in dm_control suite."""

from typing import List, Dict, Text, Union, Tuple
import jax
import jax.numpy as jnp
from jax import jit

from jaxopt import LBFGSB, LBFGS, ProjectedGradient
from jaxopt.projection import projection_box
from jaxopt import OptaxSolver

import optax

import operations as op
import utils


def huber(x, delta=5.0, max=10, max_slope=0.1):
    """Compute the Huber loss + sum."""
    x = jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))
    x = jnp.where(x > max, (x - max) * max_slope + max, x)
    return jnp.sum(x)


def squared_error(x):
    """Compute the squared error + sum."""
    return jnp.sum(jnp.square(x))


def q_loss(
    q: jnp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
    qs_to_opt: jnp.ndarray,
    kps_to_opt: jnp.ndarray,
    initial_q: jnp.ndarray,
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (jnp.ndarray): Proposed qs
        mjx_model (mjx.Model): Model object (stays constant)
        mjx_data (mjx.Data): Data object (modified to calculate new xpos)
        kp_data (jnp.ndarray): Ground truth keypoint positions
        qs_to_opt (jnp.ndarray): Boolean array; for each index in qpos, True = q and False = initial_q when calculating residual
        kps_to_opt (jnp.ndarray): Boolean array; only return residuals for the True positions
        initial_q (jnp.ndarray): Starting qs for reference

    Returns:
        float: sum of squares scalar loss
    """
    # Replace qpos with new qpos with q and initial_q, based on qs_to_opt
    mjx_data = mjx_data.replace(qpos=op.make_qs(initial_q, qs_to_opt, q))

    # Clip to bounds ourselves because of potential jaxopt bug
    # mjx_data = mjx_data.replace(qpos=jnp.clip(op.make_qs(initial_q, qs_to_opt, q), utils.params['lb'], utils.params['ub']))

    # Forward kinematics
    mjx_data = op.kinematics(mjx_model, mjx_data)
    mjx_data = op.com_pos(mjx_model, mjx_data)

    # Get marker site xpos
    markers = op.get_site_xpos(mjx_data).flatten()
    residual = kp_data - markers

    # Set irrelevant body sites to 0
    residual = residual * kps_to_opt
    residual = squared_error(residual)

    return residual


@jit
def q_opt(
    mjx_model,
    mjx_data,
    marker_ref_arr: jnp.ndarray,
    qs_to_opt: jnp.ndarray,
    kps_to_opt: jnp.ndarray,
    # maxiter: int,
    q0: jnp.ndarray,
    ftol: float,
):
    """Update q_pose using estimated marker parameters."""
    lb = utils.params["lb"]
    ub = utils.params["ub"]
    try:

        return mjx_data, q_solver.run(
            q0,
            hyperparams_proj=jnp.array((lb, ub)),
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            kp_data=marker_ref_arr.T,
            qs_to_opt=qs_to_opt,
            kps_to_opt=kps_to_opt,
            initial_q=q0,
        )

    except ValueError as ex:
        print("Warning: optimization failed.", flush=True)
        print(ex, flush=True)
        mjx_data = mjx_data.replace(qpos=q0)
        mjx_data = op.kinematics(mjx_model, mjx_data)

    return mjx_data, None


@jit
def m_loss(
    offsets: jnp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
    q: jnp.ndarray,
    initial_offsets: jnp.ndarray,
    is_regularized: bool = None,
    reg_coef: float = 0.0,
):
    # fmt: off
    # Black and pydoc conflict, so turn Black off for this doc string.
    """Compute the marker loss for optimization.

    Args:
        offsets (jnp.ndarray): vector of offsets to inferred mocap sites
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        env (TYPE): env of current environment.
        kp_data (jnp.ndarray): Mocap data in global coordinates
        time_indices (List): time_indices used for offset estimation
        sites (jnp.ndarray): sites of keypoints at frame_index
        q (jnp.ndarray): qpos values for the frames in time_indices
        initial_offsets (jnp.ndarray): Initial offset values for offset regularization
        is_regularized (bool, optional): binary vector of offsets to regularize.
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
    """
    # fmt: on
    def f(carry, input):
        # Unpack arguments
        qpos, kp = input
        mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized = carry

        # Get the offset relative to the initial position, only for markers you wish to regularize
        reg_term = (
            reg_term
            + (jnp.square(offsets - initial_offsets.flatten())) * is_regularized
        )

        # Set qpos and offsets
        mjx_data = mjx_data.replace(qpos=qpos)
        mjx_model = op.set_site_pos(mjx_model, jnp.reshape(offsets, (-1, 3)))

        # Forward kinematics
        mjx_data = op.kinematics(mjx_model, mjx_data)
        mjx_data = op.com_pos(mjx_model, mjx_data)
        markers = op.get_site_xpos(mjx_data).flatten()

        # Accumulate squared residual
        residual = residual + jnp.square((kp - markers))
        return (
            mjx_model,
            mjx_data,
            reg_term,
            residual,
            initial_offsets,
            is_regularized,
        ), None

    (mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized), _ = (
        jax.lax.scan(
            f,
            (
                mjx_model,
                mjx_data,
                jnp.zeros(69),
                jnp.zeros(69),
                initial_offsets,
                is_regularized,
            ),
            (q, kp_data),
        )
    )
    return jnp.sum(residual) + reg_coef * jnp.sum(reg_term)


@jit
def m_opt(
    offset0,
    mjx_model,
    mjx_data,
    keypoints,
    q,
    initial_offsets,
    is_regularized,
    reg_coef,
    ftol,
):
    """Compute phase optimization.

    Args:
        offset0 (_type_): _description_
        mjx_model (_type_): _description_
        mjx_data (_type_): _description_
        keypoints (_type_): _description_
        q (_type_): _description_
        initial_offsets (_type_): _description_
        is_regularized (bool): _description_
        reg_coef (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = m_solver.run(
        offset0,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        kp_data=keypoints,
        q=q,
        initial_offsets=initial_offsets,
        is_regularized=is_regularized,
        reg_coef=reg_coef,
    )

    return res


# TODO: put these values in config
opt = optax.sgd(learning_rate=5e-4, momentum=0.9, nesterov=False)

q_solver = ProjectedGradient(fun=q_loss, projection=projection_box, maxiter=250)
m_solver = OptaxSolver(opt=opt, fun=m_loss, maxiter=2000)
