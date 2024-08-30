"""Implementation of stac for animal motion capture in dm_control suite."""

import jax
import jax.numpy as jp
from jax import jit

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box
from jaxopt import OptaxSolver

import optax

from stac_mjx import operations as op
from stac_mjx import utils
import functools


def huber(x, delta=5.0, max=10, max_slope=0.1):
    """Compute the Huber loss + sum."""
    x = jp.where(jp.abs(x) < delta, 0.5 * x**2, delta * (jp.abs(x) - 0.5 * delta))
    x = jp.where(x > max, (x - max) * max_slope + max, x)
    return jp.sum(x)


def squared_error(x):
    """Compute the squared error + sum."""
    return jp.sum(jp.square(x))


@jit
def q_opt(
    mjx_model,
    mjx_data,
    marker_ref_arr: jp.ndarray,
    qs_to_opt: jp.ndarray,
    kps_to_opt: jp.ndarray,
    q0: jp.ndarray,
    lb,
    ub,
    site_idxs,
):
    """Update q_pose using estimated marker parameters."""
    try:
        return mjx_data, q_solver.run(
            q0,
            hyperparams_proj=jp.array((lb, ub)),
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            kp_data=marker_ref_arr.T,
            qs_to_opt=qs_to_opt,
            kps_to_opt=kps_to_opt,
            initial_q=q0,
            site_idxs=site_idxs,
        )

    except ValueError as ex:
        print("Warning: optimization failed.", flush=True)
        print(ex, flush=True)
        mjx_data = mjx_data.replace(qpos=q0)
        mjx_data = op.kinematics(mjx_model, mjx_data)

    return mjx_data, None


@jit
def m_loss(
    offsets: jp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    q: jp.ndarray,
    initial_offsets: jp.ndarray,
    site_idxs: jp.ndarray,
    is_regularized: bool = None,
    reg_coef: float = 0.0,
) -> jp.array:
    # fmt: off
    """Compute the marker residual for optimization.

    Args:
        offsets (jp.ndarray): vector of offsets to inferred mocap sites
        mjx_model (_type_): MJX Model
        mjx_data (_type_):  MJX Data
        kp_data (jp.ndarray):  Mocap data in global coordinates
        q (jp.ndarray): proposed qpos values
        initial_offsets (jp.ndarray): Initial offset values for offset regularization
        is_regularized (bool, optional): binary vector of offsets to regularize.. Defaults to None.
        reg_coef (float, optional):  L1 regularization coefficient during marker loss.. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    # fmt: on
    def f(carry, input):
        # Unpack arguments
        qpos, kp = input
        mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized = carry

        # Get the offset relative to the initial position, only for markers you wish to regularize
        reg_term = (
            reg_term + (jp.square(offsets - initial_offsets.flatten())) * is_regularized
        )

        # Set qpos and offsets
        mjx_data = mjx_data.replace(qpos=qpos)
        mjx_model = op.set_site_pos(mjx_model, jp.reshape(offsets, (-1, 3)), site_idxs)

        # Forward kinematics
        mjx_data = op.kinematics(mjx_model, mjx_data)
        mjx_data = op.com_pos(mjx_model, mjx_data)
        markers = op.get_site_xpos(mjx_data, site_idxs).flatten()

        # Accumulate squared residual
        residual = residual + jp.square((kp - markers))
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
                jp.zeros(69),
                jp.zeros(69),
                initial_offsets,
                is_regularized,
            ),
            (q, kp_data),
        )
    )
    return jp.sum(residual) + reg_coef * jp.sum(reg_term)


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
    site_idxs,
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
        site_idxs=site_idxs,
    )

    return res


# TODO: put these values in config, move to just optax by implementing solver loop
opt = optax.sgd(learning_rate=5e-4, momentum=0.9, nesterov=False)

# q_solver = ProjectedGradient(fun=q_loss, projection=projection_box, maxiter=250)
m_solver = OptaxSolver(opt=opt, fun=m_loss, maxiter=2000)


def q_loss(
    q: jp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    qs_to_opt: jp.ndarray,
    kps_to_opt: jp.ndarray,
    initial_q: jp.ndarray,
    site_idxs: jp.ndarray,
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (jp.ndarray): Proposed qs
        mjx_model (mjx.Model): Model object (stays constant)
        mjx_data (mjx.Data): Data object (modified to calculate new xpos)
        kp_data (jp.ndarray): Ground truth keypoint positions
        qs_to_opt (jp.ndarray): Boolean array; for each index in qpos, True = q and False = initial_q when calculating residual
        kps_to_opt (jp.ndarray): Boolean array; only return residuals for the True positions
        initial_q (jp.ndarray): Starting qs for reference

    Returns:
        float: sum of squares scalar loss
    """
    # Replace qpos with new qpos with q and initial_q, based on qs_to_opt
    mjx_data = mjx_data.replace(qpos=op.make_qs(initial_q, qs_to_opt, q))

    # Forward kinematics
    mjx_data = op.kinematics(mjx_model, mjx_data)
    mjx_data = op.com_pos(mjx_model, mjx_data)

    # Get marker site xpos
    markers = op.get_site_xpos(mjx_data, site_idxs).flatten()
    residual = kp_data - markers

    # Set irrelevant body sites to 0
    residual = residual * kps_to_opt
    residual = squared_error(residual)

    return residual


@jit
def q_opt_NEW(
    mjx_model,
    mjx_data,
    kp_data,
    qs_to_opt,
    kps_to_opt,
    q0,
    lb,
    ub,
    site_idxs,
    learning_rate=3e-4,
    num_iterations=1000,
    tol=1e-5,
):
    # Define the Adam optimizer
    optimizer = optax.adam(learning_rate)

    # Initialize the optimizer state
    opt_state = optimizer.init(q0)

    # Define the loss function and its gradient
    @jax.jit
    def loss_and_grad(params):
        return jax.value_and_grad(
            functools.partial(
                q_loss,
                mjx_model=mjx_model,
                mjx_data=mjx_data,
                kp_data=kp_data.T,
                qs_to_opt=qs_to_opt,
                kps_to_opt=kps_to_opt,
                initial_q=q0,
                site_idxs=site_idxs,
            )
        )(params)

    # Define the update step
    @jax.jit
    def update(carry):
        params, opt_state, cur_loss, prev_loss, iteration = carry
        loss, grads = loss_and_grad(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        # Project the parameters back into the feasible region
        new_params = optax.projections.projection_box(new_params, lb, ub)
        return (new_params, new_opt_state, loss, cur_loss, iteration + 1)

    # Define the condition function for while_loop
    @jax.jit
    def cond_fun(carry):
        params, opt_state, cur_loss, prev_loss, iteration = carry
        return jp.logical_and(
            iteration < num_iterations, jp.abs(cur_loss - prev_loss) >= tol
        )

    # Initialize the carry for the while_loop
    init_carry = (q0, opt_state, jp.inf, -jp.inf, 0)

    # Run the optimization using while_loop
    final_carry = jax.lax.while_loop(
        cond_fun, update, init_carry  # Only pass along the carry
    )

    final_params, opt_state, final_loss, _, num_iters = final_carry

    return final_params, final_loss, num_iters
