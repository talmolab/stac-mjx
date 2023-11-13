"""Implementation of stac for animal motion capture in dm_control suite."""
import mujoco
from mujoco import mjx
import numpy as np
from typing import List, Dict, Text, Union, Tuple
import jax
import jax.numpy as jnp


class _TestNoneArgs(BaseException):
    """Simple base exception"""

    pass


def q_loss(
    q: jnp.ndarray,
    env,
    kp_data: jnp.ndarray,
    sites: jnp.ndarray,
    qs_to_opt: jnp.ndarray = None,
    q_copy: jnp.ndarray = None,
    kps_to_opt: jnp.ndarray = None,
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (jnp.ndarray): Qpos for current frame.
        env (TYPE): env of current environment.
        kp_data (jnp.ndarray): Reference keypoint data.
        sites (jnp.ndarray): sites of keypoints at frame_index
        qs_to_opt (List, optional): Binary vector of qposes to optimize.
        q_copy (jnp.ndarray, optional): Copy of current qpos, for use in optimization of subsets
        kps_to_opt (List, optional): Vector denoting which keypoints to use in loss.

    Returns:
        float: loss value
    """
    # If optimizing arbitrary sets of qpos, add the optimizer qpos to the copy.
    if qs_to_opt is not None:
        q_copy[qs_to_opt] = q.copy()
        q = jnp.copy(q_copy)

    residual = kp_data - q_joints_to_markers(q, env, sites)
    if kps_to_opt is not None:
        residual = residual[kps_to_opt]
    return residual


def q_joints_to_markers(q: jnp.ndarray, env, sites: jnp.ndarray) -> jnp.ndarray:
    """Convert site information to marker information.

    Args:
        q (jnp.ndarray): Postural state
        env (TYPE): env of current environment
        sites (jnp.ndarray): Sites of keypoint data.

    Returns:
        jnp.ndarray: Array of marker positions.
    """
    env.named.data.qpos[:] = q.copy()

    # Forward kinematics
    mjx.forward(env.model.ptr, env.data.ptr)

    # Center of mass position
    # TODO 
    mjlib.mj_comPos(env.model.ptr, env.data.ptr)

    return jnp.array(env.bind(sites).xpos).flatten()


# TODO: Refactor
def q_phase(
    env,
    marker_ref_arr: jnp.ndarray,
    sites: jnp.ndarray,
    params: Dict,
    qs_to_opt: jnp.ndarray = None,
    kps_to_opt: jnp.ndarray = None,
    root_only: bool = False,
    trunk_only: bool = False,
    ftol: float = None,
):
    """Update q_pose using estimated marker parameters.

    Args:
        env (TYPE): env of current environment.
        marker_ref_arr (jnp.ndarray): Keypoint data reference
        sites (jnp.ndarray): sites of keypoints at frame_index
        params (Dict): Animal parameters dictionary
        qs_to_opt (jnp.ndarray, optional): Description
        kps_to_opt (jnp.ndarray, optional): Description
        root_only (bool, optional): Description
        trunk_only (bool, optional): Description
        temporal_regularization (bool, optional): If True, regularize arm joints over time.
        ftol (float, optional): Description
        qs_to_opt (None, optional Binary vector of qs to optimize.
        kps_to_opt (None, optional Logical vector of keypoints to use in q loss function.
        root_only (bool, optional If True, only optimize the root.
        trunk_only (bool, optional If True, only optimize the trunk.
    """
    lb = jnp.concatenate([-jnp.inf * jnp.ones(7), env.named.model.jnt_range[1:][:, 0]])
    lb = jnp.minimum(lb, 0.0)
    ub = jnp.concatenate([jnp.inf * jnp.ones(7), env.named.model.jnt_range[1:][:, 1]])
    # Define initial position of the optimization
    q0 = jnp.copy(env.named.data.qpos[:])
    q_copy = jnp.copy(q0)

    # Set the center to help with finding the optima (does not need to be exact)
    if root_only or trunk_only:
        q0[:3] = marker_ref_arr[12:15]
        diff_step = params["ROOT_DIFF_STEP"]
    else:
        diff_step = params["DIFF_STEP"]
    if root_only:
        qs_to_opt = jnp.zeros_like(q0, dtype=bool)
        qs_to_opt[:7] = True

    # Limit the optimizer to a subset of qpos
    if qs_to_opt is not None:
        q0 = q0[qs_to_opt]
        lb = lb[qs_to_opt]
        ub = ub[qs_to_opt]

    # Use different tolerances for root vs normal optimization
    if ftol is None:
        if root_only:
            ftol = params["ROOT_FTOL"]
        elif qs_to_opt is not None:
            ftol = params["LIMB_FTOL"]
        else:
            ftol = params["FTOL"]
    try:
        q_opt_param = jax.scipy.optimize.minimize(
            lambda q: q_loss(
                q,
                env,
                marker_ref_arr.T,
                sites,
                qs_to_opt=qs_to_opt,
                q_copy=q_copy,
                kps_to_opt=kps_to_opt,
            ),
            q0,
            method="BFGS", # only method
            # bounds=(lb, ub),
            # ftol=ftol,
            # diff_step=diff_step,
            # verbose=0,
        )

        # Set pose to the optimized q and step forward.
        if qs_to_opt is None:
            env.named.data.qpos[:] = q_opt_param.x
        else:
            q_copy[qs_to_opt] = q_opt_param.x
            env.named.data.qpos[:] = q_copy.copy()

        mjx.forward(env.model.ptr, env.data.ptr)

    except ValueError:
        print("Warning: optimization failed.", flush=True)
        q_copy[jnp.isnan(q_copy)] = 0.0
        env.named.data.qpos[:] = q_copy.copy()
        mjx.forward(env.model.ptr, env.data.ptr)


def m_loss(
    offset: jnp.ndarray,
    env,
    kp_data: jnp.ndarray,
    time_indices: List,
    sites: jnp.ndarray,
    q: jnp.ndarray,
    initial_offsets: jnp.ndarray,
    is_regularized: bool = None,
    reg_coef: float = 0.0,
):
    """Compute the marker loss for optimization.

    Args:
        offset (jnp.ndarray): vector of offsets to inferred mocap sites
        env (TYPE): env of current environment.
        kp_data (jnp.ndarray): Mocap data in global coordinates
        time_indices (List): time_indices used for offset estimation
        sites (jnp.ndarray): sites of keypoints at frame_index
        q (jnp.ndarray): qpos values for the frames in time_indices
        initial_offsets (jnp.ndarray): Initial offset values for offset regularization
        is_regularized (bool, optional): binary vector of offsets to regularize.
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
    """
    residual = 0
    reg_term = 0
    # print(len(q))
    # print(kp_data.shape)
    # print(time_indices)
    # print(offset.shape)
    for i, frame in enumerate(time_indices):
        env.named.data.qpos[:] = q[i].copy()

        # Get the offset relative to the initial position, only for
        # markers you wish to regularize
        reg_term += ((offset - initial_offsets.flatten()) ** 2) * is_regularized
        residual += (kp_data[:, i] - m_joints_to_markers(offset, env, sites)) ** 2
    return jnp.sum(residual) + reg_coef * jnp.sum(reg_term)


def m_joints_to_markers(offset, env, sites) -> jnp.ndarray:
    """Convert site information to marker information.

    Args:
        offset (TYPE):  Current offset.
        env (TYPE):  env of current environment
        sites (TYPE):  Sites of keypoint data.

    Returns:
        TYPE: Array of marker positions
    """
    env.bind(sites).pos[:] = jnp.reshape(offset.copy(), (-1, 3))

    # Forward kinematics
    mjx.forward(env.model.ptr, env.data.ptr)

    # Center of mass position
    # TODO 
    mjlib.mj_comPos(env.model.ptr, env.data.ptr)

    return jnp.array(env.bind(sites).xpos).flatten()


def m_phase(
    env,
    kp_data: jnp.ndarray,
    sites: jnp.ndarray,
    time_indices: List,
    q: jnp.ndarray,
    initial_offsets: jnp.ndarray,
    params: Dict,
    reg_coef: float = 0.0,
    maxiter: int = 50,
):
    """Estimate marker offset, keeping qpos fixed.

    Args:
        env (TYPE): env of current environment
        kp_data (jnp.ndarray): Keypoint data.
        sites (jnp.ndarray): sites of keypoints at frame_index.
        time_indices (List): time_indices used for offset estimation.
        q (jnp.ndarray): qpos values for the frames in time_indices.
        initial_offsets (jnp.ndarray): Initial offset values for offset regularization.
        params (Dict): Animal parameters dictionary
        reg_coef (float, optional): L1 regularization coefficient during marker loss.
        maxiter (int, optional): Maximum number of iterations to use in the minimization.
    """
    # Define initial position of the optimization
    offset0 = jnp.copy(env.bind(sites).pos[:]).flatten()

    # Define which offsets to regularize
    is_regularized = []
    for site in sites:
        if any(n in site.name for n in params["SITES_TO_REGULARIZE"]):
            is_regularized.append(jnp.array([1.0, 1.0, 1.0]))
        else:
            is_regularized.append(jnp.array([0.0, 0.0, 0.0]))
    is_regularized = jnp.stack(is_regularized).flatten()

    # Optimize dm
    keypoints = kp_data[time_indices, :].T
    q = [q[i] for i in time_indices]
    offset_opt_param = jax.scipy.optimize.minimize(
        lambda offset: m_loss(
            offset,
            env,
            keypoints,
            time_indices,
            sites,
            q,
            initial_offsets,
            is_regularized=is_regularized,
            reg_coef=reg_coef,
        ),
        offset0,
        method="BFGS",
        # tol=params["ROOT_FTOL"],
        # options={"maxiter": maxiter},
    )

    # Set pose to the optimized m and step forward.
    env.bind(sites).pos[:] = jnp.reshape(offset_opt_param.x, (-1, 3))

    # Forward kinematics, and save the results to the walker sites as well
    mjx.forward(env.model.ptr, env.data.ptr)
    for n_site, p in enumerate(env.bind(sites).pos):
        sites[n_site].pos = p