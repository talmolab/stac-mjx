"""Implementation of stac for animal motion capture in dm_control suite."""
import mujoco
from mujoco import mjx
import numpy as np
from typing import List, Dict, Text, Union, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jaxopt import ScipyBoundedMinimize, LevenbergMarquardt, ScipyMinimize, LBFGSB
import utils
from functools import partial

class _TestNoneArgs(BaseException):
    """Simple base exception"""

    pass

def get_site_xpos(mjx_data):
    """Returns MjxData.site_xpos of keypoint body sites

    Args:
        mjx_data (_type_): _description_
        site_index_map (_type_): _description_

    Returns:
        jax.Array: _description_
    """
    return jnp.array([mjx_data.site_xpos[i] for i in utils.params["site_index_map"].values()])

def get_site_pos(mjx_model):
    """Gets MjxModel.site_pos of keypoint body sites

    Args:
        mjx_data (_type_): _description_
        site_index_map (_type_): _description_

    Returns:
        jax.Array: _description_
    """
    return jnp.array([mjx_model.site_pos[i] for i in utils.params["site_index_map"].values()])

# Gives error when getting indices array: ValueError: setting an array element with a sequence.
def set_site_pos(mjx_model, offsets):
    """Sets MjxModel.sites_pos to offsets and returns the new mjx_model

    Args:
        mjx_data (_type_): _description_
        site_index_map (_type_): _description_

    Returns:
        _type_: _description_
    """
    indices = np.fromiter(utils.params["site_index_map"].values(), dtype=int)
    new_site_pos = jnp.put(mjx_model.site_pos, indices, offsets, inplace=False)
    mjx_model = mjx_model.replace(site_pos=new_site_pos)
    return mjx_model

def q_loss(
    q: jnp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
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
        q_copy = q_copy.at[qs_to_opt].set(q.copy())
        q = jnp.copy(q_copy)

    # pred = q_joints_to_markers(q, mjx_model, mjx_data)
    residual = kp_data - q_joints_to_markers(q, mjx_model, mjx_data)
    if kps_to_opt is not None:
        residual = residual[kps_to_opt]
        
    # residual = jnp.sum(residual)

    return residual

def q_joints_to_markers(q: jnp.ndarray, mjx_model, mjx_data) -> jnp.ndarray:
    """Convert site information to marker information.

    Args:
        q (jnp.ndarray): Postural state
        env (TYPE): env of current environment
        sites (jnp.ndarray): Sites of keypoint data.

    Returns:
        jnp.ndarray: Array of marker positions.
    """
    mjx_data = mjx_data.replace(qpos=q.copy())
    # Forward kinematics
    mjx_data = mjx.forward(mjx_model, mjx_data)

    return get_site_xpos(mjx_data).flatten()

def q_phase(
    mjx_model,
    mjx_data,
    marker_ref_arr: jnp.ndarray,
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
    lb = jnp.concatenate([-jnp.inf * jnp.ones(7), utils.params['mj_model'].jnt_range[1:][:, 0]])
    lb = jnp.minimum(lb, 0.0)
    ub = jnp.concatenate([jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 1]])
    # Define initial position of the optimization
    # q0 = jnp.copy(mujoco.MjData(utils.params["mj_model"]).qpos)

    q0 = jnp.copy(mjx_data.qpos[:])
    # print(q0, mjx_data.qpos)
    q_copy = jnp.copy(q0)

    # Set the center to help with finding the optima (does not need to be exact)
    
    q0 = jax.lax.cond(
        root_only | trunk_only,  # Condition for execution
        lambda _: q0.at[:3].set(marker_ref_arr[12:15]),  # True branch
        lambda _: q0,  # False branch (return q0 unchanged)
        operand=None  # Pass q0 as an operand to both branches
    )
    
    # if root_only or trunk_only:
    #     q0 = q0.at[:3].set(marker_ref_arr[12:15])
    #     print(q0)
    #     diff_step = utils.params["ROOT_DIFF_STEP"]
    # else:
    #     diff_step = utils.params["DIFF_STEP"]
    if root_only:
        qs_to_opt = jnp.zeros_like(q0, dtype=bool)
        qs_to_opt = qs_to_opt.at[:7].set(True)

    # Limit the optimizer to a subset of qpos
    if qs_to_opt is not None:
        q0 = q0[qs_to_opt]
        lb = lb[qs_to_opt]
        ub = ub[qs_to_opt]

    print(qs_to_opt)
    # Use different tolerances for root vs normal optimization
    if ftol is None:
        if root_only:
            ftol = utils.params["ROOT_FTOL"]
        elif qs_to_opt is not None:
            ftol = utils.params["LIMB_FTOL"]
        else:
            ftol = utils.params["FTOL"]
            
    print("q-phase:")

    try:
        loss_fn = partial(q_loss,
                            mjx_model=mjx_model,
                            mjx_data=mjx_data,
                            kp_data=marker_ref_arr.T,
                            qs_to_opt=qs_to_opt,
                            q_copy=q_copy,
                            kps_to_opt=kps_to_opt,
                            )
                            
        # Create the optimizer
        solver = LevenbergMarquardt(residual_fun=loss_fn, 
                        tol=ftol,
                        # maxls=50
                        )
        # method="l-bfgs-b",
        # jit=True,
        # diff_step=diff_step,
        # verbose=0,
        # Define the bounds
        # bounds=(lb, ub)
        # print(type(bounds), bounds)
        # state = solver.init_state(q0, bounds=bounds)
        # q_opt_param = solver.run(q0, bounds=bounds).params
        q_opt_param = solver.run(q0).params
        print(f"optimized params: {q_opt_param.x}")

        # q_opt_param = solver.run(q0, bounds=bounds).params
        # Set pose to the optimized q and step forward.
        if qs_to_opt is None:
            mjx_data = mjx_data.replace(qpos=q_opt_param.x)
        else:
            q_copy[qs_to_opt] = q_opt_param.x
            mjx_data = mjx_data.replace(qpos=q_copy.copy())

        print("last forward step")
        mjx_data = mjx.forward(mjx_model, mjx_data)

    except ValueError as ex:
        print("Warning: optimization failed.", flush=True)
        print(ex, flush=True)
        q_copy[jnp.isnan(q_copy)] = 0.0
        mjx_data.replace(qpos=q_copy.copy()) 
        mjx_data = mjx.forward(mjx_model, mjx_data)
        
    return mjx_data
    
def m_loss(
    offset: jnp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
    time_indices: jnp.ndarray,
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

    # print(len(q))
    # print(kp_data.shape)
    # print(time_indices)
    # print(offset.shape)

    def f(terms, i):
        mjx_data.qpos[:] = q[i].copy()

        # Get the offset relative to the initial position, only for
        # markers you wish to regularize
        reg_term = terms[0] + ((offset - initial_offsets.flatten()) ** 2) * is_regularized
        residual = terms[1] + (kp_data[:, i] - m_joints_to_markers(offset, mjx_model, mjx_data, sites)) ** 2
        return (reg_term, residual), None
    
    (residual, reg_term), _ = jax.lax.scan(f, (0,0), jnp.arange(time_indices.shape[0]))
        
    return jnp.sum(residual) + reg_coef * jnp.sum(reg_term)

def m_joints_to_markers(offset, mjx_model, mjx_data) -> jnp.ndarray:
    """Convert site information to marker information.

    Args:
        offset (TYPE):  Current offset.
        env (TYPE):  env of current environment
        sites (TYPE):  Sites of keypoint data.

    Returns:
        TYPE: Array of marker positions
    """
    mjx_model = set_site_pos(mjx_model, jnp.reshape(offset.copy(), (-1, 3))) 

    # Forward kinematics
    mjx_data = mjx.forward(mjx_model, mjx_data)

    return get_site_xpos(mjx_data).flatten()


def m_phase(
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
    time_indices: jnp.ndarray,
    q: jnp.ndarray,
    initial_offsets: jnp.ndarray,
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
    offset0 = jnp.copy(get_site_pos(mjx_model)).flatten()

    # Define which offsets to regularize
    is_regularized = []
    for k, v in utils.params["site_index_map"].items():
        if any(n == k for n in utils.params["SITES_TO_REGULARIZE"]):
            is_regularized.append(jnp.array([1.0, 1.0, 1.0]))
        else:
            is_regularized.append(jnp.array([0.0, 0.0, 0.0]))
    is_regularized = jnp.stack(is_regularized).flatten()

    # Optimize dm
    keypoints = kp_data[time_indices, :].T
    q = jnp.take(q, time_indices, axis=0)
    # q = [q[i] for i in time_indices]
    
    # Create the optimizer
    solver = ScipyMinimize(fun=lambda offset: m_loss(
                                            offset,
                                            mjx_model,
                                            mjx_data,
                                            keypoints,
                                            time_indices,
                                            q,
                                            initial_offsets,
                                            is_regularized=is_regularized,
                                            reg_coef=reg_coef,
                                        ),
                            method="l-bfgs-b",
                            tol=utils.params["ROOT_FTOL"],
                            maxiter=maxiter,
                            # jit=True
                            )
    
    # Run the optimization
    offset_opt_param = solver.run(offset0).params

    # Set pose to the optimized m and step forward.
    mjx_model = set_site_pos(mjx_model, jnp.reshape(offset_opt_param.x, (-1, 3))) 
    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = jit_forward(mjx_model, mjx_data)
    
    # TODO: needed??
    # for n_site, p in enumerate(env.bind(sites).pos): mjmodel.sites_pos
    #     sites[n_site].pos = p
        
    return mjx_data, mjx_model

@jit
def jit_forward(mjx_model, mjx_data):
    return mjx.forward(mjx_model, mjx_data)
    