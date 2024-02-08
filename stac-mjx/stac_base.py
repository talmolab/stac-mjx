"""Implementation of stac for animal motion capture in dm_control suite."""
import mujoco
from mujoco import mjx
from  mujoco.mjx._src import smooth
import numpy as np
from typing import List, Dict, Text, Union, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jaxopt import LBFGSB, LBFGS
import utils
from functools import partial
from jax.tree_util import Partial

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
    qs_to_opt: jnp.ndarray,
    kps_to_opt: jnp.ndarray,
    initial_q: jnp.ndarray
    # part_opt: bool = False
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (jnp.ndarray): Qpos for current frame.
        env (TYPE): env of current environment.
        kp_data (jnp.ndarray): Reference keypoint data.
        sites (jnp.ndarray): sites of keypoints at frame_index
        qs_to_opt (List, optional): Binary vector of qposes to optimize.
        trunk_only (bool, optional): Optimize based only on the trunk kps

    Returns:
        float: loss value
    """
    # TODO use jax.lax.cond
    # if part opt, compose a loss of the form: L = aF(p1) + bF(p2) ... + d(F(p4))
    # function F is what q_loss currently does
    
    # If optimizing arbitrary sets of qpos, add the optimizer qpos to the copy.
    # updates the relevant qpos elements to the corresponding new ones to calculate the loss
    q = jnp.copy((1 - qs_to_opt) * initial_q + qs_to_opt * q)

    mjx_data, markers = q_joints_to_markers(q, mjx_model, mjx_data)
    residual = kp_data - markers
    
    # Set irrelevant body sites to 0
    residual = residual * kps_to_opt
    residual = jnp.sum(jnp.square(residual))
    return residual

def q_joints_to_markers(q: jnp.ndarray, mjx_model, mjx_data) -> (mjx.Data, jnp.ndarray):
    """Convert site information to marker information.

    Args:
        q (jnp.ndarray): Postural state
        env (TYPE): env of current environment
        sites (jnp.ndarray): Sites of keypoint data.

    Returns:
        jnp.ndarray: Array of marker positions.
    """
    mjx_data = mjx_data.replace(qpos=q)
    # Forward kinematics
    mjx_data = kinematics(mjx_model, mjx_data)
    mjx_data = com_pos(mjx_model, mjx_data)

    return mjx_data, get_site_xpos(mjx_data).flatten()

def get_q_bounds(mjx_model):
    lb = jnp.concatenate([-jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 0]])
    lb = jnp.minimum(lb, 0.0)
    ub = jnp.concatenate([jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 1]])
    return (lb, ub)

@jit
def q_opt(
    mjx_model,
    mjx_data,
    marker_ref_arr: jnp.ndarray,
    q0,
    qs_to_opt: jnp.ndarray,
    kps_to_opt: jnp.ndarray,
    maxiter
):
    """Update q_pose using estimated marker parameters.
    """
    try:
        solver = LBFGSB(fun=q_loss, 
                        tol=utils.params["FTOL"],
                        maxiter=maxiter,
                        jit=True,
                        verbose=False
                        )
        # Define the bounds
        bounds = get_q_bounds(mjx_model)
        
        res = solver.run(q0, bounds, mjx_model=mjx_model, 
                                        mjx_data=mjx_data, 
                                        kp_data=marker_ref_arr.T,
                                        qs_to_opt=qs_to_opt,
                                        kps_to_opt=kps_to_opt,
                                        initial_q=q0)
        q_opt_param = res.params
        
        return mjx_data, q_opt_param

    except ValueError as ex:
        # print("Warning: optimization failed.", flush=True)
        # print(ex, flush=True)
        mjx_data = mjx_data.replace(qpos=q0) 
        mjx_data = kinematics(mjx_model, mjx_data)

    return mjx_data, None

@jit 
def m_loss(
    offset: jnp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
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

    @jit
    def f(carry, input):
        qpos, kp = input
        mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized = carry
        mjx_data = mjx_data.replace(qpos=qpos)
        
        # Get the offset relative to the initial position, only for
        # markers you wish to regularize
        reg_term = reg_term + (jnp.square(offset - initial_offsets.flatten())) * is_regularized
        
        mjx_data, markers = m_joints_to_markers(offset, mjx_model, mjx_data)
        residual = (residual + jnp.square((kp - markers)))
        return (mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized), None
    
    (mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized), _ = jax.lax.scan(
            f, 
            (mjx_model, mjx_data, jnp.zeros(69), jnp.zeros(69), initial_offsets, is_regularized), 
            (q, kp_data)
        )
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
    mjx_model = set_site_pos(mjx_model, jnp.reshape(offset, (-1, 3))) 

    # Forward kinematics
    mjx_data = kinematics(mjx_model, mjx_data)
    mjx_data = com_pos(mjx_model, mjx_data)

    return mjx_data, get_site_xpos(mjx_data).flatten()


@jit
def m_opt(offset0, 
          mjx_model, 
          mjx_data, 
          keypoints, 
          q, 
          initial_offsets, 
          is_regularized, 
          reg_coef):
    """a jitted m_phase optimization

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
    solver = LBFGS(fun=m_loss, 
                    tol=utils.params["FTOL"],
                    jit=True,
                    maxiter=utils.params["MAXITER"],
                    verbose=False
                    )
    res = solver.run(offset0, mjx_model=mjx_model,
                            mjx_data=mjx_data,
                            kp_data=keypoints,
                            q=q,
                            initial_offsets=initial_offsets,
                            is_regularized=is_regularized,
                            reg_coef=reg_coef)
    return res.params
    

def m_phase(
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
    time_indices: jnp.ndarray,
    q: jnp.ndarray,
    initial_offsets: jnp.ndarray,
    reg_coef: float = 0.0,
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
    """
    # Define initial position of the optimization
    offset0 = get_site_pos(mjx_model).flatten()
    # Define which offsets to regularize
    is_regularized = []
    for k in utils.params["site_index_map"].keys():
        if any(n == k for n in utils.params["SITES_TO_REGULARIZE"]):
            is_regularized.append(jnp.array([1.0, 1.0, 1.0]))
        else:
            is_regularized.append(jnp.array([0.0, 0.0, 0.0]))
    is_regularized = jnp.stack(is_regularized).flatten()
    # Optimize dm
    keypoints = jnp.array(kp_data[time_indices, :])
    q = jnp.take(q, time_indices, axis=0)

    offset_opt_param = m_opt(offset0, mjx_model, 
                             mjx_data, keypoints, q, 
                             initial_offsets, is_regularized, reg_coef)
    
    print(f"learned offsets: {offset_opt_param}")
    # Set pose to the optimized m and step forward.
    mjx_model = set_site_pos(mjx_model, jnp.reshape(offset_opt_param, (-1, 3))) 
    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = kinematics(mjx_model, mjx_data)
    
    return mjx_model, mjx_data

@jit
def kinematics(mjx_model, mjx_data):
    return smooth.kinematics(mjx_model, mjx_data)

@jit
def com_pos(mjx_model, mjx_data):
    return smooth.com_pos(mjx_model, mjx_data)