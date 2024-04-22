"""Implementation of stac for animal motion capture in dm_control suite."""
from typing import List, Dict, Text, Union, Tuple
import jax
import jax.numpy as jnp
import operations as op
from jax import jit
from jaxopt import LBFGSB, LBFGS
import optax
import utils
import logging 
    
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
    # mjx_data = mjx_data.replace(qpos=op.make_qs(initial_q, qs_to_opt, q))
    
    # Clip to bounds ourselves because of potential jaxopt bug
    mjx_data = mjx_data.replace(qpos=jnp.clip(op.make_qs(initial_q, qs_to_opt, q), utils.params['lb'], utils.params['ub']))
    # Forward kinematics
    mjx_data = op.kinematics(mjx_model, mjx_data)
    mjx_data = op.com_pos(mjx_model, mjx_data)

    # Get marker site xpos
    markers = op.get_site_xpos(mjx_data).flatten()
    residual = kp_data - markers

    # Set irrelevant body sites to 0
    residual = residual * kps_to_opt
    residual =  jnp.sum(jnp.square(residual))

    return residual


@jit
def q_opt(
    mjx_model,
    mjx_data,
    marker_ref_arr: jnp.ndarray,
    qs_to_opt: jnp.ndarray,
    kps_to_opt: jnp.ndarray,
    maxiter: int,
    q0: jnp.ndarray,
    ftol: float,
):
    """Update q_pose using estimated marker parameters.
    """
    try:
        lb = utils.params['lb']
        ub = utils.params['ub']
        # jax.debug.print(f"{len(lb)} {len(ub)} {q0.shape}")
        return mjx_data, q_solver.run(q0, bounds=jnp.array((lb, ub)), mjx_model=mjx_model, 
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
        # Unpack arguments
        qpos, kp = input
        mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized = carry

        # Get the offset relative to the initial position, only for markers you wish to regularize
        reg_term = reg_term + (jnp.square(offsets - initial_offsets.flatten())) * is_regularized

        # Set qpos and offsets
        mjx_data = mjx_data.replace(qpos=qpos)
        mjx_model = op.set_site_pos(mjx_model, jnp.reshape(offsets, (-1, 3))) 

        # Forward kinematics
        mjx_data = op.kinematics(mjx_model, mjx_data)
        mjx_data = op.com_pos(mjx_model, mjx_data)
        markers = op.get_site_xpos(mjx_data).flatten()

        # Accumulate squared residual 
        residual = (residual + jnp.square((kp - markers)))
        return (mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized), None
    
    (mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized), _ = jax.lax.scan(
            f, 
            (mjx_model, mjx_data, jnp.zeros(69), jnp.zeros(69), initial_offsets, is_regularized), 
            (q, kp_data)
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
    ftol
):
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
    
    res = m_solver.run(offset0, mjx_model=mjx_model,
                            mjx_data=mjx_data,
                            kp_data=keypoints,
                            q=q,
                            initial_offsets=initial_offsets,
                            is_regularized=is_regularized,
                            reg_coef=reg_coef)
    
    return res
    

def m_phase(
    mjx_model,
    mjx_data,
    kp_data: jnp.ndarray,
    time_indices: jnp.ndarray,
    q: jnp.ndarray,
    initial_offsets: jnp.ndarray,
    ftol,
    reg_coef: float = 0.0,
):
    """Estimate marker offset, keeping qpos fixed.

    Args:
        mjx_model (_type_): _description_
        mjx_data (_type_): _description_
        kp_data (jnp.ndarray): _description_
        time_indices (jnp.ndarray): _description_
        q (jnp.ndarray): _description_
        initial_offsets (jnp.ndarray): _description_
        ftol (_type_): _description_
        reg_coef (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    # Define initial position of the optimization
    offset0 = op.get_site_pos(mjx_model).flatten()

    # Define which offsets to regularize
    is_regularized = []
    for k in utils.params["site_index_map"].keys():
        if any(n == k for n in utils.params["SITES_TO_REGULARIZE"]):
            is_regularized.append(jnp.array([1.0, 1.0, 1.0]))
        else:
            is_regularized.append(jnp.array([0.0, 0.0, 0.0]))
    is_regularized = jnp.stack(is_regularized).flatten()

    keypoints = jnp.array(kp_data[time_indices, :])
    q = jnp.take(q, time_indices, axis=0)

    res = m_opt(offset0, mjx_model, 
                mjx_data, keypoints, q, 
                initial_offsets, is_regularized, 
                reg_coef, ftol)
    
    offset_opt_param = res.params
    print(f"Final error of {res.state.error} \n params: {offset_opt_param}")

    # Set pose to the optimized m and step forward.
    mjx_model = op.set_site_pos(mjx_model, jnp.reshape(offset_opt_param, (-1, 3))) 

    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = op.kinematics(mjx_model, mjx_data)
    
    return mjx_model, mjx_data

q_solver = LBFGSB(fun=q_loss, 
                        tol=utils.params["FTOL"],
                        maxiter=utils.params["Q_MAXITER"],
                        history_size=20,
                        # maxls=40,
                        # use_gamma=False,
                        stepsize=1.0,
                        jit=True,
                        verbose=0
                        )
m_solver = LBFGS(fun=m_loss, 
                    tol=utils.params["FTOL"],
                    jit=True,
                    maxiter=utils.params["M_MAXITER"],
                    history_size=20,
                    verbose=0
                    )
