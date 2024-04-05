"""Compute stac optimization on data."""
import jax
from jax import vmap
import jax.numpy as jnp
import stac_base
import operations as op
import utils
from typing import List, Dict, Tuple, Text
import time
import logging

def root_optimization(mjx_model, mjx_data, kp_data, frame: int = 0):
    """Optimize only the root.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary
        frame (int, optional): Frame to optimize
    """
    lb = jnp.concatenate([-jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 0]])
    lb = jnp.minimum(lb, 0.0)
    ub = jnp.concatenate([jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 1]])
    utils.params['lb'] = lb
    utils.params['ub'] = ub
    s = time.time()
    print("Root Optimization:")

    q0 = jnp.copy(mjx_data.qpos[:])

    # Set the center to help with finding the optima (does not need to be exact)
    q0 = q0.at[:3].set(kp_data[frame, :][12:15])
    qs_to_opt = jnp.zeros_like(q0, dtype=bool)
    qs_to_opt = qs_to_opt.at[:7].set(True)
    print(f"Initial qs: {q0}")
    kps_to_opt = jnp.repeat(jnp.ones(len(utils.params["kp_names"]), dtype=bool), 3)
    j = time.time()
    mjx_data, res = stac_base.q_opt(
        mjx_model,
        mjx_data,
        kp_data[frame, :],
        qs_to_opt,
        kps_to_opt,
        utils.params["ROOT_MAXITER"],
        q0,
        utils.params["ROOT_FTOL"],
    )
    q_opt_param = jnp.clip(res.params, utils.params['lb'], utils.params['ub'])

    print(f"q_opt 1 finished in {time.time()-j} with an error of {res.state.error}")
    print(f"Resulting qs: {q_opt_param}")

    r = time.time()

    mjx_data = op.replace_qs(mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, q_opt_param))
    print(f"Replace 1 finished in {time.time()-r}")
    
    kps_to_opt = jnp.repeat(
            jnp.array([
                any([n in kp_name for n in utils.params["TRUNK_OPTIMIZATION_KEYPOINTS"]])
                for kp_name in utils.params["kp_names"]
            ]), 3)
    
    q0 = jnp.copy(mjx_data.qpos[:])

    q0 = q0.at[:3].set(kp_data[frame, :][12:15])

    # Trunk only optimization
    j = time.time()
    print("starting q_opt 2")
    print(f"starting qs: {q0}")
    mjx_data, res = stac_base.q_opt(
        mjx_model, 
        mjx_data,
        kp_data[frame, :],
        qs_to_opt,
        kps_to_opt,
        utils.params["ROOT_MAXITER"],
        q0,
        utils.params["ROOT_FTOL"],
    )
    
    q_opt_param = jnp.clip(res.params, utils.params['lb'], utils.params['ub'])

    print(f"q_opt 1 finished in {time.time()-j} with an error of {res.state.error}")
    r = time.time()

    mjx_data = op.replace_qs(mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, q_opt_param))

    print(f"Replace 2 finished in {time.time()-r}")
    print(f"qs after replace: {mjx_data.qpos}")
    print(f"Root optimization finished in {time.time()-s}")

    return mjx_data


def offset_optimization(mjx_model, mjx_data, kp_data, offsets, q):
    key = jax.random.PRNGKey(0)
    # N_SAMPLE_FRAMES has to be less than N_FRAMES_PER_CLIP
    N_FRAMES_PER_CLIP = utils.params["N_FRAMES_PER_CLIP"]  # Total number of frames per clip
    N_SAMPLE_FRAMES = utils.params["N_SAMPLE_FRAMES"]      # Number of frames to sample

    # shuffle frames to get sample frames
    all_indices = jnp.arange(N_FRAMES_PER_CLIP)
    shuffled_indices = jax.random.permutation(key, all_indices, independent=True)
    time_indices = shuffled_indices[:N_SAMPLE_FRAMES]
    
    s = time.time()
    print("Begining offset optimization:")

    mjx_model, mjx_data = stac_base.m_phase(
        mjx_model, 
        mjx_data,
        kp_data,
        time_indices,
        q,
        offsets,
        utils.params["ROOT_FTOL"],
        utils.params["M_REG_COEF"],
    )
    
    print(f"offset optimization finished in {time.time()-s}")

    return mjx_model, mjx_data


def pose_optimization(mjx_model, mjx_data, kp_data) -> Tuple:
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary.

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    s = time.time()
    q = []
    x = []
    walker_body_sites = []
    
    parts = utils.params["indiv_parts"]

    # Iterate through all of the frames
    frames = jnp.arange(kp_data.shape[0])
    
    kps_to_opt = jnp.repeat(jnp.ones(len(utils.params["kp_names"]), dtype=bool), 3)
    qs_to_opt = jnp.ones(mjx_model.nq, dtype=bool)
    print("Pose Optimization:")
    
    def f(mjx_data, kp_data, n_frame, parts):
        q0 = jnp.copy(mjx_data.qpos[:])
        
        # While body opt, then part opt
        mjx_data, res = stac_base.q_opt(
            mjx_model, 
            mjx_data,
            kp_data[n_frame, :],
            qs_to_opt,
            kps_to_opt,
            utils.params["Q_MAXITER"],
            q0,
            utils.params["FTOL"],
        )

        q_opt_param = jnp.clip(res.params, utils.params['lb'], utils.params['ub'])
        
        mjx_data = op.replace_qs(mjx_model, mjx_data, q_opt_param)
        
        for part in parts:
            q0 = jnp.copy(mjx_data.qpos[:])

            mjx_data, res = stac_base.q_opt(
                mjx_model, 
                mjx_data,
                kp_data[n_frame, :],
                part,
                kps_to_opt,
                utils.params["Q_MAXITER"],
                q0,
                utils.params["LIMB_FTOL"],
            )
            q_opt_param = jnp.clip(res.params, utils.params['lb'], utils.params['ub'])

            mjx_data = op.replace_qs(mjx_model, mjx_data, op.make_qs(q0, part, q_opt_param))
        
        return mjx_data, res.state.error
    
    # Optimize over each frame, storing all the results
    frame_data = []
    for n_frame in frames:
        loop_start = time.time()
        
        mjx_data, error = f(mjx_data, kp_data, n_frame, parts)
        
        q.append(mjx_data.qpos[:])
        x.append(mjx_data.xpos[:])
        walker_body_sites.append(op.get_site_xpos(mjx_data))
        
        frame_data.append((time.time()-loop_start, error))
    
    print(f"Pose Optimization done in {time.time()-s}")
    return mjx_data, jnp.array(q), jnp.array(walker_body_sites), jnp.array(x), frame_data

def initialize_part_names(physics):
    # Get the ids of the limbs, accounting for quaternion and pos
    part_names = physics.named.data.qpos.axes.row.names
    for _ in range(6):
        part_names.insert(0, part_names[0])
    return part_names

def package_data(mjx_model, physics, q, x, walker_body_sites, kp_data, batched=False):
    # Extract pose, offsets, data, and all parameters
    if batched:
        # prepare batched data to be packaged
        get_batch_offsets = vmap(op.get_site_pos)
        offsets = get_batch_offsets(mjx_model).copy()[0]
        x = x.reshape(-1, x.shape[-1])
        q = q.reshape(-1, q.shape[-1])
    else:
        offsets = op.get_site_pos(mjx_model).copy()
        
    names_xpos = physics.named.data.xpos.axes.row.names
    
    print(f"shape of qpos: {q.shape}")
    kp_data = kp_data.reshape(-1, kp_data.shape[-1])
    data = {
        "qpos": q,
        "xpos": x,
        "walker_body_sites": walker_body_sites,
        "offsets": offsets,
        "names_qpos": utils.params["part_names"],
        "names_xpos": names_xpos,
        "kp_data": jnp.copy(kp_data),
    }
    
    for k, v in utils.params.items():
        data[k] = v
    
    return data
