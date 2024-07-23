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
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        kp_data (jnp.Array): Keypoint data
        frame (int, optional): Frame to optimize. Defaults to 0.

    Returns:
        mjx.Data: An updated MJX Data
    """
    print("Root Optimization:")
    s = time.time()
    q0 = jnp.copy(mjx_data.qpos[:])

    # Set the center to help with finding the optima (does not need to be exact)
    q0 = q0.at[:3].set(kp_data[frame, :][12:15])
    qs_to_opt = jnp.zeros_like(q0, dtype=bool)
    qs_to_opt = qs_to_opt.at[:7].set(True)
    # kps_to_opt = jnp.repeat(jnp.ones(len(utils.params["kp_names"]), dtype=bool), 3)
    kps_to_opt = jnp.repeat(
        jnp.array(
            [
                any(
                    [n in kp_name for n in utils.params["TRUNK_OPTIMIZATION_KEYPOINTS"]]
                )
                for kp_name in utils.params["KP_NAMES"]
            ]
        ),
        3,
    )
    j = time.time()
    mjx_data, res = stac_base.q_opt(
        mjx_model,
        mjx_data,
        kp_data[frame, :],
        qs_to_opt,
        kps_to_opt,
        q0,
        utils.params["ROOT_FTOL"],
    )
    q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

    print(f"q_opt 1 finished in {time.time()-j} with an error of {res.state.error}")

    r = time.time()

    mjx_data = op.replace_qs(
        mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, q_opt_param)
    )
    print(f"Replace 1 finished in {time.time()-r}")

    q0 = jnp.copy(mjx_data.qpos[:])

    q0 = q0.at[:3].set(kp_data[frame, :][12:15])

    # Trunk only optimization
    j = time.time()
    print("starting q_opt 2")
    mjx_data, res = stac_base.q_opt(
        mjx_model,
        mjx_data,
        kp_data[frame, :],
        qs_to_opt,
        kps_to_opt,
        q0,
        utils.params["ROOT_FTOL"],
    )

    q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

    print(f"q_opt 1 finished in {time.time()-j} with an error of {res.state.error}")
    r = time.time()

    mjx_data = op.replace_qs(
        mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, q_opt_param)
    )

    print(f"Replace 2 finished in {time.time()-r}")
    print(f"Root optimization finished in {time.time()-s}")

    return mjx_data


def offset_optimization(mjx_model, mjx_data, kp_data, offsets, q):
    """Optimize the marker offsets based on proposed joint angles (q).

    Args:
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        kp_data (jnp.Array): Keypoint data
        offsets (jax.Array): List of offsets for the marker sites (to match up with keypoints)
        q (jax.Array): Proposed joint angles (relates to mjx_data.qpos)

    Returns:
        (mjx.Model, mjx.Data): An updated MJX Model and Data
    """
    key = jax.random.PRNGKey(0)

    # shuffle frames to get sample frames
    all_indices = jnp.arange(kp_data.shape[0])
    shuffled_indices = jax.random.permutation(key, all_indices, independent=True)
    time_indices = shuffled_indices[: utils.params["N_SAMPLE_FRAMES"]]

    s = time.time()
    print("Begining offset optimization:")

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

    res = stac_base.m_opt(
        offset0,
        mjx_model,
        mjx_data,
        keypoints,
        q,
        offsets,
        is_regularized,
        utils.params["M_REG_COEF"],
        utils.params["ROOT_FTOL"],
    )

    offset_opt_param = res.params
    print(f"Final error of {res.state.error}")

    # Set pose to the optimized m and step forward.
    mjx_model = op.set_site_pos(mjx_model, jnp.reshape(offset_opt_param, (-1, 3)))

    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = op.kinematics(mjx_model, mjx_data)

    print(f"offset optimization finished in {time.time()-s}")

    return mjx_model, mjx_data


def pose_optimization(mjx_model, mjx_data, kp_data) -> Tuple:
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.


    Args:
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        kp_data (jnp.Array): Keypoint data

    Returns:
        Tuple: _description_
    """
    s = time.time()
    q = []
    x = []
    walker_body_sites = []

    parts = utils.params["indiv_parts"]

    # Iterate through all of the frames
    frames = jnp.arange(kp_data.shape[0])

    kps_to_opt = jnp.repeat(jnp.ones(len(utils.params["KP_NAMES"]), dtype=bool), 3)
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
            q0,
            utils.params["FTOL"],
        )

        q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

        mjx_data = op.replace_qs(mjx_model, mjx_data, q_opt_param)

        for part in parts:
            q0 = jnp.copy(mjx_data.qpos[:])

            mjx_data, res = stac_base.q_opt(
                mjx_model,
                mjx_data,
                kp_data[n_frame, :],
                part,
                kps_to_opt,
                q0,
                utils.params["LIMB_FTOL"],
            )
            q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

            mjx_data = op.replace_qs(
                mjx_model, mjx_data, op.make_qs(q0, part, q_opt_param)
            )

        return mjx_data, res.state.error

    # Optimize over each frame, storing all the results
    frame_time = []
    frame_error = []
    for n_frame in frames:
        loop_start = time.time()

        mjx_data, error = f(mjx_data, kp_data, n_frame, parts)

        q.append(mjx_data.qpos[:])
        x.append(mjx_data.xpos[:])
        walker_body_sites.append(op.get_site_xpos(mjx_data))

        frame_time.append(time.time() - loop_start)
        frame_error.append(error)

    print(f"Pose Optimization done in {time.time()-s}")
    return (
        mjx_data,
        jnp.array(q),
        jnp.array(walker_body_sites),
        jnp.array(x),
        jnp.array(frame_time),
        jnp.array(frame_error),
    )


def package_data(mjx_model, physics, q, x, walker_body_sites, kp_data, batched=False):
    """Extract pose, offsets, data, and all parameters."""
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
