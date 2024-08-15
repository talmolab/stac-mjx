"""Compute stac optimization on data."""

import jax
import jax.numpy as jnp

from typing import Tuple
import time

from stac_mjx import stac_base
from stac_mjx import utils
from stac_mjx import operations as op


def root_optimization(mjx_model, mjx_data, kp_data, ub, lb, frame: int = 0):
    """Optimize fit for only the root.

    The root is optimized first so as to remove a common contribution to
    to rest of the child nodes. The choice of "root" node is somewhat
    aribitrary since the body model is an undirected graph. The root here
    is intended to mean the node closest to the center of mass of the
    animal at rest.


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
    # However should be close to the center of mass of the animal. The "magic numbers"
    # below are for the rodent.xml model. These will need to be changed for other
    # models, and possibly be computed for arbitray animal models.
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
        lb,
        ub,
    )
    # q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

    print(f"q_opt 1 finished in {time.time()-j} with an error of {res.state.error}")

    r = time.time()

    mjx_data = op.replace_qs(mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, res.params))
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
        lb,
        ub,
    )

    # q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

    print(f"q_opt 1 finished in {time.time()-j} with an error of {res.state.error}")
    r = time.time()

    mjx_data = op.replace_qs(mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, res.params))

    print(f"Replace 2 finished in {time.time()-r}")
    print(f"Root optimization finished in {time.time()-s}")

    return mjx_data


def offset_optimization(
    mjx_model, mjx_data, kp_data, offsets, q, n_sample_frames, is_regularized
):
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
    time_indices = shuffled_indices[:n_sample_frames]

    s = time.time()
    print("Begining offset optimization:")

    # Define initial position of the optimization
    offset0 = op.get_site_pos(mjx_model).flatten()

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
    )

    offset_opt_param = res.params
    print(f"Final error of {res.state.error}")

    # Set pose to the optimized m and step forward.
    mjx_model = op.set_site_pos(mjx_model, jnp.reshape(offset_opt_param, (-1, 3)))

    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = op.kinematics(mjx_model, mjx_data)

    print(f"offset optimization finished in {time.time()-s}")

    return mjx_model, mjx_data


def pose_optimization(mjx_model, mjx_data, kp_data, lb, ub) -> Tuple:
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
            mjx_model, mjx_data, kp_data[n_frame, :], qs_to_opt, kps_to_opt, q0, lb, ub
        )

        # q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

        mjx_data = op.replace_qs(mjx_model, mjx_data, res.params)

        for part in parts:
            q0 = jnp.copy(mjx_data.qpos[:])

            mjx_data, res = stac_base.q_opt(
                mjx_model,
                mjx_data,
                kp_data[n_frame, :],
                part,
                kps_to_opt,
                q0,
                lb,
                ub,
            )
            # q_opt_param = jnp.clip(res.params, utils.params["lb"], utils.params["ub"])

            mjx_data = op.replace_qs(
                mjx_model, mjx_data, op.make_qs(q0, part, res.params)
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
