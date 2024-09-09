"""Compute stac optimization on data."""

import jax
import jax.numpy as jp

from typing import Tuple, List
import time

from stac_mjx import stac_base
from stac_mjx import operations as op

jit_vmap_q_opt_NEW = jax.jit(
    jax.vmap(stac_base.q_opt_NEW, in_axes=(0, 0, 0, 0, None, 0, None, None, None, None))
)

jit_vmap_m_opt_NEW = jax.jit(
    jax.vmap(stac_base.m_opt_NEW, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))
)

jit_vmap_replace_qs = jax.jit(jax.vmap(op.replace_qs))

jit_vmap_get_site_xpos = jax.jit(jax.vmap(op.get_site_xpos, in_axes=(0, None)))

jit_vmap_get_site_pos = jax.jit(jax.vmap(op.get_site_pos, in_axes=(0, None)))

jit_vmap_set_site_pos = jax.jit(jax.vmap(op.set_site_pos, in_axes=(0, 0, None)))

jit_vmap_kinematics = jax.jit(jax.vmap(op.kinematics))


def root_optimization(
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    lb: jp.ndarray,
    ub: jp.ndarray,
    site_idxs: jp.ndarray,
    trunk_kps: jp.ndarray,
    frame: int = 0,
):
    """Optimize fit for only the root.

    The root is optimized first so as to remove a common contribution to
    to rest of the child nodes. The choice of "root" node is somewhat
    aribitrary since the body model is an undirected graph. The root here
    is intended to mean the node closest to the center of mass of the
    animal at rest.

    Args:
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        kp_data (jp.Array): Keypoint data
        lb (jp.ndarray): Array of lower bounds for corresponding qpos elements
        ub (jp.ndarray): Array of upper bounds for corresponding qpos elements
        site_idxs (jp.ndarray): Array of indices of offset sites
        trunk_kps (jp.ndarray): Array of indices of keypoints to optimize
        frame (int, optional): Frame to optimize. Defaults to 0.

    Returns:
        mjx.Data: An updated MJX Data
    """
    # Set the center to help with finding the optima (does not need to be exact)
    # However should be close to the center of mass of the animal. The "magic numbers"
    # below are for the rodent.xml model. These will need to be changed for other
    # models, and possibly be computed for arbitray animal models.
    s = time.time()
    q0 = mjx_data.qpos.at[:, :3].set(kp_data[:, frame, 12:15])

    qs_to_opt = jp.zeros_like(mjx_data.qpos, dtype=bool)
    qs_to_opt = qs_to_opt.at[:, :7].set(True)
    kps_to_opt = jp.repeat(trunk_kps, 3)

    # NEW OPTIMIZE
    final_params, final_loss, num_iters = jit_vmap_q_opt_NEW(
        mjx_model,
        mjx_data,
        kp_data[:, frame, :],
        qs_to_opt,
        kps_to_opt,
        q0,
        lb,
        ub,
        site_idxs,
        1e-5,
    )
    r = time.time()

    print(f"opt done in {r - s}")
    mjx_data = jit_vmap_replace_qs(
        mjx_model, mjx_data, op.make_qs(q0, qs_to_opt, final_params)
    )
    print(f"replace done in {time.time()-r}")
    return mjx_data, final_loss, num_iters


def offset_optimization(
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    offsets: jp.ndarray,
    q: jp.ndarray,
    time_indices: jp.ndarray,
    is_regularized: jp.ndarray,
    site_idxs: jp.ndarray,
    m_reg_coef: float,
):
    """Optimize the marker offsets based on proposed joint angles (q).

    Args:
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        kp_data (jp.Array): Keypoint data
        offsets (jax.Array): List of offsets for the marker sites (to match up with keypoints)
        q (jax.Array): Proposed joint angles (corresponds to mjx_data.qpos)
        n_sample_frames (int): Number of frames to sample when computing residual
        is_regularized (jp.ndarray): Boolean mask representing sites to regularize
        site_idxs (jp.ndarray): Array of indices of offset sites
        m_reg_coef (float): Regularization coefficient to apply to regularized sites

    Returns:
        (mjx.Model, mjx.Data): An updated MJX Model and Data
    """

    print("Begining offset optimization:")

    cur_offsets = jit_vmap_get_site_pos(mjx_model, site_idxs)
    # Define initial position of the optimization
    offset0 = jp.reshape(cur_offsets, (kp_data.shape[0], -1))
    keypoints = jp.take(kp_data, time_indices, axis=1)
    q = jp.take(q, time_indices, axis=1)

    final_params, final_loss, num_iters = jit_vmap_m_opt_NEW(
        offset0,
        mjx_model,
        mjx_data,
        keypoints,
        q,
        offsets,
        is_regularized,
        m_reg_coef,
        site_idxs,
        1e-4,
    )

    # offset_opt_param = res.params
    # print(f"Final error of {res.state.error}")

    # Set pose to the optimized m and step forward.
    mjx_model = jit_vmap_set_site_pos(
        mjx_model, jp.reshape(final_params, (kp_data.shape[0], -1, 3)), site_idxs
    )

    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = jit_vmap_kinematics(mjx_model, mjx_data)

    return mjx_model, mjx_data, final_loss, num_iters


def pose_optimization(
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    lb: jp.ndarray,
    ub: jp.ndarray,
    site_idxs: jp.ndarray,
    indiv_parts: List[jp.ndarray],
) -> Tuple:
    """Perform q_phase over the entire clip.

    Args:
        mjx_model (mjx.Model): MJX Model
        mjx_data (mjx.Data): MJX Data
        kp_data (jp.ndarray): Keypoint data
        lb (jp.ndarray): Array of lower bounds for corresponding qpos elements
        ub (jp.ndarray): Array of upper bounds for corresponding qpos elements
        site_idxs (jp.ndarray): Array of indices of offset sites
        indiv_parts (List[jp.ndarray]): List of joints to optimize, used in individual part optimization

    Returns:
        Tuple: Updated mjx.Data, optimized qpos, offset site xpos, mjx.Data.xpos for each frame, and info for logging (optimization time and errors)
    """
    s = time.time()
    q = []
    x = []
    walker_body_sites = []

    # Iterate through all of the frames
    frames = jp.arange(kp_data.shape[1])

    kps_to_opt = jp.ones(kp_data.shape[2], dtype=bool)
    qs_to_opt = jp.ones(mjx_data.qpos.shape, dtype=bool)
    print("Pose Optimization:")

    def f(mjx_data, kp_data, n_frame, parts):
        q0 = mjx_data.qpos
        total_iters = 0
        final_params, final_loss, num_iters = jit_vmap_q_opt_NEW(
            mjx_model,
            mjx_data,
            kp_data[:, n_frame, :],
            qs_to_opt,
            kps_to_opt,
            q0,
            lb,
            ub,
            site_idxs,
            1e-3,
        )
        total_iters += num_iters
        mjx_data = jit_vmap_replace_qs(mjx_model, mjx_data, final_params)

        for part in parts:
            q0 = mjx_data.qpos

            final_params, final_loss, num_iters = jit_vmap_q_opt_NEW(
                mjx_model,
                mjx_data,
                kp_data[:, n_frame, :],
                qs_to_opt,
                kps_to_opt,
                q0,
                lb,
                ub,
                site_idxs,
                1e-6,
            )

            mjx_data = jit_vmap_replace_qs(
                mjx_model, mjx_data, op.make_qs(q0, part, final_params)
            )
            total_iters += num_iters

        return mjx_data, final_loss, total_iters

    # Optimize over each frame, storing all the results
    frame_time = []
    frame_error = []
    frame_iters = []
    for n_frame in frames:
        loop_start = time.time()

        mjx_data, final_loss, total_iters = f(mjx_data, kp_data, n_frame, indiv_parts)

        q.append(mjx_data.qpos)
        x.append(mjx_data.xpos)
        walker_body_sites.append(jit_vmap_get_site_xpos(mjx_data, site_idxs))

        frame_time.append(time.time() - loop_start)
        frame_error.append(final_loss)
        frame_iters.append(total_iters)

    print(f"Pose Optimization done in {time.time()-s}")
    return (
        mjx_data,
        jp.array(q),
        jp.array(walker_body_sites),
        jp.array(x),
        jp.array(frame_time),
        jp.array(frame_error),
        jp.array(frame_iters),
    )
