"""Compute stac optimization on data."""

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from typing import Tuple, List
import time

from stac_mjx import stac_core
from stac_mjx import utils


def root_optimization(
    stac_core_obj: stac_core.StacCore,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    root_kp_idx: int,
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
    print(f"Root Optimization:")

    if mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE:  # better way to handle?
        root_dims = 4
    else:
        root_dims = 7
    print(f"Optimizing first {root_dims} qposes for root")
    s = time.time()
    q0 = jp.copy(mjx_data.qpos[:])

    q0.at[:3].set(kp_data[frame, :][root_kp_idx : root_kp_idx + 3])
    qs_to_opt = jp.zeros_like(q0, dtype=bool)
    qs_to_opt = qs_to_opt.at[:root_dims].set(True)
    kps_to_opt = jp.repeat(trunk_kps, 3)

    mjx_data, res = stac_core_obj.q_opt(
        mjx_model,
        mjx_data,
        kp_data[frame, :],
        qs_to_opt,
        kps_to_opt,
        q0,
        lb,
        ub,
        site_idxs,
    )

    mjx_data = utils.replace_qs(
        mjx_model, mjx_data, utils.make_qs(q0, qs_to_opt, res.params)
    )

    q0 = jp.copy(mjx_data.qpos[:])
    q0.at[:3].set(kp_data[frame, :][root_kp_idx : root_kp_idx + 3])

    # Trunk only optimization
    mjx_data, res = stac_core_obj.q_opt(
        mjx_model,
        mjx_data,
        kp_data[frame, :],
        qs_to_opt,
        kps_to_opt,
        q0,
        lb,
        ub,
        site_idxs,
    )

    mjx_data = utils.replace_qs(
        mjx_model, mjx_data, utils.make_qs(q0, qs_to_opt, res.params)
    )

    print(
        f"Root optimization finished in {(time.time() - s) / 60:.2f} minutes with an error of {res.state.error}"
    )

    return mjx_data


def offset_optimization(
    stac_core_obj: stac_core.StacCore,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    offsets: jp.ndarray,
    q: jp.ndarray,
    n_sample_frames: int,
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
    key = jax.random.PRNGKey(0)

    # shuffle frames to get sample frames
    all_indices = jp.arange(kp_data.shape[0])
    shuffled_indices = jax.random.permutation(key, all_indices, independent=True)
    time_indices = shuffled_indices[:n_sample_frames]

    s = time.time()
    print("Begining offset optimization:")

    # Define initial position of the optimization
    offset0 = utils.get_site_pos(mjx_model, site_idxs).flatten()

    keypoints = jp.array(kp_data[time_indices, :])
    q = jp.take(q, time_indices, axis=0)

    res = stac_core_obj.m_opt(
        offset0,
        mjx_model,
        mjx_data,
        keypoints,
        q,
        offsets,
        is_regularized,
        m_reg_coef,
        site_idxs,
    )

    offset_opt_param = res.params
    print(f"Final error of {res.state.error}")

    # Set body sites according to optimized offsets
    mjx_model = utils.set_site_pos(
        mjx_model, jp.reshape(offset_opt_param, (-1, 3)), site_idxs
    )

    # Forward kinematics, and save the results to the walker sites as well
    mjx_data = utils.kinematics(mjx_model, mjx_data)

    print(f"Offset optimization finished in {time.time() - s} seconds")

    return mjx_model, mjx_data, offset_opt_param


def pose_optimization(
    stac_core_obj: stac_core.StacCore,
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
    qposes = []
    xposes = []
    xquats = []
    marker_sites = []

    # Iterate through all of the frames
    frames = jp.arange(kp_data.shape[0])

    kps_to_opt = jp.ones(kp_data.shape[1], dtype=bool)
    qs_to_opt = jp.ones(mjx_model.nq, dtype=bool)
    print("Pose Optimization:")

    def f(mjx_data, kp_data, n_frame, parts):
        q0 = jp.copy(mjx_data.qpos[:])

        # While body opt, then part opt
        mjx_data, res = stac_core_obj.q_opt(
            mjx_model,
            mjx_data,
            kp_data[n_frame, :],
            qs_to_opt,
            kps_to_opt,
            q0,
            lb,
            ub,
            site_idxs,
        )

        mjx_data = utils.replace_qs(mjx_model, mjx_data, res.params)

        for part in parts:
            q0 = jp.copy(mjx_data.qpos[:])

            mjx_data, res = stac_core_obj.q_opt(
                mjx_model,
                mjx_data,
                kp_data[n_frame, :],
                part,
                kps_to_opt,
                q0,
                lb,
                ub,
                site_idxs,
            )

            mjx_data = utils.replace_qs(
                mjx_model, mjx_data, utils.make_qs(q0, part, res.params)
            )

        return mjx_data, res.state.error

    # Optimize over each frame, storing all the results
    frame_time = []
    frame_error = []
    for n_frame in frames:
        loop_start = time.time()

        mjx_data, error = f(mjx_data, kp_data, n_frame, indiv_parts)

        qposes.append(mjx_data.qpos[:])
        xposes.append(mjx_data.xpos[:])
        xquats.append(mjx_data.xquat[:])
        marker_sites.append(utils.get_site_xpos(mjx_data, site_idxs))

        frame_time.append(time.time() - loop_start)
        frame_error.append(error)

    print(f"Pose Optimization finished in {(time.time() - s) / 60.0:.2f} minutes")
    return (
        mjx_data,
        jp.array(qposes),
        xposes,
        xquats,
        marker_sites,
        frame_time,
        frame_error,
    )
