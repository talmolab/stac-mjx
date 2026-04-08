"""Compute stac optimization on data."""

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from typing import Tuple, List
import time

from stac_mjx import stac_core
from stac_mjx import utils


def _estimate_orientation_from_keypoints(
    kp_flat: jp.ndarray,
    rear_idx: int,
    left_idx: int,
    right_idx: int,
    front_idx: int = -1,
) -> jp.ndarray:
    """Estimate per-frame body orientation quaternion from trunk keypoints.

    Uses 3-4 keypoints to build a body-frame rotation matrix per frame
    and converts to MuJoCo quaternion [w, x, y, z].

    Args:
        kp_flat: Keypoint data (T, n_kp*3).
        rear_idx: Index of rear trunk keypoint in KP_NAMES.
        left_idx: Index of left keypoint in KP_NAMES.
        right_idx: Index of right keypoint in KP_NAMES.
        front_idx: Index of front keypoint. -1 to skip.

    Returns:
        Quaternions (T, 4) in MuJoCo [w, x, y, z] format.
    """

    def _quat_from_frame(rear, left, right, front, has_front):
        """Build quaternion from 3-4 keypoint positions (single frame)."""
        # Lateral axis: left -> right (fly's left-to-right)
        lat = left - right
        lat_norm = jp.linalg.norm(lat)
        lat = jp.where(lat_norm > 1e-12, lat / lat_norm, jp.array([0.0, 1.0, 0.0]))

        # Forward-ish vector
        mid = (left + right) * 0.5
        fwd_raw = jp.where(has_front, front - rear, mid - rear)
        fwd_raw_norm = jp.linalg.norm(fwd_raw)
        fwd_raw = jp.where(
            fwd_raw_norm > 1e-12, fwd_raw / fwd_raw_norm, jp.array([1.0, 0.0, 0.0])
        )

        # Up = cross(fwd_raw, lat), then re-orthogonalize fwd
        up = jp.cross(fwd_raw, lat)
        up_norm = jp.linalg.norm(up)
        up = jp.where(up_norm > 1e-12, up / up_norm, jp.array([0.0, 0.0, 1.0]))
        fwd = jp.cross(lat, up)
        fwd_norm = jp.linalg.norm(fwd)
        fwd = jp.where(fwd_norm > 1e-12, fwd / fwd_norm, jp.array([1.0, 0.0, 0.0]))

        # Rotation matrix columns: [fwd, lat, up]
        R = jp.stack([fwd, lat, up], axis=-1)  # (3, 3)

        # Shepperd method: rotation matrix -> quaternion [w, x, y, z]
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        s0 = jp.sqrt(jp.maximum(trace + 1.0, 0.0)) * 2.0
        s1 = jp.sqrt(jp.maximum(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
        s2 = jp.sqrt(jp.maximum(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
        s3 = jp.sqrt(jp.maximum(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0

        q0 = jp.array([s0 / 4.0, (R[2, 1] - R[1, 2]) / s0,
                        (R[0, 2] - R[2, 0]) / s0, (R[1, 0] - R[0, 1]) / s0])
        q1 = jp.array([(R[2, 1] - R[1, 2]) / s1, s1 / 4.0,
                        (R[0, 1] + R[1, 0]) / s1, (R[0, 2] + R[2, 0]) / s1])
        q2 = jp.array([(R[0, 2] - R[2, 0]) / s2, (R[0, 1] + R[1, 0]) / s2,
                        s2 / 4.0, (R[1, 2] + R[2, 1]) / s2])
        q3 = jp.array([(R[1, 0] - R[0, 1]) / s3, (R[0, 2] + R[2, 0]) / s3,
                        (R[1, 2] + R[2, 1]) / s3, s3 / 4.0])

        diag = jp.array([trace, R[0, 0], R[1, 1], R[2, 2]])
        best = jp.argmax(diag)
        q = jp.where(best == 0, q0,
            jp.where(best == 1, q1,
            jp.where(best == 2, q2, q3)))

        q = q / jp.linalg.norm(q)
        q = jp.where(q[0] < 0, -q, q)
        return q

    T = kp_flat.shape[0]

    rear = kp_flat[:, rear_idx * 3 : rear_idx * 3 + 3]
    left = kp_flat[:, left_idx * 3 : left_idx * 3 + 3]
    right = kp_flat[:, right_idx * 3 : right_idx * 3 + 3]

    if front_idx >= 0:
        front = kp_flat[:, front_idx * 3 : front_idx * 3 + 3]
        has_front = jp.array(True)
    else:
        front = jp.zeros((T, 3))
        has_front = jp.array(False)

    quats = jax.vmap(_quat_from_frame)(rear, left, right, front, jp.broadcast_to(has_front, (T,)))

    # Enforce consistent quaternion sign across trajectory
    def _consistent_sign(carry, q):
        prev = carry
        dot = jp.dot(prev, q)
        q = jp.where(dot < 0, -q, q)
        return q, q

    _, quats = jax.lax.scan(_consistent_sign, quats[0], quats)

    return quats


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

    q0 = q0.at[:3].set(kp_data[frame, :][root_kp_idx : root_kp_idx + 3])
    qs_to_opt = jp.zeros_like(q0, dtype=bool)
    qs_to_opt = qs_to_opt.at[:root_dims].set(True)
    kps_to_opt = jp.repeat(trunk_kps, 3)

    no_reg = jp.zeros(mjx_model.nq)

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
        no_reg,
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
        no_reg,
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
    kp_weights: jp.ndarray = None,
    q_reg_weights: jp.ndarray = None,
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
        kp_weights (jp.ndarray, optional): Per-keypoint weights for the loss (repeated 3x for xyz). Defaults to uniform 1.0.
        q_reg_weights (jp.ndarray, optional): Per-qpos L2 regularization weights toward rest (q=0). Defaults to zeros.

    Returns:
        Tuple: Updated mjx.Data, optimized qpos, offset site xpos, mjx.Data.xpos for each frame, and info for logging (optimization time and errors)
    """
    s = time.time()

    # Iterate through all of the frames
    frames = jp.arange(kp_data.shape[0])

    kps_to_opt = kp_weights if kp_weights is not None else jp.ones(kp_data.shape[1])
    _q_reg = q_reg_weights if q_reg_weights is not None else jp.zeros(mjx_model.nq)
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
            _q_reg,
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
                _q_reg,
            )

            mjx_data = utils.replace_qs(
                mjx_model, mjx_data, utils.make_qs(q0, part, res.params)
            )

        return mjx_data, res.state.error

    if getattr(stac_core_obj, '_use_jaxls', False):
        root_kp_idx = getattr(stac_core_obj, '_root_kp_idx', -1)
        return _pose_optimization_jaxls(
            stac_core_obj, mjx_model, mjx_data, kp_data,
            lb, ub, site_idxs, kps_to_opt, _q_reg, s,
            root_kp_idx=root_kp_idx,
        )

    # Optimize over each frame using lax.scan to avoid Python loop unrolling.
    def scan_fn(mjx_data, n_frame):
        mjx_data, error = f(mjx_data, kp_data, n_frame, indiv_parts)
        outputs = (
            mjx_data.qpos[:],
            mjx_data.xpos[:],
            mjx_data.xquat[:],
            utils.get_site_xpos(mjx_data, site_idxs),
            error,
        )
        return mjx_data, outputs

    mjx_data, (qposes, xposes, xquats, marker_sites, frame_error) = jax.lax.scan(
        scan_fn, mjx_data, frames
    )
    frame_time = []

    print(f"Pose Optimization finished in {(time.time() - s) / 60.0:.2f} minutes")
    return (
        mjx_data,
        qposes,
        xposes,
        xquats,
        marker_sites,
        frame_time,
        frame_error,
    )


def _pose_optimization_jaxls(
    stac_core_obj: stac_core.StacCore,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    lb: jp.ndarray,
    ub: jp.ndarray,
    site_idxs: jp.ndarray,
    kps_to_opt: jp.ndarray,
    q_reg_weights: jp.ndarray,
    s: float,
    root_kp_idx: int = -1,
) -> Tuple:
    """Batch trajectory pose optimization using jaxls Levenberg-Marquardt.

    Solves all T frames simultaneously in one jaxls LeastSquaresProblem.
    Adjacent frames are coupled via a smoothness cost when smooth_weight > 0.

    Args:
        stac_core_obj: StacCore with _use_jaxls=True.
        mjx_model: MJX Model.
        mjx_data: MJX Data (used as initial state / FK template).
        kp_data: Keypoint observations (T, n_kp*3) or (T, n_kp, 3).
        lb, ub: Joint bounds (nq,).
        site_idxs: Marker site indices.
        kps_to_opt: Per-keypoint weight mask (n_kp*3,).
        q_reg_weights: Per-joint regularization weights (nq,).
        s: Wall-clock start time for logging.
        root_kp_idx: Index of root keypoint for warm-starting.

    Returns:
        Same tuple as pose_optimization().
    """
    T = kp_data.shape[0]
    qs_to_opt = jp.ones(mjx_model.nq, dtype=bool)

    # Flatten kp_data to (T, n_kp*3) if needed
    kp_flat = kp_data.reshape(T, -1) if kp_data.ndim == 3 else kp_data

    # Build per-frame warm-start: tile the root-optimized qpos, then
    # override root xyz with each frame's root keypoint position.
    q_base = mjx_data.qpos
    q_init_all = jp.tile(q_base, (T, 1))
    if root_kp_idx >= 0 and mjx_model.jnt_type[0] in (
        mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_SLIDE,
    ):
        kp_root_xyz = kp_flat[:, root_kp_idx * 3 : root_kp_idx * 3 + 3]
        q_init_all = q_init_all.at[:, :3].set(kp_root_xyz)

    # Per-frame orientation warm-start from trunk keypoints
    orient_indices = getattr(stac_core_obj, '_orientation_kp_indices', None)
    if orient_indices is not None and mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        rear_idx, left_idx, right_idx, front_idx = orient_indices
        quats = _estimate_orientation_from_keypoints(
            kp_flat, rear_idx, left_idx, right_idx, front_idx,
        )
        q_init_all = q_init_all.at[:, 3:7].set(quats)

    chunk_size = stac_core_obj._jaxls_chunk_size
    if chunk_size > 0 and T > chunk_size:
        qposes_chunks = []
        for c_start in range(0, T, chunk_size):
            c_end = min(c_start + chunk_size, T)
            q_init_chunk = q_init_all[c_start:c_end]
            if qposes_chunks:
                q_prev = qposes_chunks[-1][-1]
                if mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
                    q_init_chunk = q_init_chunk.at[:, 7:].set(q_prev[7:])
                elif mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE:
                    q_init_chunk = q_init_chunk.at[:, 4:].set(q_prev[4:])
            qposes_chunk = stac_core_obj._jaxls_solver.solve_trajectory(
                q_init=q_init_chunk,
                mjx_model=mjx_model,
                mjx_data_template=mjx_data,
                kp_data=kp_flat[c_start:c_end],
                qs_to_opt=qs_to_opt,
                kps_to_opt=kps_to_opt,
                lb=lb,
                ub=ub,
                site_idxs=site_idxs,
                q_reg_weights=q_reg_weights,
            )
            qposes_chunks.append(qposes_chunk)
            print(f"  Chunk {c_start}-{c_end} / {T}")
        qposes = jp.concatenate(qposes_chunks, axis=0)
    else:
        qposes = stac_core_obj._jaxls_solver.solve_trajectory(
            q_init=q_init_all,
            mjx_model=mjx_model,
            mjx_data_template=mjx_data,
            kp_data=kp_flat,
            qs_to_opt=qs_to_opt,
            kps_to_opt=kps_to_opt,
            lb=lb,
            ub=ub,
            site_idxs=site_idxs,
            q_reg_weights=q_reg_weights,
        )

    # Compute xpos / xquat / marker_sites for all frames via vmap
    def fk_frame(q):
        data = mjx_data.replace(qpos=q)
        data = utils.kinematics(mjx_model, data)
        data = utils.com_pos(mjx_model, data)
        return data.xpos, data.xquat, utils.get_site_xpos(data, site_idxs)

    xposes, xquats, marker_sites = jax.vmap(fk_frame)(qposes)

    # Update mjx_data to last frame for consistency with callers
    mjx_data = utils.replace_qs(mjx_model, mjx_data, qposes[-1])

    # frame_error: compute per-frame marker residual norms
    def frame_err(q, kp):
        data = mjx_data.replace(qpos=q)
        data = utils.kinematics(mjx_model, data)
        data = utils.com_pos(mjx_model, data)
        markers = utils.get_site_xpos(data, site_idxs).flatten()
        return jp.sum(jp.square((kp - markers) * kps_to_opt))

    frame_error = jax.vmap(frame_err)(qposes, kp_flat)

    print(f"Pose Optimization (jaxls batch) finished in {(time.time() - s) / 60.0:.2f} minutes")
    return mjx_data, qposes, xposes, xquats, marker_sites, [], frame_error
