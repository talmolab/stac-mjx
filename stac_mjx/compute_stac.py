"""Compute stac optimization on data."""

import time

import jax
import jax.numpy as jp
import mujoco
from jax import Array
from jaxtyping import Bool, Float, Int
from mujoco import mjx

from stac_mjx import stac_core, utils
from stac_mjx.stac_core import build_q_opt_problem, q_opt


def root_optimization(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    root_kp_idx: int,
    lb: Float[Array, " n_qpos"],
    ub: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
    trunk_kps: Bool[Array, " n_keypoints"],
    frame: int = 0,
    n_solver_max_iters: int = 50,
    initial_step_damping: float = 1.0,
) -> mjx.Data:
    """Optimize root DOFs for a single frame.

    Seeds root translation from the selected root keypoint, then runs
    two root-only q-phase solves using trunk keypoints as the objective.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.
        kp_data: Flattened keypoint data.
        root_kp_idx: Index of root keypoint in the ordered keypoint list.
        lb: Lower bounds on joint angles.
        ub: Upper bounds on joint angles.
        site_idxs: Indices of marker sites.
        trunk_kps: Boolean mask selecting trunk keypoints.
        frame: Frame index to optimize.
        n_solver_max_iters: Maximum solver iterations.
        initial_step_damping: Initial damping on the solver step.

    Returns:
        Updated MJX data after root optimization.
    """
    print(f"Root Optimization:")

    if mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE:  # better way to handle?
        root_dims = 4
    else:
        root_dims = 7
    print(f"Optimizing first {root_dims} qpos for root optimization")
    t_start = time.time()

    q0 = jp.copy(mjx_data.qpos[:])
    root_xyz = kp_data[frame, 3 * root_kp_idx : 3 * root_kp_idx + 3]
    q0 = q0.at[:3].set(root_xyz)
    joint_mask = jp.zeros_like(q0, dtype=bool).at[:root_dims].set(True)
    kp_mask = jp.repeat(trunk_kps, 3)

    joint_reg_weights = jp.zeros(mjx_model.nq)
    kp_frame = kp_data[frame, :][None]  # (1, n_kp_coords)
    q_init = q0[None]  # (1, nq)

    problem = build_q_opt_problem(
        1,
        mjx_model,
        mjx_data,
        joint_mask,
        kp_mask,
        lb,
        ub,
        site_idxs,
        kp_frame.shape[1],
        joint_reg_weights,
    )

    for _ in range(2):
        q_out = q_opt(
            problem,
            q_init,
            kp_frame,
            n_solver_max_iters=n_solver_max_iters,
            initial_step_damping=initial_step_damping,
        )
        q_solved = q_out[0]
        mjx_data = mjx_data.replace(qpos=q_solved)
        mjx_data = utils.kinematics(mjx_model, mjx_data)
        mjx_data = utils.com_pos(mjx_model, mjx_data)
        q_init = mjx_data.qpos[None]
        q_init = q_init.at[:, :3].set(root_xyz[None])

    print(f"Root optimization finished in {(time.time() - t_start) / 60:.2f} minutes")
    return mjx_data


def offset_optimization(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    offsets: Float[Array, "n_keypoints 3"],
    q: Float[Array, "n_frames n_qpos"],
    n_sample_frames: int,
    is_regularized: Float[Array, "n_keypoints 3"],
    site_idxs: Int[Array, " n_keypoints"],
    m_reg_coef: float,
) -> tuple[mjx.Model, mjx.Data, Float[Array, "n_keypoints 3"]]:
    """Optimize marker offsets based on proposed joint angles.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.
        kp_data: Flattened keypoint data.
        offsets: Current site offsets per keypoint.
        q: Proposed joint angles over all frames.
        n_sample_frames: Number of frames to sample for offset residual.
        is_regularized: 0/1 mask for regularized coordinates.
        site_idxs: Indices of marker sites.
        m_reg_coef: Regularization coefficient.

    Returns:
        Tuple of (updated model, updated data, optimized offsets).
    """
    key = jax.random.PRNGKey(0)

    all_indices = jp.arange(kp_data.shape[0])
    shuffled_indices = jax.random.permutation(key, all_indices, independent=True)
    time_indices = shuffled_indices[:n_sample_frames]

    s = time.time()
    print("Begining offset optimization:")

    keypoints = kp_data[time_indices, :]
    q = jp.take(q, time_indices, axis=0)

    res = stac_core.m_opt(
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
    print(f"Final residual error of {res.error}")

    mjx_model = utils.set_site_pos(mjx_model, offset_opt_param, site_idxs)
    mjx_data = utils.kinematics(mjx_model, mjx_data)

    print(f"Offset optimization finished in {time.time() - s} seconds")

    return mjx_model, mjx_data, offset_opt_param


def pose_optimization(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    lb: Float[Array, " n_qpos"],
    ub: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
    q_init: Float[Array, "n_frames n_qpos"],
    kp_mask: Bool[Array, " n_keypoints_xyz"] | None = None,
    joint_reg_weights: Float[Array, " n_qpos"] | None = None,
    acceleration_smoothness_weight: float = 0.0,
    n_solver_max_iters: int = 50,
    initial_step_damping: float = 1.0,
    problem: stac_core.QOptProblem | None = None,
) -> tuple[
    mjx.Data,
    Float[Array, "n_frames n_qpos"],
    Float[Array, "n_frames n_bodies 3"],
    Float[Array, "n_frames n_bodies 4"],
    Float[Array, "n_frames n_keypoints 3"],
    Float[Array, " n_frames"],
]:
    """Run pose optimization over an entire clip.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data (used as FK template; qpos is overwritten).
        kp_data: Observed keypoint positions, flattened xyz.
        lb: Joint lower bounds.
        ub: Joint upper bounds.
        site_idxs: Marker site indices.
        q_init: Warm-start joint angles.
        kp_mask: Boolean mask selecting which keypoint coordinates to fit
            (default: all True).
        joint_reg_weights: Per-joint regularization weights (default: zeros).
        acceleration_smoothness_weight: Temporal acceleration smoothness coupling.
        n_solver_max_iters: Maximum solver iterations.
        initial_step_damping: Initial damping on the solver step.
        problem: Pre-built QOptProblem (reuse across clips of same n_frames).

    Returns:
        Tuple of (final mjx_data, qpos, body_pos, body_quat, marker_pos,
        per-frame marker error).
    """
    t_start = time.time()
    n_frames = kp_data.shape[0]

    if kp_mask is None:
        kp_mask = jp.ones(kp_data.shape[1], dtype=bool)
    if joint_reg_weights is None:
        joint_reg_weights = jp.zeros(mjx_model.nq)

    joint_mask = jp.ones(mjx_model.nq, dtype=bool)

    if problem is None:
        problem = build_q_opt_problem(
            n_frames,
            mjx_model,
            mjx_data,
            joint_mask,
            kp_mask,
            lb,
            ub,
            site_idxs,
            kp_data.shape[1],
            joint_reg_weights,
            acceleration_smoothness_weight,
        )

    qpos = q_opt(
        problem,
        q_init,
        kp_data,
        n_solver_max_iters=n_solver_max_iters,
        initial_step_damping=initial_step_damping,
    )

    def fk_frame(q):
        fk_data = mjx_data.replace(qpos=q)
        fk_data = utils.kinematics(mjx_model, fk_data)
        fk_data = utils.com_pos(mjx_model, fk_data)
        return fk_data.xpos, fk_data.xquat, utils.get_site_xpos(fk_data, site_idxs)

    body_pos, body_quat, marker_pos = jax.vmap(fk_frame)(qpos)

    markers_flat = marker_pos.reshape(n_frames, -1)
    kp_flat = kp_data.reshape(n_frames, -1)
    marker_error = jp.sum((kp_flat - markers_flat) ** 2, axis=-1)

    mjx_data = mjx_data.replace(qpos=qpos[-1])
    mjx_data = utils.kinematics(mjx_model, mjx_data)
    mjx_data = utils.com_pos(mjx_model, mjx_data)

    print(f"Pose Optimization finished in {(time.time() - t_start) / 60.0:.2f} minutes")
    return mjx_data, qpos, body_pos, body_quat, marker_pos, marker_error
