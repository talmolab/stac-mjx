"""Compute stac optimization on data."""

import jax
import jax.numpy as jp
from jax import Array
import numpy as np
import mujoco
import time

from jaxtyping import Float, Int, Bool
from jaxtyping import jaxtyped
from beartype import beartype
from mujoco import mjx

from stac_mjx import stac_core
from stac_mjx import utils


@jaxtyped(typechecker=beartype)
def root_optimization(
    stac_core_obj: stac_core.StacCore,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    root_kp_idx: int,
    lb: Float[Array, " n_qpos"],
    ub: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
    trunk_kps: Bool[Array, " n_keypoints"],
    frame: int = 0,
) -> mjx.Data:
    """Optimize root DOFs for a single frame.

    Seeds root translation from the selected root keypoint, then runs
    two root-only q-phase solves using trunk keypoints as the objective.

    Args:
        stac_core_obj: Solver wrapper for q-phase optimization.
        mjx_model: MJX model.
        mjx_data: MJX data.
        kp_data: Flattened keypoint data.
        root_kp_idx: Index of root keypoint in the ordered keypoint list.
        lb: Lower bounds on joint angles.
        ub: Upper bounds on joint angles.
        site_idxs: Indices of marker sites.
        trunk_kps: Boolean mask selecting trunk keypoints.
        frame: Frame index to optimize.

    Returns:
        Updated MJX data after root optimization.
    """
    print(f"Root Optimization:")

    if mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE:  # better way to handle?
        root_dims = 4
    else:
        root_dims = 7
    print(f"Optimizing first {root_dims} qposes for root optimization")
    s = time.time()
    q0 = jp.copy(mjx_data.qpos[:])
    root_xyz = kp_data[frame, 3 * root_kp_idx : 3 * root_kp_idx + 3]
    q0 = q0.at[:3].set(root_xyz)
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
    q0 = q0.at[:3].set(root_xyz)

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


@jaxtyped(typechecker=beartype)
def offset_optimization(
    stac_core_obj: stac_core.StacCore,
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
        stac_core_obj: Solver wrapper.
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

    keypoints = jp.array(kp_data[time_indices, :])
    q = jp.take(q, time_indices, axis=0)

    res = stac_core_obj.m_opt(
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


@jaxtyped(typechecker=beartype)
def pose_optimization(
    stac_core_obj: stac_core.StacCore,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    lb: Float[Array, " n_qpos"],
    ub: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
    indiv_parts: list[Bool[Array, " n_qpos"]],
) -> tuple[
    mjx.Data,
    Float[Array, "n_frames n_qpos"],
    list,
    list,
    list,
    list[float],
    list,
]:
    """Run pose optimization over an entire clip.

    Args:
        stac_core_obj: Solver wrapper.
        mjx_model: MJX model.
        mjx_data: MJX data.
        kp_data: Flattened keypoint data.
        lb: Lower bounds on joint angles.
        ub: Upper bounds on joint angles.
        site_idxs: Indices of marker sites.
        indiv_parts: Per-part joint masks for individual part optimization.

    Returns:
        Tuple of (final mjx_data, qposes, xposes, xquats, marker_sites,
        per-frame times, per-frame errors).
    """
    s = time.time()
    qposes = []
    xposes = []
    xquats = []
    marker_sites = []

    frames = jp.arange(kp_data.shape[0])

    kps_to_opt = jp.ones(kp_data.shape[1], dtype=bool)
    qs_to_opt = jp.ones(mjx_model.nq, dtype=bool)
    print("Pose Optimization:")

    def f(mjx_data, kp_data, n_frame, parts):
        q0 = jp.copy(mjx_data.qpos[:])

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
