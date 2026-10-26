"""Implementation of STAC for animal motion capture."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp
import jaxlie
import jaxls
import mujoco
import numpy as np
from jax import Array, jit
from jaxtyping import Bool, Float, Int
from mujoco import mjx

from stac_mjx import utils
from stac_mjx.joints import JointLayout, normalize_quaternions

_FREE_JOINT_NQ = 7
_CG_TOLERANCE_MIN = 1e-7
_CG_TOLERANCE_MAX = 1e-3


class MOptResult(NamedTuple):
    """Result of marker offset optimization."""

    params: Float[Array, "n_keypoints 3"]
    error: Float[Array, ""]


class QOptResult(NamedTuple):
    """Optimized joint poses and solver diagnostics."""

    qpos: Float[Array, "n_frames n_qpos"]
    summary: jaxls.SolveSummary


@jit
def m_opt(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    keypoints: Float[Array, "n_sample_frames n_keypoints_xyz"],
    q: Float[Array, "n_sample_frames n_qpos"],
    initial_offsets: Float[Array, "n_keypoints 3"],
    is_regularized: Float[Array, "n_keypoints 3"],
    reg_coef: float,
    site_idxs: Int[Array, " n_keypoints"],
) -> MOptResult:
    """Closed-form marker offset solve.

    Exactly solves the quadratic marker-offset objective, assuming
    standard MuJoCo/MJX site kinematics (site_quat = identity):

        min_m  sum_t || y_t - (p_t + R_t m) ||^2
             + reg_coef * || D (m - m0) ||^2

    where y_t are observed markers, m are local site offsets, p_t/R_t are the
    body translation/rotation at frame t, D is the diagonal regularization
    mask, and T is the number of sampled frames.

    Uses `jax.vmap` over frames for parallel FK evaluation.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.
        keypoints: Sampled observed marker positions, flattened xyz.
        q: Fixed pose trajectory for sampled frames.
        initial_offsets: Reference offsets per keypoint.
        is_regularized: 0/1 mask for regularized coordinates.
        reg_coef: Regularization coefficient.
        site_idxs: Indices of the optimized sites.

    Returns:
        Optimized offsets and scalar residual loss.
    """
    T = keypoints.shape[0]
    K = site_idxs.shape[0]

    y = keypoints.reshape(T, K, 3)
    d = is_regularized.astype(y.dtype)

    site_bodyid = jp.array(mjx_model.site_bodyid)[site_idxs]

    def fk_single(q_t):
        """Run FK for a single frame, return body pos and rot for sites."""
        data_t = mjx_data.replace(qpos=q_t)
        data_t = utils.kinematics(mjx_model, data_t)
        data_t = utils.com_pos(mjx_model, data_t)
        return data_t.xpos[site_bodyid], data_t.xmat[site_bodyid]

    p_all, R_all = jax.vmap(fk_single)(q)  # (T, K, 3), (T, K, 3, 3)
    z_all = y - p_all  # (T, K, 3)
    s = jp.einsum(
        "tkji,tkj->ki", R_all, z_all
    )  # s_k = sum_t R_{t,k}^T z_{t,k}  ->  (K, 3)
    z2 = jp.sum(z_all**2)  # z2 = sum_t ||z_t||^2  ->  scalar

    # Closed-form solve, coordinate-wise
    denom = T + reg_coef * d
    numer = s + reg_coef * d * initial_offsets
    m_star = numer / denom

    # Residual error at the solution
    data_term = z2 - 2.0 * jp.sum(m_star * s) + T * jp.sum(m_star**2)
    reg_term = reg_coef * jp.sum((d * (m_star - initial_offsets)) ** 2)
    error = data_term + reg_term

    return MOptResult(params=m_star, error=error)


@dataclass
class QOptProblem:
    """Compiled jaxls least-squares problem for pose optimization."""

    analyzed: jaxls.AnalyzedLeastSquaresProblem
    n_frames: int
    n_kp_coords: int
    site_offsets: Float[Array, "n_keypoints 3"]
    dynamic_site_offsets: bool
    layout: JointLayout
    joint_mask: Array
    qpos_reference: Array
    _ObservedKeypointsVar: type
    _JointPoseVar: type
    _RootPoseVar: type | None = None
    _MarkerOffsetsVar: type | None = None


def build_q_opt_problem(
    n_frames: int,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    joint_mask: Bool[Array, " n_qpos"],
    kp_mask: Bool[Array, " n_keypoints_xyz"],
    joint_lb: Float[Array, " n_qpos"],
    joint_ub: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
    n_kp_coords: int,
    joint_reg_weights: Float[Array, " n_qpos"],
    velocity_smoothness_weight: float = 0.0,
    site_offsets: Float[Array, "n_keypoints 3"] | None = None,
    dynamic_site_offsets: bool = False,
) -> QOptProblem:
    """Build and analyze a jaxls least-squares pose problem.

    Call once per unique (n_frames, nq, n_kp_coords) shape.
    Reusable for any clip with the same shape.

    Uses one optimizer with SE(3) updates for a fully optimized free root
    and SO(3) updates for other quaternion joints.

    Args:
        n_frames: Number of timesteps.
        mjx_model: MJX model (constant across frames).
        mjx_data: MJX data used as FK template.
        joint_mask: Boolean mask selecting which joints to optimize.
        kp_mask: Boolean mask selecting which keypoint coordinates to fit.
        joint_lb: Joint lower bounds.
        joint_ub: Joint upper bounds.
        site_idxs: Marker site indices.
        n_kp_coords: Total keypoint coordinate count (n_keypoints * 3).
        joint_reg_weights: Per-joint L2 regularization weights.
        velocity_smoothness_weight: Temporal velocity smoothness coupling
            (0 = independent frames).
        site_offsets: Current marker offsets. Defaults to model site positions.
        dynamic_site_offsets: Pass marker offsets as solve data rather than
            baking them into the analyzed problem.

    Returns:
        Analyzed problem handle for ``q_opt``.
    """
    free_root = (
        int(mjx_model.njnt) > 0
        and int(mjx_model.jnt_type[0]) == mujoco.mjtJoint.mjJNT_FREE
        and bool(jp.all(joint_mask[:_FREE_JOINT_NQ]))
    )
    if site_offsets is None:
        site_offsets = mjx_model.site_pos[site_idxs]

    layout = JointLayout(mjx_model.jnt_type)
    qpos_offset = _FREE_JOINT_NQ if free_root else 0
    joint_layout = JointLayout(mjx_model.jnt_type[1:]) if free_root else layout
    active_joints = joint_mask[qpos_offset:]
    # A fully optimized free root has no posture penalty or scalar bounds.
    reg_weights = joint_reg_weights[qpos_offset:] * active_joints
    site_bodyid = jp.array(mjx_model.site_bodyid)[site_idxs]
    observed_keypoints_default = jp.zeros((n_kp_coords,))

    joint_layout.validate_mask(active_joints)

    def retract(qpos, delta):
        return jp.where(active_joints, joint_layout.retract(qpos, delta), qpos)

    class JointPoseVar(
        jaxls.Var[jp.ndarray],
        default_factory=joint_layout.identity,
        retract_fn=retract,
        tangent_dim=joint_layout.nv,
    ): ...

    def get_qpos(values, joint_pose, root_pose):
        if root_pose is None:
            return values[joint_pose]
        pose = values[root_pose]
        return jp.concatenate(
            [pose.translation(), pose.rotation().wxyz, values[joint_pose]]
        )

    class ObservedKeypointsVar(
        jaxls.Var[jp.ndarray], default_factory=lambda: observed_keypoints_default
    ): ...

    MarkerOffsetsVar, marker_offsets_vars = None, None
    if dynamic_site_offsets:
        marker_offsets_default = jp.zeros((site_offsets.size,))

        class MarkerOffsetsVar(
            jaxls.Var[jp.ndarray], default_factory=lambda: marker_offsets_default
        ): ...

        marker_offsets_vars = MarkerOffsetsVar(jp.arange(n_frames))

    joint_pose_vars, observed_keypoints_vars = (
        JointPoseVar(jp.arange(n_frames)),
        ObservedKeypointsVar(jp.arange(n_frames)),
    )
    root_pose_vars = jaxls.SE3Var(jp.arange(n_frames)) if free_root else None
    costs: list[jaxls.Cost] = []

    if dynamic_site_offsets:

        @jaxls.Cost.factory
        def marker_cost(
            values: jaxls.VarValues,
            joint_pose: JointPoseVar,
            observed_keypoints: ObservedKeypointsVar,
            root_pose: jaxls.SE3Var | None,
            marker_offsets: MarkerOffsetsVar,
        ) -> jp.ndarray:
            observations = jax.lax.stop_gradient(values[observed_keypoints])
            # Offsets are per-solve data, not optimized state.
            marker_offset_values = jax.lax.stop_gradient(
                values[marker_offsets]
            ).reshape((-1, 3))
            full_q = jp.where(
                joint_mask, get_qpos(values, joint_pose, root_pose), mjx_data.qpos
            )
            fk_data = mjx_data.replace(qpos=full_q)
            fk_data = utils.kinematics(mjx_model, fk_data)
            fk_data = utils.com_pos(mjx_model, fk_data)
            marker_pos = fk_data.xpos[site_bodyid] + jp.einsum(
                "kij,kj->ki", fk_data.xmat[site_bodyid], marker_offset_values
            )
            return (observations - marker_pos.flatten()) * kp_mask

        costs.append(
            marker_cost(
                joint_pose_vars,
                observed_keypoints_vars,
                root_pose_vars,
                marker_offsets_vars,
            )
        )
    else:

        @jaxls.Cost.factory
        def marker_cost(
            values: jaxls.VarValues,
            joint_pose: JointPoseVar,
            observed_keypoints: ObservedKeypointsVar,
            root_pose: jaxls.SE3Var | None,
        ) -> jp.ndarray:
            observations = jax.lax.stop_gradient(values[observed_keypoints])
            full_q = jp.where(
                joint_mask, get_qpos(values, joint_pose, root_pose), mjx_data.qpos
            )
            fk_data = mjx_data.replace(qpos=full_q)
            fk_data = utils.kinematics(mjx_model, fk_data)
            fk_data = utils.com_pos(mjx_model, fk_data)
            markers = utils.get_site_xpos(fk_data, site_idxs).flatten()
            return (observations - markers) * kp_mask

        costs.append(
            marker_cost(joint_pose_vars, observed_keypoints_vars, root_pose_vars)
        )

    if jp.any(reg_weights > 0):

        @jaxls.Cost.factory
        def reg_cost(values: jaxls.VarValues, joint_pose: JointPoseVar) -> jp.ndarray:
            return joint_layout.regularization(values[joint_pose], reg_weights)

        costs.append(reg_cost(joint_pose_vars))

    lb, ub = joint_lb[qpos_offset:], joint_ub[qpos_offset:]
    scalar = joint_layout.scalar_qpos
    active = np.asarray(active_joints)
    lower_idx = scalar[active[scalar] & np.isfinite(np.asarray(lb)[scalar])]
    upper_idx = scalar[active[scalar] & np.isfinite(np.asarray(ub)[scalar])]
    ball_joints = [
        i
        for i, joint_type in enumerate(mjx_model.jnt_type)
        if joint_type == mujoco.mjtJoint.mjJNT_BALL
        and mjx_model.jnt_limited[i]
        and int(mjx_model.jnt_qposadr[i]) >= qpos_offset
        and active[int(mjx_model.jnt_qposadr[i]) - qpos_offset]
    ]
    ball_qpos = np.asarray(
        [
            np.arange(4) + int(mjx_model.jnt_qposadr[i]) - qpos_offset
            for i in ball_joints
        ],
        dtype=np.int32,
    ).reshape(-1, 4)
    ball_max_angle = mjx_model.jnt_range[np.asarray(ball_joints, dtype=np.int32), 1]
    if lower_idx.size or upper_idx.size or ball_qpos.size:

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(values: jaxls.VarValues, joint_pose: JointPoseVar) -> jp.ndarray:
            qpos = values[joint_pose]
            parts = [lb[lower_idx] - qpos[lower_idx], qpos[upper_idx] - ub[upper_idx]]
            if ball_qpos.size:
                rotation = jaxlie.SO3(normalize_quaternions(qpos[ball_qpos])).log()
                parts.append(jp.sum(rotation**2, axis=-1) - ball_max_angle**2)
            return jp.concatenate(parts)

        costs.append(limit_cost(joint_pose_vars))

    if velocity_smoothness_weight > 0.0 and n_frames > 1:

        @jaxls.Cost.factory
        def smooth_velocity_cost(
            values: jaxls.VarValues,
            joint_pose_prev: JointPoseVar,
            joint_pose_cur: JointPoseVar,
            root_pose_prev: jaxls.SE3Var | None,
            root_pose_cur: jaxls.SE3Var | None,
        ) -> jp.ndarray:
            displacement = joint_layout.difference(
                values[joint_pose_prev], values[joint_pose_cur]
            )
            if free_root:
                root_displacement = (
                    values[root_pose_prev].inverse() @ values[root_pose_cur]
                ).log()
                displacement = jp.concatenate([root_displacement, displacement])
            return displacement * velocity_smoothness_weight

        costs.append(
            smooth_velocity_cost(
                JointPoseVar(jp.arange(n_frames - 1)),
                JointPoseVar(jp.arange(1, n_frames)),
                jaxls.SE3Var(jp.arange(n_frames - 1)) if free_root else None,
                jaxls.SE3Var(jp.arange(1, n_frames)) if free_root else None,
            )
        )

    problem_vars = [joint_pose_vars, observed_keypoints_vars]
    if free_root:
        problem_vars.insert(0, root_pose_vars)
    if dynamic_site_offsets:
        problem_vars.append(marker_offsets_vars)

    analyzed = jaxls.LeastSquaresProblem(costs=costs, variables=problem_vars).analyze()

    return QOptProblem(
        analyzed=analyzed,
        n_frames=n_frames,
        n_kp_coords=n_kp_coords,
        site_offsets=site_offsets,
        dynamic_site_offsets=dynamic_site_offsets,
        layout=layout,
        joint_mask=joint_mask,
        qpos_reference=mjx_data.qpos,
        _ObservedKeypointsVar=ObservedKeypointsVar,
        _MarkerOffsetsVar=MarkerOffsetsVar,
        _JointPoseVar=JointPoseVar,
        _RootPoseVar=jaxls.SE3Var if free_root else None,
    )


def q_opt(
    problem: QOptProblem,
    q_init: Float[Array, "n_frames n_qpos"],
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    n_solver_max_iters: int = 150,
    initial_step_damping: float = 1.0,
    site_offsets: Float[Array, "n_keypoints 3"] | None = None,
) -> QOptResult:
    """Solve the analyzed q optimization problem.

    Args:
        problem: Handle from ``build_q_opt_problem``.
        q_init: Warm-start joint angles.
        kp_data: Observed keypoint positions.
        n_solver_max_iters: Maximum solver iterations.
        initial_step_damping: Initial damping on the solver step.
        site_offsets: Marker offsets for this solve.

    Returns:
        Optimized joint poses and the JAXLS solve summary.
    """
    if kp_data.ndim == 3:
        kp_data = kp_data.reshape(kp_data.shape[0], -1)
    q_init = problem.layout.normalize(jp.asarray(q_init))
    q_init = jp.where(problem.joint_mask, q_init, problem.qpos_reference)
    if site_offsets is None:
        site_offsets = problem.site_offsets
    marker_offsets_data = None
    if problem.dynamic_site_offsets:
        marker_offsets_data = jp.broadcast_to(
            site_offsets.reshape(1, -1), (kp_data.shape[0], site_offsets.size)
        )

    solver_cfg = dict(
        verbose=False,
        linear_solver=jaxls.ConjugateGradientConfig(
            tolerance_min=_CG_TOLERANCE_MIN,
            tolerance_max=_CG_TOLERANCE_MAX,
        ),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=initial_step_damping),
        termination=jaxls.TerminationConfig(max_iterations=n_solver_max_iters),
    )

    JointPoseVar, ObservedKeypointsVar = (
        problem._JointPoseVar,
        problem._ObservedKeypointsVar,
    )
    frame_ids = jp.arange(problem.n_frames)
    qpos_offset = _FREE_JOINT_NQ if problem._RootPoseVar is not None else 0
    initial_values = [
        JointPoseVar(frame_ids).with_value(q_init[:, qpos_offset:]),
        ObservedKeypointsVar(frame_ids).with_value(kp_data),
    ]
    if problem._RootPoseVar is not None:
        quat = q_init[:, 3:7]
        quat_norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
        root_init = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(quat / jp.where(quat_norm > 0, quat_norm, 1.0)), q_init[:, :3]
        )
        initial_values.insert(0, problem._RootPoseVar(frame_ids).with_value(root_init))
    if problem.dynamic_site_offsets:
        initial_values.append(
            problem._MarkerOffsetsVar(frame_ids).with_value(marker_offsets_data)
        )

    sol, summary = problem.analyzed.solve(
        initial_vals=jaxls.VarValues.make(initial_values),
        return_summary=True,
        **solver_cfg,
    )

    qpos = sol[JointPoseVar(frame_ids)]
    if problem._RootPoseVar is not None:
        roots = sol[problem._RootPoseVar(frame_ids)]
        qpos = jp.concatenate(
            [roots.translation(), roots.rotation().wxyz, qpos], axis=-1
        )
    else:
        qpos = problem.layout.normalize(qpos)
    return QOptResult(qpos, summary)
