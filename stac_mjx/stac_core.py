"""Implementation of STAC for animal motion capture."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jp
import jaxlie
import jaxls
from jax import Array, jit
from jaxtyping import Bool, Float, Int
from mujoco import mjx

from stac_mjx import utils

_FREE_JOINT_NDOF = 7
_CG_TOLERANCE_MIN = 1e-7
_CG_TOLERANCE_MAX = 1e-3


class MOptResult(NamedTuple):
    """Result of marker offset optimization."""

    params: Float[Array, "n_keypoints 3"]
    error: Float[Array, ""]


class QOptResult(NamedTuple):
    """Optional telemetry result for q optimization."""

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

    site_bodyid = mjx_model.site_bodyid[site_idxs]

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


# ---------------------------------------------------------------------------
# Jaxls Levenberg-Marquardt pose optimizer
# ---------------------------------------------------------------------------


@dataclass
class QOptProblem:
    analyzed: jaxls.AnalyzedLeastSquaresProblem
    se3_mode: bool
    n_frames: int
    _KpVar: type
    _SE3Var: type | None = None
    _JointVar: type | None = None
    _QVar: type | None = None


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
    acceleration_smoothness_weight: float = 1.0,
) -> QOptProblem:
    """Build and analyze a jaxls least-squares pose problem.

    Call once per unique (n_frames, nq, n_kp_coords) shape.
    Reusable for any clip with the same shape.

    Uses SE3 manifold parameterization for the root when all seven
    free-joint DOFs are optimized, otherwise uses flat Euclidean.

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
        acceleration_smoothness_weight: Temporal acceleration smoothness coupling
            (0 = independent frames).

    Returns:
        Analyzed problem handle for ``q_opt``.
    """
    se3_mode = bool(jp.all(joint_mask[:_FREE_JOINT_NDOF]))

    if se3_mode:
        return _build_se3_problem(
            n_frames,
            mjx_model,
            mjx_data,
            joint_mask,
            kp_mask,
            joint_lb,
            joint_ub,
            site_idxs,
            n_kp_coords,
            joint_reg_weights,
            acceleration_smoothness_weight,
        )
    return _build_flat_problem(
        n_frames,
        mjx_model,
        mjx_data,
        joint_mask,
        kp_mask,
        joint_lb,
        joint_ub,
        site_idxs,
        n_kp_coords,
        joint_reg_weights,
        acceleration_smoothness_weight,
    )


def q_opt(
    problem: QOptProblem,
    q_init: Float[Array, "n_frames n_qpos"],
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    n_solver_max_iters: int = 50,
    initial_step_damping: float = 1.0,
    return_summary: bool = False,
) -> Float[Array, "n_frames n_qpos"] | QOptResult:
    """Solve the analyzed q optimization problem.

    Args:
        problem: Handle from ``build_q_opt_problem``.
        q_init: Warm-start joint angles.
        kp_data: Observed keypoint positions.
        n_solver_max_iters: Maximum solver iterations.
        initial_step_damping: Initial damping on the solver step.
        return_summary: Return qpos and jaxls SolveSummary for telemetry.

    Returns:
        Optimized joint angles, or qpos with SolveSummary when requested.
    """
    if kp_data.ndim == 3:
        kp_data = kp_data.reshape(kp_data.shape[0], -1)

    solver_cfg = dict(
        verbose=False,
        linear_solver=jaxls.ConjugateGradientConfig(
            tolerance_min=_CG_TOLERANCE_MIN,
            tolerance_max=_CG_TOLERANCE_MAX,
        ),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=initial_step_damping),
        termination=jaxls.TerminationConfig(max_iterations=n_solver_max_iters),
    )

    if problem.se3_mode:
        return _solve_se3(problem, q_init, kp_data, solver_cfg, return_summary)
    return _solve_flat(problem, q_init, kp_data, solver_cfg, return_summary)


def _build_se3_problem(
    n_frames,
    mjx_model,
    mjx_data,
    joint_mask,
    kp_mask,
    joint_lb,
    joint_ub,
    site_idxs,
    n_kp_coords,
    joint_reg_weights,
    acceleration_smoothness_weight,
):
    n_hinges = int(mjx_model.nq) - _FREE_JOINT_NDOF
    joint_default = jp.zeros((n_hinges,))
    kp_default = jp.zeros((n_kp_coords,))

    SE3Var = jaxls.SE3Var

    class JointVar(jaxls.Var[jp.ndarray], default_factory=lambda: joint_default): ...

    class KpVar(jaxls.Var[jp.ndarray], default_factory=lambda: kp_default): ...

    frame_ids = jp.arange(n_frames)
    root_vars, joint_vars, kp_vars = (
        SE3Var(frame_ids),
        JointVar(frame_ids),
        KpVar(frame_ids),
    )
    costs: list[jaxls.Cost] = []

    @jaxls.Cost.factory
    def marker_cost(
        vals: jaxls.VarValues, root: SE3Var, joints: JointVar, kp: KpVar
    ) -> jp.ndarray:
        se3 = vals[root]
        hinge_angles = vals[joints]
        obs = jax.lax.stop_gradient(vals[kp])
        q = jp.concatenate([se3.translation(), se3.rotation().wxyz, hinge_angles])
        full_q = jp.where(joint_mask, q, mjx_data.qpos)
        fk_data = mjx_data.replace(qpos=full_q)
        fk_data = utils.kinematics(mjx_model, fk_data)
        fk_data = utils.com_pos(mjx_model, fk_data)
        return (obs - utils.get_site_xpos(fk_data, site_idxs).flatten()) * kp_mask

    costs.append(marker_cost(root_vars, joint_vars, kp_vars))

    hinge_reg_weights = joint_reg_weights[_FREE_JOINT_NDOF:]
    hinge_mask = joint_mask[_FREE_JOINT_NDOF:]
    if jp.any(hinge_reg_weights > 0):

        @jaxls.Cost.factory
        def reg_cost(vals: jaxls.VarValues, joints: JointVar) -> jp.ndarray:
            return jp.sqrt(hinge_reg_weights * hinge_mask) * vals[joints]

        costs.append(reg_cost(joint_vars))

    hinge_lb, hinge_ub = joint_lb[_FREE_JOINT_NDOF:], joint_ub[_FREE_JOINT_NDOF:]

    @jaxls.Cost.factory(kind="constraint_leq_zero")
    def limit_cost(vals: jaxls.VarValues, joints: JointVar) -> jp.ndarray:
        angles = vals[joints]
        return jp.concatenate([hinge_lb - angles, angles - hinge_ub])

    costs.append(limit_cost(joint_vars))

    if acceleration_smoothness_weight > 0.0 and n_frames > 2:

        @jaxls.Cost.factory
        def smooth_acceleration_cost(
            vals: jaxls.VarValues,
            root_prev: SE3Var,
            root_cur: SE3Var,
            root_next: SE3Var,
            joint_prev: JointVar,
            joint_cur: JointVar,
            joint_next: JointVar,
        ) -> jp.ndarray:
            root_prev_inc = (vals[root_prev].inverse() @ vals[root_cur]).log()
            root_next_inc = (vals[root_cur].inverse() @ vals[root_next]).log()
            root_acc = root_next_inc - root_prev_inc
            joint_acc = vals[joint_next] - 2.0 * vals[joint_cur] + vals[joint_prev]
            return (
                jp.concatenate([root_acc, joint_acc]) * acceleration_smoothness_weight
            )

        costs.append(
            smooth_acceleration_cost(
                SE3Var(jp.arange(n_frames - 2)),
                SE3Var(jp.arange(1, n_frames - 1)),
                SE3Var(jp.arange(2, n_frames)),
                JointVar(jp.arange(n_frames - 2)),
                JointVar(jp.arange(1, n_frames - 1)),
                JointVar(jp.arange(2, n_frames)),
            )
        )

    analyzed = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=[root_vars, joint_vars, kp_vars],
    ).analyze()

    return QOptProblem(
        analyzed=analyzed,
        se3_mode=True,
        n_frames=n_frames,
        _KpVar=KpVar,
        _SE3Var=SE3Var,
        _JointVar=JointVar,
    )


def _solve_se3(problem, q_init, kp_data, solver_cfg, return_summary):
    n_frames = problem.n_frames
    SE3Var, JointVar, KpVar = problem._SE3Var, problem._JointVar, problem._KpVar

    xyz = q_init[:, :3]
    wxyz = q_init[:, 3:7]
    hinges = q_init[:, _FREE_JOINT_NDOF:]

    quat_norm = jp.linalg.norm(wxyz, axis=-1, keepdims=True)
    wxyz = wxyz / jp.where(quat_norm > 0, quat_norm, 1.0)
    se3_init = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz=wxyz), xyz)

    frame_ids = jp.arange(n_frames)
    solve_out = problem.analyzed.solve(
        initial_vals=jaxls.VarValues.make(
            [
                SE3Var(frame_ids).with_value(se3_init),
                JointVar(frame_ids).with_value(hinges),
                KpVar(frame_ids).with_value(kp_data),
            ]
        ),
        return_summary=return_summary,
        **solver_cfg,
    )
    if return_summary:
        sol, summary = solve_out
    else:
        sol, summary = solve_out, None

    roots = sol[SE3Var(frame_ids)]
    joints = sol[JointVar(frame_ids)]
    qpos = jp.concatenate([roots.translation(), roots.rotation().wxyz, joints], axis=-1)
    if return_summary:
        return QOptResult(qpos, summary)
    return qpos


def _build_flat_problem(
    n_frames,
    mjx_model,
    mjx_data,
    joint_mask,
    kp_mask,
    joint_lb,
    joint_ub,
    site_idxs,
    n_kp_coords,
    joint_reg_weights,
    acceleration_smoothness_weight,
):
    nq = int(mjx_model.nq)
    qpos_default = jp.zeros((nq,))
    kp_default = jp.zeros((n_kp_coords,))

    class QVar(jaxls.Var[jp.ndarray], default_factory=lambda: qpos_default): ...

    class KpVar(jaxls.Var[jp.ndarray], default_factory=lambda: kp_default): ...

    frame_ids = jp.arange(n_frames)
    qpos_vars, kp_vars = QVar(frame_ids), KpVar(frame_ids)

    costs: list[jaxls.Cost] = []

    @jaxls.Cost.factory
    def marker_cost(vals: jaxls.VarValues, q: QVar, kp: KpVar) -> jp.ndarray:
        obs = jax.lax.stop_gradient(vals[kp])
        full_q = jp.where(joint_mask, vals[q], mjx_data.qpos)
        fk_data = mjx_data.replace(qpos=full_q)
        fk_data = utils.kinematics(mjx_model, fk_data)
        fk_data = utils.com_pos(mjx_model, fk_data)
        return (obs - utils.get_site_xpos(fk_data, site_idxs).flatten()) * kp_mask

    costs.append(marker_cost(qpos_vars, kp_vars))

    if jp.any(joint_reg_weights > 0):

        @jaxls.Cost.factory
        def reg_cost(vals: jaxls.VarValues, q: QVar) -> jp.ndarray:
            return jp.sqrt(joint_reg_weights * joint_mask) * vals[q]

        costs.append(reg_cost(qpos_vars))

    @jaxls.Cost.factory(kind="constraint_leq_zero")
    def limit_cost(vals: jaxls.VarValues, q: QVar) -> jp.ndarray:
        q_val = vals[q]
        return jp.concatenate([joint_lb - q_val, q_val - joint_ub])

    costs.append(limit_cost(qpos_vars))

    if acceleration_smoothness_weight > 0.0 and n_frames > 2:

        @jaxls.Cost.factory
        def smooth_acceleration_cost(
            vals: jaxls.VarValues, q_prev: QVar, q_cur: QVar, q_next: QVar
        ) -> jp.ndarray:
            return (
                vals[q_next] - 2.0 * vals[q_cur] + vals[q_prev]
            ) * acceleration_smoothness_weight

        costs.append(
            smooth_acceleration_cost(
                QVar(jp.arange(n_frames - 2)),
                QVar(jp.arange(1, n_frames - 1)),
                QVar(jp.arange(2, n_frames)),
            )
        )

    analyzed = jaxls.LeastSquaresProblem(
        costs=costs, variables=[qpos_vars, kp_vars]
    ).analyze()

    return QOptProblem(
        analyzed=analyzed,
        se3_mode=False,
        n_frames=n_frames,
        _KpVar=KpVar,
        _QVar=QVar,
    )


def _solve_flat(problem, q_init, kp_data, solver_cfg, return_summary):
    n_frames = problem.n_frames
    QVar, KpVar = problem._QVar, problem._KpVar

    frame_ids = jp.arange(n_frames)
    solve_out = problem.analyzed.solve(
        initial_vals=jaxls.VarValues.make(
            [
                QVar(frame_ids).with_value(q_init),
                KpVar(frame_ids).with_value(kp_data),
            ]
        ),
        return_summary=return_summary,
        **solver_cfg,
    )
    if return_summary:
        sol, summary = solve_out
    else:
        sol, summary = solve_out, None

    qpos = sol[QVar(frame_ids)]
    quat = qpos[:, 3:7]
    quat_norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
    qpos = qpos.at[:, 3:7].set(quat / jp.where(quat_norm > 0, quat_norm, 1.0))
    if return_summary:
        return QOptResult(qpos, summary)
    return qpos
