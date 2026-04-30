"""Implementation of STAC for animal motion capture."""

import jax
import jax.numpy as jp
from jax import Array
from jax import jit

from functools import partial
from typing import NamedTuple
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box
from jaxtyping import Float, Int, Bool
from jaxtyping import jaxtyped
from beartype import beartype
from mujoco import mjx

from stac_mjx import utils


class MOptResult(NamedTuple):
    """Result of marker offset optimization."""

    params: Float[Array, "n_keypoints 3"]
    error: Float[Array, ""]


def q_loss(
    q: Float[Array, " n_qpos"],
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    kp_data: Float[Array, " n_keypoints_xyz"],
    qs_to_opt: Bool[Array, " n_qpos"],
    kps_to_opt: Bool[Array, " n_keypoints_xyz"],
    initial_q: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
) -> float:
    """Compute marker loss for pose optimization.

    Args:
        q: Proposed joint angles.
        mjx_model: MJX model (constant).
        mjx_data: MJX data (modified for FK).
        kp_data: Ground truth keypoint positions.
        qs_to_opt: Mask selecting which joints to optimize.
        kps_to_opt: Mask selecting which keypoints contribute to loss.
        initial_q: Starting joint angles for reference.
        site_idxs: Indices of marker sites.

    Returns:
        Sum of squared residuals.
    """
    mjx_data = mjx_data.replace(qpos=utils.make_qs(initial_q, qs_to_opt, q))

    mjx_data = utils.kinematics(mjx_model, mjx_data)
    mjx_data = utils.com_pos(mjx_model, mjx_data)

    markers = utils.get_site_xpos(mjx_data, site_idxs).flatten()
    residual = kp_data - markers

    residual = residual * kps_to_opt
    residual = jp.sum(jp.square(residual))

    return residual


@partial(jit, static_argnames=["q_solver"])
def _q_opt(
    q_solver: ProjectedGradient,
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    marker_ref_arr: Float[Array, "n_keypoints_xyz n_frames"],
    qs_to_opt: Bool[Array, " n_qpos"],
    kps_to_opt: Bool[Array, " n_keypoints_xyz"],
    q0: Float[Array, " n_qpos"],
    lb: Float[Array, " n_qpos"],
    ub: Float[Array, " n_qpos"],
    site_idxs: Int[Array, " n_keypoints"],
) -> tuple[mjx.Data, object | None]:
    """Update q_pose using estimated marker parameters."""
    try:
        return mjx_data, q_solver.run(
            q0,
            hyperparams_proj=jp.array((lb, ub)),
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            kp_data=marker_ref_arr.T,
            qs_to_opt=qs_to_opt,
            kps_to_opt=kps_to_opt,
            initial_q=q0,
            site_idxs=site_idxs,
        )

    except ValueError as ex:
        print("Warning: optimization failed.", flush=True)
        print(ex, flush=True)
        mjx_data = mjx_data.replace(qpos=q0)
        mjx_data = utils.kinematics(mjx_model, mjx_data)

    return mjx_data, None


@jit
def _m_opt(
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


class StacCore:
    """Pose and offset optimization core.

    Pose optimization uses a ProjectedGradient solver.
    Offset optimization uses a closed-form solver.
    """

    def __init__(self, tol: float = 1e-5, n_iter_q: int = 400):
        """Initialize StacCore.

        Args:
            tol: Convergence tolerance for ProjectedGradient solver.
            n_iter_q: Maximum iterations for pose optimization.
        """
        self.q_solver = ProjectedGradient(
            fun=q_loss, projection=projection_box, maxiter=n_iter_q, tol=tol
        )

    @jaxtyped(typechecker=beartype)
    def q_opt(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        marker_ref_arr: Float[Array, "n_keypoints_xyz n_frames"],
        qs_to_opt: Bool[Array, " n_qpos"],
        kps_to_opt: Bool[Array, " n_keypoints_xyz"],
        q0: Float[Array, " n_qpos"],
        lb: Float[Array, " n_qpos"],
        ub: Float[Array, " n_qpos"],
        site_idxs: Int[Array, " n_keypoints"],
    ) -> tuple[mjx.Data, object | None]:
        """Update joint angles using estimated marker parameters.

        Wrapper for `_q_opt`.

        Args:
            mjx_model: MJX model.
            mjx_data: MJX data.
            marker_ref_arr: Marker reference positions (transposed layout).
            qs_to_opt: Mask selecting which joints to optimize.
            kps_to_opt: Mask selecting which keypoints contribute to loss.
            q0: Initial joint angles.
            lb: Lower bounds on joint angles.
            ub: Upper bounds on joint angles.
            site_idxs: Indices of marker sites.

        Returns:
            Tuple of (updated MJX data, optimization result or None on failure).
        """
        return _q_opt(
            self.q_solver,
            mjx_model,
            mjx_data,
            marker_ref_arr,
            qs_to_opt,
            kps_to_opt,
            q0,
            lb,
            ub,
            site_idxs,
        )

    @jaxtyped(typechecker=beartype)
    def m_opt(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        keypoints: Float[Array, "n_sample_frames n_keypoints_xyz"],
        q: Float[Array, "n_sample_frames n_qpos"],
        initial_offsets: Float[Array, "n_keypoints 3"],
        is_regularized: Float[Array, "n_keypoints 3"],
        reg_coef: float,
        site_idxs: Int[Array, " n_keypoints"],
    ) -> MOptResult:
        """Compute marker offset optimization.

        Wrapper for `_m_opt`.

        Args:
            mjx_model: MJX model.
            mjx_data: MJX data.
            keypoints: Sampled observed marker positions, flattened xyz.
            q: Fixed pose trajectory for sampled frames.
            initial_offsets: Reference offsets per keypoint.
            is_regularized: 0/1 mask for regularized coordinates.
            reg_coef: Regularization coefficient.
            site_idxs: Indices of marker sites.

        Returns:
            Optimized offsets and scalar residual loss.
        """
        return _m_opt(
            mjx_model,
            mjx_data,
            keypoints,
            q,
            initial_offsets,
            is_regularized,
            reg_coef,
            site_idxs,
        )
