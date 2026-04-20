"""Implementation of STAC for animal motion capture."""

import jax
import jax.numpy as jp
from jax import jit

from functools import partial
from typing import NamedTuple
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

from stac_mjx import utils


class MOptResult(NamedTuple):
    params: jp.ndarray
    error: jp.ndarray


def q_loss(
    q: jp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    qs_to_opt: jp.ndarray,
    kps_to_opt: jp.ndarray,
    initial_q: jp.ndarray,
    site_idxs: jp.ndarray,
) -> float:
    """Compute the marker loss for q_phase optimization.

    Args:
        q (jp.ndarray): Proposed qs
        mjx_model (mjx.Model): Model object (stays constant)
        mjx_data (mjx.Data): Data object (modified to calculate new xpos)
        kp_data (jp.ndarray): Ground truth keypoint positions
        qs_to_opt (jp.ndarray): Boolean array; for each index in qpos, True = q and False = initial_q when calculating residual
        kps_to_opt (jp.ndarray): Boolean array; only return residuals for the True positions
        initial_q (jp.ndarray): Starting qs for reference

    Returns:
        float: sum of squares scalar loss
    """
    # Replace qpos with new qpos with q and initial_q, based on qs_to_opt
    mjx_data = mjx_data.replace(qpos=utils.make_qs(initial_q, qs_to_opt, q))

    # Clip to bounds ourselves because of potential jaxopt bug
    # mjx_data = mjx_data.replace(qpos=jp.clip(op_utils.make_qs(initial_q, qs_to_opt, q), utils.params['lb'], utils.params['ub']))

    # Forward kinematics
    mjx_data = utils.kinematics(mjx_model, mjx_data)
    mjx_data = utils.com_pos(mjx_model, mjx_data)

    # Get marker site xpos
    markers = utils.get_site_xpos(mjx_data, site_idxs).flatten()
    residual = kp_data - markers

    # Set irrelevant body sites to 0
    residual = residual * kps_to_opt
    residual = jp.sum(jp.square(residual))

    return residual



@partial(jit, static_argnames=["q_solver"])
def _q_opt(
    q_solver,
    mjx_model,
    mjx_data,
    marker_ref_arr: jp.ndarray,
    qs_to_opt: jp.ndarray,
    kps_to_opt: jp.ndarray,
    q0: jp.ndarray,
    lb,
    ub,
    site_idxs,
):
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


@partial(jit, static_argnames=())
def _m_opt(
    mjx_model,
    mjx_data,
    keypoints: jp.ndarray,
    q: jp.ndarray,
    initial_offsets: jp.ndarray,
    is_regularized: jp.ndarray,
    reg_coef: float,
    site_idxs: jp.ndarray,
):
    """Closed-form marker offset solve.

    Exactly solves the quadratic marker-offset objective, assuming
    standard MuJoCo/MJX site kinematics (site_quat = identity):

        min_m  sum_t || y_t - (p_t + R_t m) ||^2
             + reg_coef * T * || D (m - m0) ||^2

    where y_t are observed markers, m are local site offsets, p_t/R_t are the
    body translation/rotation at frame t, D is the diagonal regularization
    mask, and T is the number of sampled frames.

    Uses `jax.vmap` over frames for parallel FK evaluation.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.
        keypoints: Array of shape (T, 3*K), sampled observed markers.
        q: Array of shape (T, nq), fixed pose trajectory for sampled frames.
        initial_offsets: Array of shape (3*K,), reference offsets.
        is_regularized: Array of shape (3*K,), 0/1 mask for regularized coords.
        reg_coef: Scalar regularization coefficient.
        site_idxs: Array of shape (K,), indices of the optimized sites.

    Returns:
        MOptResult: NamedTuple with `.params` (optimized flattened offsets)
        and `.error` (scalar residual loss at the solution).
    """
    T = keypoints.shape[0]
    K = site_idxs.shape[0]

    y = keypoints.reshape(T, K, 3)
    m0 = initial_offsets.reshape(K, 3)
    d = is_regularized.reshape(K, 3).astype(y.dtype)

    site_bodyid = jp.array(mjx_model.site_bodyid)[site_idxs]

    def fk_single(q_t):
        """Run FK for a single frame, return body pos and rot for sites."""
        data_t = mjx_data.replace(qpos=q_t)
        data_t = utils.kinematics(mjx_model, data_t)
        data_t = utils.com_pos(mjx_model, data_t)
        return data_t.xpos[site_bodyid], data_t.xmat[site_bodyid]

    p_all, R_all = jax.vmap(fk_single)(q)         # (T, K, 3), (T, K, 3, 3)
    z_all = y - p_all                             # (T, K, 3)
    s = jp.einsum("tkji,tkj->ki", R_all, z_all)   # s_k = sum_t R_{t,k}^T z_{t,k}  ->  (K, 3)
    z2 = jp.sum(z_all ** 2)                       # z2 = sum_t ||z_t||^2  ->  scalar

    # Closed-form solve, coordinate-wise
    denom = T + reg_coef * d
    numer = s + reg_coef * d * m0
    m_star = numer / denom

    # Residual error at the solution
    data_term = z2 - 2.0 * jp.sum(m_star * s) + T * jp.sum(m_star ** 2)
    reg_term = reg_coef * jp.sum((d * (m_star - m0)) ** 2)
    error = data_term + reg_term

    params = m_star.reshape(-1)
    return MOptResult(params=params, error=error)


class StacCore:
    """StacCore computes pose and offset optimization.

    Pose optimization uses a ProjectedGradient solver (``q_solver``).
    Offset optimization uses a closed-form solver (``_m_opt``).

    Args:
        tol (float): Tolerance for the q_solver.
        n_iter_q (int): Number of iterations for q optimization.
    """

    def __init__(self, tol=1e-5, n_iter_q=400):
        """Initialize StacCore.

        Args:
            tol (float): Tolerance value for ProjectedGradient 'q_solver'.
            n_iter_q (int): Number of iterations for q optimization.
        """
        self.q_solver = ProjectedGradient(
            fun=q_loss, projection=projection_box, maxiter=n_iter_q, tol=tol
        )

    def q_opt(
        self,
        mjx_model,
        mjx_data,
        marker_ref_arr: jp.ndarray,
        qs_to_opt: jp.ndarray,
        kps_to_opt: jp.ndarray,
        q0: jp.ndarray,
        lb,
        ub,
        site_idxs,
    ):
        """Updates q_pose using estimated marker parameters.

        This function is a wrapper for `_q_opt()` and updates `q_pose`
        based on estimated marker parameters.
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

    def m_opt(
        self,
        mjx_model,
        mjx_data,
        keypoints,
        q,
        initial_offsets,
        is_regularized,
        reg_coef,
        site_idxs,
    ):
        """Compute offset optimization.

        This function serves as a wrapper for `_m_opt()` and computes
        offset optimization based on the given parameters.

        Args:
            mjx_model (_type_): mjx.Model
            mjx_data (_type_): mjx.Data
            keypoints (jp.ndarray): Keypoints for each frame
            q (jp.ndarray): Joint angles for each frame
            initial_offsets (jp.ndarray): Initial offset values (from config)
            is_regularized (jp.ndarray): Boolean mask for regularized sites
            reg_coef (jp.ndarray): Regularization coefficient
            site_idxs (jp.ndarray): Site indices in mjx_model.site_xpos

        Returns:
            MOptResult: Result with .params (optimized offsets) and .error
                (scalar residual loss at the solution).
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
