"""Implementation of STAC for animal motion capture."""

import jax
import jax.numpy as jp
from jax import jit

from functools import partial
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box
from jaxopt import OptaxSolver

import optax

from stac_mjx import utils


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
    residual = squared_error(residual)

    return residual


@jit
def m_loss(
    offsets: jp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    q: jp.ndarray,
    initial_offsets: jp.ndarray,
    site_idxs: jp.ndarray,
    is_regularized: bool = None,
    reg_coef: float = 0.0,
) -> jp.array:
    # fmt: off
    """Compute the marker residual for optimization.

    Args:
        offsets (jp.ndarray): vector of offsets to inferred mocap sites
        mjx_model (_type_): MJX Model
        mjx_data (_type_):  MJX Data
        kp_data (jp.ndarray):  Mocap data in global coordinates
        q (jp.ndarray): proposed qpos values
        initial_offsets (jp.ndarray): Initial offset values for offset regularization
        is_regularized (bool, optional): binary vector of offsets to regularize.. Defaults to None.
        reg_coef (float, optional):  L1 regularization coefficient during marker loss.. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    # fmt: on
    def f(carry, input):
        # Unpack arguments
        qpos, kp = input
        mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized = carry

        # Get the offset relative to the initial position, only for markers you wish to regularize
        reg_term = (
            reg_term + (jp.square(offsets - initial_offsets.flatten())) * is_regularized
        )

        # Set qpos and offsets
        mjx_data = mjx_data.replace(qpos=qpos)
        mjx_model = utils.set_site_pos(
            mjx_model, jp.reshape(offsets, (-1, 3)), site_idxs
        )

        # Forward kinematics
        mjx_data = utils.kinematics(mjx_model, mjx_data)
        mjx_data = utils.com_pos(mjx_model, mjx_data)
        markers = utils.get_site_xpos(mjx_data, site_idxs).flatten()

        # Accumulate squared residual
        residual = residual + jp.square((kp - markers))
        return (
            mjx_model,
            mjx_data,
            reg_term,
            residual,
            initial_offsets,
            is_regularized,
        ), None

    (mjx_model, mjx_data, reg_term, residual, initial_offsets, is_regularized), _ = (
        jax.lax.scan(
            f,
            (
                mjx_model,
                mjx_data,
                jp.zeros(kp_data.shape[1]),
                jp.zeros(kp_data.shape[1]),
                initial_offsets,
                is_regularized,
            ),
            (q, kp_data),
        )
    )
    return jp.sum(residual) + reg_coef * jp.sum(reg_term)


def squared_error(x):
    """Compute the squared error + sum."""
    return jp.sum(jp.square(x))


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


@partial(jit, static_argnames=["m_solver"])
def _m_opt(
    m_solver,
    offset0,
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

    Args:
        offset0 (jp.ndarray): Proposed offset values
        mjx_model (_type_): mjx.Model
        mjx_data (_type_): mjx.Data
        keypoints (jp.ndarray): Keypoints for each frame
        q (jp.ndarray): Joint angles for each frame
        initial_offsets (jp.ndarray): Initial offset values (from config)
        is_regularized (jp.ndarray): Boolean mask for regularized sites
        reg_coef (jp.ndarray): Regularization coefficient
        site_idxs (jp.ndarray): Site indices in mjx_model.site_xpos

    Returns:
        _type_: result of optimization
    """
    res = m_solver.run(
        offset0,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        kp_data=keypoints,
        q=q,
        initial_offsets=initial_offsets,
        is_regularized=is_regularized,
        reg_coef=reg_coef,
        site_idxs=site_idxs,
    )

    return res


class StacCore:
    """StacCore computes offset optimization.

    This class contains the 'q_solver' and 'm_solver' attributes that are used to
    compute optimize pose and offsets.

    Args:
        tol (float): Tolerance for the q_solver.
    """

    def __init__(self, tol=1e-5):
        """Initialze StacCore with 'q_solver' and 'm_solver'.

        Args:
            tol (float): Tolerance value for ProjectedGradient 'q_solver'.
        """
        self.opt = optax.sgd(learning_rate=5e-4, momentum=0.9, nesterov=False)

        # TODO: make maxiter a config parameter
        self.q_solver = ProjectedGradient(
            fun=q_loss, projection=projection_box, maxiter=400, tol=tol
        )
        self.m_solver = OptaxSolver(opt=self.opt, fun=m_loss, maxiter=2000)

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
        offset0,
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

        This function serves as a wrapper for `_m_opt()` and computes offset optimization
        based on the given parameters.

        Args:
            offset0 (jp.ndarray): Proposed offset values
            mjx_model (_type_): mjx.Model
            mjx_data (_type_): mjx.Data
            keypoints (jp.ndarray): Keypoints for each frame
            q (jp.ndarray): Joint angles for each frame
            initial_offsets (jp.ndarray): Initial offset values (from config)
            is_regularized (jp.ndarray): Boolean mask for regularized sites
            reg_coef (jp.ndarray): Regularization coefficient
            site_idxs (jp.ndarray): Site indices in mjx_model.site_xpos

        Returns:
            _type_: result of optimization
        """
        return _m_opt(
            self.m_solver,
            offset0,
            mjx_model,
            mjx_data,
            keypoints,
            q,
            initial_offsets,
            is_regularized,
            reg_coef,
            site_idxs,
        )
