"""Implementation of STAC for animal motion capture."""

import jax
import jax.numpy as jp
from jax import jit

from functools import partial

try:
    from jaxopt import ProjectedGradient
    from jaxopt.projection import projection_box
    from jaxopt import OptaxSolver
    _JAXOPT_AVAILABLE = True
except ImportError:
    _JAXOPT_AVAILABLE = False

try:
    import optax
    _OPTAX_AVAILABLE = True
except ImportError:
    _OPTAX_AVAILABLE = False

from stac_mjx import utils

try:
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    _JAXLS_AVAILABLE = True
except ImportError:
    _JAXLS_AVAILABLE = False


def q_loss(
    q: jp.ndarray,
    mjx_model,
    mjx_data,
    kp_data: jp.ndarray,
    qs_to_opt: jp.ndarray,
    kps_to_opt: jp.ndarray,
    initial_q: jp.ndarray,
    site_idxs: jp.ndarray,
    q_reg_weights: jp.ndarray,
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
        q_reg_weights (jp.ndarray): Per-qpos L2 regularization weights toward rest (q=0).
            Masked by qs_to_opt so only currently-optimized joints are penalized.

    Returns:
        float: sum of squares scalar loss plus joint regularization
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
    kp_loss = squared_error(residual)

    # Per-joint L2 regularization toward rest pose (q=0), masked to joints
    # currently being optimized so fixed joints are not spuriously pulled.
    q_reg = jp.sum(q_reg_weights * qs_to_opt * jp.square(q))

    return kp_loss + q_reg


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


def _q_opt_jaxls(
    jaxls_solver,
    mjx_model,
    mjx_data,
    marker_ref_arr: jp.ndarray,
    qs_to_opt: jp.ndarray,
    kps_to_opt: jp.ndarray,
    q0: jp.ndarray,
    lb,
    ub,
    site_idxs,
    q_reg_weights: jp.ndarray,
):
    """Update q_pose using the jaxls Levenberg-Marquardt solver.

    Returns (mjx_data, result) where result is a simple namespace with
    a .params attribute holding the optimized q — matching the interface
    expected by compute_stac.py callers.
    """
    import types

    q_opt = jaxls_solver.run_frame(
        q0,
        mjx_model,
        mjx_data,
        marker_ref_arr,
        qs_to_opt,
        kps_to_opt,
        lb,
        ub,
        site_idxs,
        q_reg_weights,
    )

    # Return a lightweight result object compatible with the existing caller pattern:
    #   res.params  -> optimized q
    #   res.state.error -> not computed, set to 0.0
    result = types.SimpleNamespace(
        params=q_opt,
        state=types.SimpleNamespace(error=0.0),
    )

    mjx_data = mjx_data.replace(qpos=utils.make_qs(q0, qs_to_opt, q_opt))
    mjx_data = utils.kinematics(mjx_model, mjx_data)

    return mjx_data, result


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
    q_reg_weights: jp.ndarray,
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
            q_reg_weights=q_reg_weights,
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

    def __init__(self, tol=1e-5, n_iter_q=400, n_iter_m=2000, stepsize_q=0.0,
                 use_jaxls=False, jaxls_lambda_initial=1.0, smooth_weight=0.0,
                 jaxls_linear_solver="auto", jaxls_chunk_size=100,
                 use_se3_root=True):
        """Initialze StacCore with 'q_solver' and 'm_solver'.

        Args:
            tol (float): Tolerance value for ProjectedGradient 'q_solver'.
            n_iter_q (int): Number of iterations for q optimization.
            n_iter_m (int): Number of iterations for m optimization.
            stepsize_q (float): Fixed step size for q optimizer. If > 0, disables the
                FISTA backtracking line search (a nested while_loop that runs up to 30
                extra kinematics evaluations per gradient step — very slow inside
                jax.lax.scan). Set to 0.0 to restore line search. Default: 0.0.
            use_jaxls (bool): If True and jaxls is available, use the batch
                Levenberg-Marquardt solver from jaxls instead of ProjectedGradient.
                All frames are solved simultaneously in one LM problem. Default: False.
            jaxls_lambda_initial (float): Initial LM damping factor. Default: 1.0.
            smooth_weight (float): Weight for ||q[t]-q[t-1]||² smoothness cost.
                0.0 disables smoothness (pure per-frame tracking). Start with 0.01–0.1.
                Only used when use_jaxls=True.
            jaxls_linear_solver (str): Linear solver for LM normal equations.
                "dense_cholesky" is fastest for short clips (T*nq < ~5000).
                "conjugate_gradient" for longer clips. Default: "auto".
        """
        self.opt = optax.sgd(learning_rate=5e-4, momentum=0.9, nesterov=False) if _OPTAX_AVAILABLE else None
        self._smooth_weight = smooth_weight

        self._use_jaxls = use_jaxls and _JAXLS_AVAILABLE
        self._jaxls_chunk_size = jaxls_chunk_size
        if use_jaxls and not _JAXLS_AVAILABLE:
            print("Warning: use_jaxls=True but jaxls is not installed. Falling back to ProjectedGradient.")

        if self._use_jaxls:
            self._jaxls_solver = JaxlsBatchSolver(
                n_iter=n_iter_q,
                linear_solver=jaxls_linear_solver,
                lambda_initial=jaxls_lambda_initial,
                smooth_weight=smooth_weight,
                use_se3_root=use_se3_root,
            )
            self.q_solver = None
        else:
            if not _JAXOPT_AVAILABLE:
                raise ImportError(
                    "jaxopt is required for the default ProjectedGradient solver. "
                    "Install it with: pip install jaxopt\n"
                    "Or use the jaxls solver: StacCore(..., use_jaxls=True)"
                )
            self.q_solver = ProjectedGradient(
                fun=q_loss, projection=projection_box, maxiter=n_iter_q, tol=tol,
                stepsize=stepsize_q,
            )
        if not _JAXOPT_AVAILABLE:
            # m_solver (offset optimization) always uses OptaxSolver from jaxopt
            # If jaxopt is unavailable we still create it so offset_optimization
            # can be called; it will fail at runtime if jaxopt is truly absent.
            self.m_solver = None
        else:
            self.m_solver = OptaxSolver(opt=self.opt, fun=m_loss, maxiter=n_iter_m)

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
        q_reg_weights: jp.ndarray,
    ):
        """Updates q_pose using estimated marker parameters.

        Dispatches to jaxls Levenberg-Marquardt solver when use_jaxls=True,
        otherwise falls back to the original ProjectedGradient solver.
        """
        if self._use_jaxls:
            return _q_opt_jaxls(
                self._jaxls_solver,
                mjx_model,
                mjx_data,
                marker_ref_arr,
                qs_to_opt,
                kps_to_opt,
                q0,
                lb,
                ub,
                site_idxs,
                q_reg_weights,
            )
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
            q_reg_weights,
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
