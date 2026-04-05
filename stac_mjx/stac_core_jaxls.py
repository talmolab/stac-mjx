"""Jaxls (Levenberg-Marquardt) batch trajectory IK solver for stac-mjx.

Solves ALL frames of a clip simultaneously in a single jaxls LeastSquaresProblem,
with optional smoothness coupling between adjacent timesteps — the same approach
pyroki uses for URDF robots, but using MJX forward kinematics so MJCF models
work natively without any format conversion.

Design (use_se3_root=True, default)
------------------------------------
Variables per timestep:
  SE3Var(jnp.arange(T))    — jaxlie.SE3 root pose (tangent_dim=6, no quat drift)
  JointVar(jnp.arange(T))  — shape (nq-7,) hinge angles
  KpVar(jnp.arange(T))     — shape (n_kp*3,) keypoint observations (held fixed)

Costs (vmapped internally by jaxls):
  marker_tracking_cost  : ||FK(root⊕joints) - kp||² * kp_weights   (T instances)
  regularization_cost   : ||sqrt(reg_w) * joints||²                  (T instances, if any reg>0)
  limit_constraint      : lb[7:] ≤ joints ≤ ub[7:]  (aug. Lagrangian)(T instances)
  smoothness_cost       : ||[SE3_log_diff; joint_diff]||² * smooth_w (T-1 instances, optional)

The SE3Var uses jaxlie's right-plus retraction (R_new = R_old @ exp(δ)), keeping
the quaternion on SO(3) without any post-solve normalization hack.

Fallback (use_se3_root=False)
-------------------------------
Uses a single flat QVar(nq) + Euclidean updates. Needed when some root DOFs are
frozen (qs_to_opt[:7] has False entries). Adds a quat normalization post-solve.

Linear solver auto-selection
------------------------------
  T * nq < 5000  → dense_cholesky (faster for small problems)
  T * nq ≥ 5000  → conjugate_gradient (exploits block-diagonal Jacobian sparsity)
Override via linear_solver="dense_cholesky" or "conjugate_gradient".

The problem graph is analyzed once per unique key and cached. Subsequent clips of
the same shape reuse the cached analysis and only call the JIT-compiled solve.

Usage
-----
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver

    solver = JaxlsBatchSolver(n_iter=50, smooth_weight=0.1)

    # Solve entire clip at once
    qposes = solver.solve_trajectory(
        q_init=jnp.tile(q0, (T, 1)),   # (T, nq) warm start
        mjx_model=mjx_model,
        mjx_data_template=mjx_data,
        kp_data=kp_data,               # (T, n_kp*3)
        qs_to_opt=qs_to_opt,
        kps_to_opt=kps_to_opt,
        lb=lb, ub=ub,
        site_idxs=site_idxs,
        q_reg_weights=q_reg_weights,
    )
    # qposes.shape == (T, nq)

    # Also works per-frame (for compatibility with root_optimization path)
    q_opt = solver.run_frame(q0, mjx_model, mjx_data, kp_data_frame, ...)
"""

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from stac_mjx import utils

# Number of free-joint DOFs in MuJoCo (3 translation + 4 quaternion).
_FREE_JOINT_NDOF = 7


# ---------------------------------------------------------------------------
# Internal state cache key
# ---------------------------------------------------------------------------

class _AnalyzedProblem:
    """Holds one analyzed jaxls problem and its variable classes.

    Supports two modes:
      SE3 mode (use_se3_root=True):   SE3Var + JointVar + KpVar
      Flat mode (use_se3_root=False): QVar  + KpVar
    """
    def __init__(self, analyzed, KpVar, *,
                 QVar=None, SE3Var=None, JointVar=None):
        self.analyzed = analyzed
        self.KpVar = KpVar
        # Flat mode
        self.QVar = QVar
        # SE3 mode
        self.SE3Var = SE3Var
        self.JointVar = JointVar

    @property
    def use_se3_root(self) -> bool:
        return self.SE3Var is not None


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------

class JaxlsBatchSolver:
    """Batch trajectory IK solver using jaxls Levenberg-Marquardt.

    Solves all T frames of a clip as a single least-squares problem,
    optionally coupling adjacent frames via a smoothness cost.

    Args:
        n_iter: Maximum LM iterations. Default 50.
        linear_solver: "auto" (default) picks dense_cholesky for T*nq < 5000 and
            conjugate_gradient otherwise. Explicit "dense_cholesky" or
            "conjugate_gradient" override the auto rule.
            dense_cholesky is O(n³) in tangent_dim — fast for small problems.
            conjugate_gradient exploits the block-diagonal Jacobian sparsity that
            arises when smooth_weight=0 (each frame is independent).
        lambda_initial: Initial LM damping. Default 1.0.
        smooth_weight: Weight for the smoothness cost. 0.0 = per-frame equivalent.
        use_se3_root: If True (default), represent the free-joint root pose as a
            jaxls.SE3Var so LM updates stay on the SO(3) manifold — no quaternion
            drift, no post-solve normalization needed. Requires qs_to_opt[:7] all
            True. Set False only when some root DOFs are frozen.
    """

    # Threshold below which dense_cholesky is faster than conjugate_gradient.
    _DENSE_THRESHOLD = 5000  # T * tangent_dim

    def __init__(
        self,
        n_iter: int = 50,
        linear_solver: str = "auto",
        lambda_initial: float = 1.0,
        smooth_weight: float = 0.0,
        use_se3_root: bool = True,
    ):
        self.n_iter = n_iter
        self.linear_solver = linear_solver
        self.lambda_initial = lambda_initial
        self.smooth_weight = smooth_weight
        self.use_se3_root = use_se3_root
        # Cache analyzed problems keyed by (T, nq, n_kp_dim, has_smooth, has_reg, se3)
        self._cache: dict[tuple, _AnalyzedProblem] = {}

    def _pick_linear_solver(self, T: int, tangent_dim: int) -> str:
        """Auto-select linear solver based on problem size."""
        if self.linear_solver != "auto":
            return self.linear_solver
        return (
            "dense_cholesky"
            if T * tangent_dim < self._DENSE_THRESHOLD
            else "conjugate_gradient"
        )

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    def _build_se3(
        self,
        T: int,
        nq: int,
        n_kp_dim: int,
        mjx_model,
        mjx_data_template,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
        smooth_weight: float,
    ) -> _AnalyzedProblem:
        """Build the jaxls problem using SE3Var (root) + JointVar (hinges).

        The SE3Var uses jaxlie's right-plus retraction, keeping the quaternion
        on SO(3) without any post-solve normalization. The JointVar covers the
        (nq - 7) hinge DOFs with box constraints.

        Assumes qs_to_opt[:7] are all True (root fully optimized).
        """
        n_hinges = nq - _FREE_JOINT_NDOF
        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        # ---- Variable classes ----
        class SE3Var(
            jaxls.Var[jaxlie.SE3],
            default_factory=jaxlie.SE3.identity,
            retract_fn=jaxlie.manifold.rplus,
            tangent_dim=6,
        ): ...
        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

        # Batched instances: one per timestep
        root_all  = SE3Var(jnp.arange(T))
        joint_all = JointVar(jnp.arange(T))
        kp_all    = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []

        # ---- Marker tracking cost ----
        @jaxls.Cost.factory
        def marker_cost(
            var_values: jaxls.VarValues,
            root_var: SE3Var,
            joint_var: JointVar,
            kp_var: KpVar,
        ) -> jnp.ndarray:
            T_root  = var_values[root_var]   # jaxlie.SE3
            joints  = var_values[joint_var]  # (n_hinges,)
            kp      = jax.lax.stop_gradient(var_values[kp_var])  # (n_kp_dim,)

            # Reconstruct MuJoCo qpos: [x,y,z, qw,qx,qy,qz, hinges...]
            xyz  = T_root.translation()      # (3,)
            wxyz = T_root.rotation().wxyz    # (4,)  w,x,y,z
            q    = jnp.concatenate([xyz, wxyz, joints])  # (nq,)

            full_q = jnp.where(qs_to_opt, q, mjx_data_template.qpos)
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            return (kp - markers) * kps_to_opt

        costs.append(marker_cost(root_all, joint_all, kp_all))

        # ---- Joint regularization cost (hinges only; root typically unreg.) ----
        if jnp.any(q_reg_weights[_FREE_JOINT_NDOF:] > 0):
            hinge_regs = q_reg_weights[_FREE_JOINT_NDOF:]
            hinge_opt  = qs_to_opt[_FREE_JOINT_NDOF:]

            @jaxls.Cost.factory
            def reg_cost(
                var_values: jaxls.VarValues,
                joint_var: JointVar,
            ) -> jnp.ndarray:
                j = var_values[joint_var]
                return jnp.sqrt(hinge_regs * hinge_opt) * j

            costs.append(reg_cost(joint_all))

        # ---- Hinge limit constraint (SE3 root is unconstrained by design) ----
        hinge_lb = lb[_FREE_JOINT_NDOF:]
        hinge_ub = ub[_FREE_JOINT_NDOF:]

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(
            var_values: jaxls.VarValues,
            joint_var: JointVar,
        ) -> jnp.ndarray:
            j = var_values[joint_var]
            return jnp.concatenate([hinge_lb - j, j - hinge_ub])

        costs.append(limit_cost(joint_all))

        # ---- Smoothness: SE3 log-diff + joint diff ----
        if smooth_weight > 0.0 and T > 1:
            @jaxls.Cost.factory
            def smoothness_cost(
                var_values: jaxls.VarValues,
                root_curr: SE3Var,
                root_prev: SE3Var,
                joint_curr: JointVar,
                joint_prev: JointVar,
            ) -> jnp.ndarray:
                # SE3 geodesic difference in tangent space (6D)
                root_diff  = (var_values[root_prev].inverse() @ var_values[root_curr]).log()
                joint_diff = var_values[joint_curr] - var_values[joint_prev]
                return jnp.concatenate([root_diff, joint_diff]) * smooth_weight

            costs.append(smoothness_cost(
                SE3Var(jnp.arange(1, T)),      # root_curr
                SE3Var(jnp.arange(0, T-1)),    # root_prev
                JointVar(jnp.arange(1, T)),    # joint_curr
                JointVar(jnp.arange(0, T-1)), # joint_prev
            ))

        variables = [root_all, joint_all, kp_all]

        analyzed = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
        )

        return _AnalyzedProblem(
            analyzed=analyzed, KpVar=KpVar,
            SE3Var=SE3Var, JointVar=JointVar,
        )

    def _build(
        self,
        T: int,
        nq: int,
        n_kp_dim: int,
        mjx_model,
        mjx_data_template,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
        smooth_weight: float,
    ) -> _AnalyzedProblem:
        """Build and analyze the jaxls problem for a given (T, nq, n_kp_dim) shape.

        This is called once per unique combination and cached.
        """
        dummy_q = jnp.zeros((nq,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        # ---- Variable classes ----
        class QVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_q): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

        # Batched instances: one per timestep
        q_all = QVar(jnp.arange(T))
        kp_all = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []

        # ---- Marker tracking cost ----
        # jaxls vmaps this over T via the batch dimension of q_all and kp_all.
        # mjx_model and site_idxs are closed over (constant across frames).
        @jaxls.Cost.factory
        def marker_cost(
            var_values: jaxls.VarValues,
            q_var: QVar,
            kp_var: KpVar,
        ) -> jnp.ndarray:
            q = var_values[q_var]    # (nq,) — one timestep (jaxls vmaps over T)
            # stop_gradient: kp is an observation, not a parameter to optimize.
            # Zero Jacobian w.r.t. kp_var → LM never updates it, only q is moved.
            kp = jax.lax.stop_gradient(var_values[kp_var])  # (n_kp_dim,)
            # qs_to_opt selects which joints are optimized; rest stay at q
            # (for the batch solver we typically pass qs_to_opt=ones so full_q==q)
            full_q = jnp.where(qs_to_opt, q, mjx_data_template.qpos)
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            return (kp - markers) * kps_to_opt

        costs.append(marker_cost(q_all, kp_all))

        # ---- Joint regularization cost ----
        if jnp.any(q_reg_weights > 0):
            @jaxls.Cost.factory
            def reg_cost(
                var_values: jaxls.VarValues,
                q_var: QVar,
            ) -> jnp.ndarray:
                q = var_values[q_var]
                return jnp.sqrt(q_reg_weights * qs_to_opt) * q

            costs.append(reg_cost(q_all))

        # ---- Joint limit constraint (augmented Lagrangian) ----
        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(
            var_values: jaxls.VarValues,
            q_var: QVar,
        ) -> jnp.ndarray:
            q = var_values[q_var]
            return jnp.concatenate([lb - q, q - ub])

        costs.append(limit_cost(q_all))

        # ---- Smoothness cost: ||q[t] - q[t-1]||² * weight ----
        if smooth_weight > 0.0 and T > 1:
            @jaxls.Cost.factory
            def smoothness_cost(
                var_values: jaxls.VarValues,
                q_curr: QVar,
                q_prev: QVar,
            ) -> jnp.ndarray:
                return (var_values[q_curr] - var_values[q_prev]) * smooth_weight

            costs.append(smoothness_cost(
                QVar(jnp.arange(1, T)),      # q[1..T-1]
                QVar(jnp.arange(0, T - 1)),  # q[0..T-2]
            ))

        # KpVar must be in variables so jaxls can build the sparsity pattern.
        # stop_gradient in marker_cost gives it a zero Jacobian — LM sees no
        # gradient direction for kp and leaves it fixed. kp_data is supplied
        # per-clip via initial_vals in solve_trajectory().
        variables = [q_all, kp_all]

        analyzed = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
        )

        return _AnalyzedProblem(analyzed=analyzed, KpVar=KpVar, QVar=QVar)

    def _get_analyzed(
        self,
        T: int,
        mjx_model,
        mjx_data_template,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
    ) -> _AnalyzedProblem:
        nq = int(mjx_model.nq)
        n_kp_dim = int(kp_data.shape[-1]) if kp_data.ndim > 1 else int(kp_data.shape[0])
        has_reg = bool(jnp.any(q_reg_weights > 0))
        has_smooth = self.smooth_weight > 0.0
        key = (T, nq, n_kp_dim, has_reg, has_smooth, self.use_se3_root)

        if key not in self._cache:
            builder = self._build_se3 if self.use_se3_root else self._build
            self._cache[key] = builder(
                T=T,
                nq=nq,
                n_kp_dim=n_kp_dim,
                mjx_model=mjx_model,
                mjx_data_template=mjx_data_template,
                qs_to_opt=qs_to_opt,
                kps_to_opt=kps_to_opt,
                lb=lb,
                ub=ub,
                site_idxs=site_idxs,
                q_reg_weights=q_reg_weights,
                smooth_weight=self.smooth_weight,
            )
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public API: batch trajectory solve
    # ------------------------------------------------------------------

    def solve_trajectory(
        self,
        q_init: jnp.ndarray,
        mjx_model,
        mjx_data_template,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve IK for an entire clip simultaneously.

        Solves all T frames as one jaxls LeastSquaresProblem. Adjacent frames
        are coupled via the smoothness cost when smooth_weight > 0.

        Args:
            q_init: Initial joint config for all frames, shape (T, nq).
                Warm-start from previous solution or tile a single q0.
            mjx_model: MJX model (constant).
            mjx_data_template: MJX data used as FK template (qpos overwritten).
            kp_data: Observed keypoints, shape (T, n_kp*3) or (T, n_kp, 3).
            qs_to_opt: Boolean mask (nq,) selecting joints to optimize.
            kps_to_opt: Per-coordinate weight mask (n_kp*3,).
            lb: Joint lower bounds (nq,).
            ub: Joint upper bounds (nq,).
            site_idxs: Site indices for marker positions.
            q_reg_weights: Per-joint L2 regularization weights (nq,).

        Returns:
            Optimized joint angles, shape (T, nq).
        """
        # Flatten kp_data to (T, n_kp*3)
        if kp_data.ndim == 3:
            kp_data = kp_data.reshape(kp_data.shape[0], -1)
        T = q_init.shape[0]

        prob = self._get_analyzed(
            T, mjx_model, mjx_data_template,
            kp_data, qs_to_opt, kps_to_opt, lb, ub, site_idxs, q_reg_weights,
        )
        KpVar = prob.KpVar

        if prob.use_se3_root:
            return self._solve_se3(prob, T, q_init, kp_data)
        else:
            return self._solve_flat(prob, T, q_init, kp_data)

    def _solve_se3(
        self,
        prob: _AnalyzedProblem,
        T: int,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve trajectory using the SE3Var + JointVar representation."""
        SE3Var  = prob.SE3Var
        JointVar = prob.JointVar
        KpVar   = prob.KpVar

        # Split q_init into root pose (SE3) + hinge angles.
        # MuJoCo free-joint layout: [x,y,z, qw,qx,qy,qz, hinges...]
        xyz_init    = q_init[:, :3]                      # (T, 3)
        wxyz_init   = q_init[:, 3:7]                     # (T, 4)
        hinges_init = q_init[:, _FREE_JOINT_NDOF:]       # (T, n_hinges)

        # Normalize quaternion before building SE3 (invalid quat → NaN gradients)
        qn = jnp.linalg.norm(wxyz_init, axis=-1, keepdims=True)
        wxyz_init = wxyz_init / jnp.where(qn > 0, qn, 1.0)

        # Build batched jaxlie.SE3 initial values
        roots_init = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz=wxyz_init),
            xyz_init,
        )  # batch shape (T,)

        # Pick linear solver based on problem size.
        # SE3 path: tangent_dim = 6 (root) + n_hinges per frame
        n_hinges   = q_init.shape[1] - _FREE_JOINT_NDOF
        tangent_dim = 6 + n_hinges
        linear_solver = self._pick_linear_solver(T, tangent_dim)

        sol = prob.analyzed.solve(
            verbose=False,
            linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals=jaxls.VarValues.make([
                SE3Var(jnp.arange(T)).with_value(roots_init),
                JointVar(jnp.arange(T)).with_value(hinges_init),
                KpVar(jnp.arange(T)).with_value(kp_data),
            ]),
        )

        # Recombine SE3 + joints back into (T, nq) qpos array.
        sol_roots  = sol[SE3Var(jnp.arange(T))]        # SE3 batch (T,)
        sol_joints = sol[JointVar(jnp.arange(T))]      # (T, n_hinges)

        xyz_sol  = sol_roots.translation()              # (T, 3)
        wxyz_sol = sol_roots.rotation().wxyz            # (T, 4) — already on SO(3)

        return jnp.concatenate([xyz_sol, wxyz_sol, sol_joints], axis=-1)  # (T, nq)

    def _solve_flat(
        self,
        prob: _AnalyzedProblem,
        T: int,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve trajectory using the flat QVar representation."""
        QVar  = prob.QVar
        KpVar = prob.KpVar

        linear_solver = self._pick_linear_solver(T, q_init.shape[1])

        sol = prob.analyzed.solve(
            verbose=False,
            linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals=jaxls.VarValues.make([
                QVar(jnp.arange(T)).with_value(q_init),
                KpVar(jnp.arange(T)).with_value(kp_data),
            ]),
        )

        qposes = sol[QVar(jnp.arange(T))]  # (T, nq)

        # Post-normalize quaternion: Euclidean LM updates can drift off SO(3).
        quat      = qposes[:, 3:7]
        quat_norm = jnp.linalg.norm(quat, axis=-1, keepdims=True)
        return qposes.at[:, 3:7].set(quat / jnp.where(quat_norm > 0, quat_norm, 1.0))

    # ------------------------------------------------------------------
    # Public API: single-frame solve (for root_optimization compatibility)
    # ------------------------------------------------------------------

    def run_frame(
        self,
        q0: jnp.ndarray,
        mjx_model,
        mjx_data,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kps_to_opt: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        q_reg_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve IK for a single frame (T=1 batch).

        Used by root_optimization() and any other per-frame path.

        Returns:
            Optimized joint angles (nq,).
        """
        kp_flat = kp_data.flatten() if kp_data.ndim > 1 else kp_data
        q_init = q0[None]          # (1, nq)
        kp_batch = kp_flat[None]   # (1, n_kp*3)

        result = self.solve_trajectory(
            q_init=q_init,
            mjx_model=mjx_model,
            mjx_data_template=mjx_data,
            kp_data=kp_batch,
            qs_to_opt=qs_to_opt,
            kps_to_opt=kps_to_opt,
            lb=lb,
            ub=ub,
            site_idxs=site_idxs,
            q_reg_weights=q_reg_weights,
        )
        return result[0]  # (nq,)
