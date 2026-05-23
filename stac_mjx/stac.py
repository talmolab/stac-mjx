"""Stac class handling high level functionality of stac-mjx."""

from pathlib import Path

import imageio
import jaxlie
import mujoco
import numpy as np
from beartype import beartype
from jax import Array
from jax import numpy as jp
from jaxtyping import Float, Int, jaxtyped
from mujoco import mjx
from omegaconf import DictConfig
from tqdm import tqdm

from stac_mjx import compute_stac, io, rescale, stac_core, utils

_ROOT_QPOS_LB = jp.concatenate([-jp.inf * jp.ones(3), -1.0 * jp.ones(4)])
_ROOT_QPOS_UB = jp.concatenate([jp.inf * jp.ones(3), 1.0 * jp.ones(4)])

# mujoco jnt_type enums: https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtjoint
_MUJOCO_JOINT_TYPE_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

_MUJOCO_JOINT_TYPE_UNCONSTRAINED = {
    mujoco.mjtJoint.mjJNT_FREE: (
        jp.concatenate([-jp.inf * jp.ones(3), -1.0 * jp.ones(4)]),
        jp.concatenate([jp.inf * jp.ones(3), 1.0 * jp.ones(4)]),
    ),
    mujoco.mjtJoint.mjJNT_BALL: (
        jp.concatenate([-1.0 * jp.ones(4)]),
        jp.concatenate([1.0 * jp.ones(4)]),
    ),
    mujoco.mjtJoint.mjJNT_SLIDE: (
        jp.concatenate([-jp.inf * jp.ones(1)]),
        jp.concatenate([jp.inf * jp.ones(1)]),
    ),
    mujoco.mjtJoint.mjJNT_HINGE: (
        jp.concatenate([-2 * jp.pi * jp.ones(1)]),
        jp.concatenate([2 * jp.pi * jp.ones(1)]),
    ),
}


def _align_joint_dims(
    types: np.ndarray,
    ranges: np.ndarray,
    names: list[str],
) -> tuple[Float[Array, " n_qpos"], Float[Array, " n_qpos"], list[str]]:
    """Create bounds and joint names aligned with qpos dimensions.

    Args:
        types: Array of MuJoCo joint type enums.
        ranges: Array of joint ranges (n_joints, 2).
        names: Joint names.

    Returns:
        Tuple of (lower bounds, upper bounds, per-qpos-dim joint names).
    """
    lb = []
    ub = []
    part_names = []
    for type, range, name in zip(types, ranges, names):
        dims = _MUJOCO_JOINT_TYPE_DIMS[type]
        # Set inf bounds for freejoint
        if type == mujoco.mjtJoint.mjJNT_FREE:
            lb.append(_MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][0])
            ub.append(_MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][1])
            part_names += [name] * dims
        else:
            l, u = range
            if l == 0 and u == 0:  # default joint lims are 0 0, which is unconstrained
                l = _MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][0]
                u = _MUJOCO_JOINT_TYPE_UNCONSTRAINED[type][1]
            lb.append(l * jp.ones(dims))
            ub.append(u * jp.ones(dims))
            part_names += [name] * dims

    return jp.minimum(jp.concatenate(lb), 0.0), jp.concatenate(ub), part_names


class Stac:
    """Main class for skeletal registration and rendering.

    Handles model setup, body site initialization, pose/offset optimization,
    and rendering of fitted results.
    """

    def __init__(self, xml_path: str, cfg: DictConfig, kp_names: list[str]):
        """Initialize Stac with model, config, and keypoint names.

        Args:
            xml_path: Path to model MJCF file.
            cfg: STAC configuration.
            kp_names: Ordered list of mocap keypoint names.
        """
        self.cfg = cfg
        self._kp_names = kp_names
        self._xml_path = Path(xml_path)
        self._marker_size = cfg.model.MARKER_SIZE

        (
            self._mj_model,
            self._body_site_idxs,
            self._is_regularized,
        ) = self._init_body_sites()

        self._body_names = [
            self._mj_model.body(i).name for i in range(self._mj_model.nbody)
        ]

        if "ROOT_OPTIMIZATION_KEYPOINT" in self.cfg.model:
            self._root_kp_idx = self._kp_names.index(
                self.cfg.model.ROOT_OPTIMIZATION_KEYPOINT
            )
        else:
            self._root_kp_idx = -1

        # Set up bounds and part_names based on joint ranges, taking into account the dimensionality of parameters
        joint_names = [self._mj_model.joint(i).name for i in range(self._mj_model.njnt)]
        self._lb, self._ub, self._part_names = _align_joint_dims(
            self._mj_model.jnt_type, self._mj_model.jnt_range, joint_names
        )

        # Generate boolean flags for keypoints included in trunk optimization.
        self._trunk_kps = jp.array(
            [n in self.cfg.model.TRUNK_OPTIMIZATION_KEYPOINTS for n in kp_names],
        )

        self._mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[cfg.stac.mujoco.solver.lower()]

        self._mj_model.opt.iterations = cfg.stac.mujoco.iterations
        self._mj_model.opt.ls_iterations = cfg.stac.mujoco.ls_iterations

        self._mj_model.opt.jacobian = 0  # dense — runs faster on GPU
        self._freejoint = bool(self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
        self._slidejoint = bool(
            self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE
        )
        self._fixed = not (self._freejoint or self._slidejoint)

    def _build_body_spec(self) -> mujoco.MjSpec:
        """Create a fresh spec with body sites for keypoints.

        Returns:
            Scaled MjSpec with keypoint sites attached.
        """
        spec = mujoco.MjSpec.from_file(str(self._xml_path))
        for key, v in self.cfg.model.KEYPOINT_MODEL_PAIRS.items():
            parent = spec.body(v)
            pos = self.cfg.model.KEYPOINT_INITIAL_OFFSETS[key]

            if isinstance(pos, str):
                pos = [float(p) for p in pos.split(" ")]

            parent.add_site(
                name=key,
                size=[self._marker_size] * 3,
                rgba=(0, 0, 0, 0.8),
                pos=pos,
                group=3,
            )

        return rescale.dm_scale_spec(spec, self.cfg.model.SCALE_FACTOR)

    def _init_body_sites(
        self,
    ) -> tuple[
        mujoco.MjModel, Int[Array, " n_keypoints"], Float[Array, "n_keypoints 3"]
    ]:
        """Compile the fitting model and create site indices and masks.

        Returns:
            Tuple of (compiled MjModel, site indices, regularization mask).
        """
        spec = self._build_body_spec()
        model = spec.compile()

        site_index_map = {
            site_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for site_name in self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()
        }

        is_regularized = []
        for k in site_index_map.keys():
            if any(n == k for n in self.cfg.model.get("SITES_TO_REGULARIZE", [])):
                is_regularized.append(jp.array([1.0, 1.0, 1.0]))
            else:
                is_regularized.append(jp.array([0.0, 0.0, 0.0]))
        is_regularized = jp.stack(is_regularized)
        body_site_idxs = jp.array(list(site_index_map.values()))
        return model, body_site_idxs, is_regularized

    def _get_error_stats(self, errors: list) -> tuple[np.ndarray, float, float]:
        """Compute error statistics.

        Args:
            errors: List of per-frame error values.

        Returns:
            Tuple of (flattened errors, mean, standard deviation).
        """
        flattened_errors = np.array(errors).reshape(-1)

        mean = np.mean(flattened_errors)
        std = np.std(flattened_errors)

        return flattened_errors, mean, std

    def _run_root_optimization(
        self, mjx_model, mjx_data, kp_data, n_solver_max_iters: int
    ):
        """Run root optimization when configured."""
        if self._root_kp_idx == -1:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT not specified, skipping Root Optimization."
            )
            return mjx_data
        if self._fixed:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT specified but model has fixed root, "
                "skipping Root Optimization"
            )
            return mjx_data

        return compute_stac.root_optimization(
            mjx_model,
            mjx_data,
            kp_data,
            self._root_kp_idx,
            self._lb,
            self._ub,
            self._body_site_idxs,
            self._trunk_kps,
            n_solver_max_iters=n_solver_max_iters,
            initial_step_damping=self.cfg.stac.q_opt.initial_step_damping,
        )

    def _run_pose_optimization(
        self, mjx_model, mjx_data, kp_data, n_solver_max_iters: int
    ):
        """Run pose optimization.

        Returns:
            Tuple of (mjx_data, qpos, body_pos, body_quat, marker_pos, frame_error).
        """
        n_frames = kp_data.shape[0]
        q_init = jp.tile(mjx_data.qpos, (n_frames, 1))
        return compute_stac.pose_optimization(
            mjx_model,
            mjx_data,
            kp_data,
            self._lb,
            self._ub,
            self._body_site_idxs,
            q_init,
            acceleration_smoothness_weight=(
                self.cfg.stac.q_opt.acceleration_smoothness_weight
            ),
            n_solver_max_iters=n_solver_max_iters,
            initial_step_damping=self.cfg.stac.q_opt.initial_step_damping,
        )

    @jaxtyped(typechecker=beartype)
    def calibrate(
        self, kp_data: Float[Array, "n_frames n_keypoints_xyz"]
    ) -> io.StacData:
        """Calibrate marker offsets using alternating pose and offset solves.

        Runs root optimization, then alternates pose and offset optimization
        for N_ITERS iterations, followed by a final pose optimization pass.

        Args:
            kp_data: Flattened mocap keypoint data.

        Returns:
            Packaged STAC output data.
        """
        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        self._offsets = jp.copy(utils.get_site_pos(mjx_model, self._body_site_idxs))

        mjx_model = utils.set_site_pos(mjx_model, self._offsets, self._body_site_idxs)

        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)

        mjx_data = self._run_root_optimization(
            mjx_model,
            mjx_data,
            kp_data,
            n_solver_max_iters=self.cfg.stac.q_opt.calibration_max_iterations,
        )

        for calibration_iter in range(self.cfg.model.N_ITERS):
            print(
                f"Calibration iteration: {calibration_iter + 1}/{self.cfg.model.N_ITERS}"
            )
            mjx_data, qpos, body_pos, body_quat, marker_pos, frame_error = (
                self._run_pose_optimization(
                    mjx_model,
                    mjx_data,
                    kp_data,
                    n_solver_max_iters=self.cfg.stac.q_opt.calibration_max_iterations,
                )
            )

            flattened_errors, mean, std = self._get_error_stats(frame_error)
            print(f"Mean: {mean}")
            print(f"Standard deviation: {std}")

            mjx_model, mjx_data, self._offsets = compute_stac.offset_optimization(
                mjx_model,
                mjx_data,
                kp_data,
                self._offsets,
                qpos,
                self.cfg.model.N_SAMPLE_FRAMES,
                self._is_regularized,
                self._body_site_idxs,
                self.cfg.model.M_REG_COEF,
            )

        print("Final pose optimization", flush=True)
        mjx_data, qpos, body_pos, body_quat, marker_pos, frame_error = (
            self._run_pose_optimization(
                mjx_model,
                mjx_data,
                kp_data,
                n_solver_max_iters=self.cfg.stac.q_opt.final_pose_max_iterations,
            )
        )

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")
        return self._package_data(
            np.array(qpos),
            np.array(body_pos),
            np.array(body_quat),
            np.array(marker_pos),
            np.array(kp_data),
        )

    def run_ik(
        self,
        kp_data: Float[Array, "n_frames n_keypoints_xyz"],
        offsets: np.ndarray,
    ) -> io.StacData:
        """Run inverse kinematics using calibrated marker offsets.

        Stand-alone IK step for use after marker offsets have been calibrated.
        Useful when running IK on a different dataset than was used during
        calibration.

        Args:
            kp_data: Flattened keypoint data in meters.
            offsets: Marker offsets from a previous calibration run.

        Returns:
            Packaged STAC output data.
        """
        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        self._offsets = jp.asarray(offsets)
        mjx_model = utils.set_site_pos(mjx_model, offsets, self._body_site_idxs)
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)

        mjx_data = self._run_root_optimization(
            mjx_model,
            mjx_data,
            kp_data,
            n_solver_max_iters=self.cfg.stac.q_opt.ik_max_iterations,
        )

        chunk_size = int(self.cfg.stac.n_frames_per_clip)
        q_opt_cfg = self.cfg.stac.q_opt
        context_frames = int(q_opt_cfg.context_frames)
        coarse_stride = int(q_opt_cfg.coarse_init_stride)
        total_frames = int(kp_data.shape[0])
        n_chunks = (total_frames + chunk_size - 1) // chunk_size
        solve_frames = chunk_size + 2 * context_frames

        joint_mask = jp.ones(mjx_model.nq, dtype=bool)
        kp_mask = jp.ones(kp_data.shape[1], dtype=bool)
        joint_reg_weights = jp.zeros(mjx_model.nq)
        acceleration_smoothness_weight = q_opt_cfg.acceleration_smoothness_weight
        problems = {}
        coarse_problems = {}
        root_kp_start = self._root_kp_idx * 3

        all_qpos, all_body_pos, all_body_quat = [], [], []
        all_marker_pos, all_errors = [], []
        prev_overlap_q = None

        for chunk_idx, start in enumerate(range(0, total_frames, chunk_size)):
            keep = min(chunk_size, total_frames - start)
            data_start = max(start - context_frames, 0)
            data_stop = min(start + chunk_size + context_frames, total_frames)
            prefix = max(context_frames - start, 0)
            suffix = solve_frames - prefix - (data_stop - data_start)
            kp_parts = []
            if prefix > 0:
                kp_parts.append(jp.repeat(kp_data[:1], prefix, axis=0))
            kp_parts.append(kp_data[data_start:data_stop])
            if suffix > 0:
                kp_parts.append(jp.repeat(kp_data[-1:], suffix, axis=0))
            kp_chunk = jp.concatenate(kp_parts, axis=0)
            if chunk_idx % 100 == 0 or chunk_idx == n_chunks - 1:
                print(f"Clip {chunk_idx + 1}/{n_chunks}")

            if solve_frames not in problems:
                problems[solve_frames] = stac_core.build_q_opt_problem(
                    solve_frames,
                    mjx_model,
                    mjx_data,
                    joint_mask,
                    kp_mask,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    kp_data.shape[1],
                    joint_reg_weights,
                    acceleration_smoothness_weight=acceleration_smoothness_weight,
                )

            if prev_overlap_q is None:
                q_init = jp.tile(mjx_data.qpos, (solve_frames, 1))
                if self._root_kp_idx >= 0 and not self._fixed:
                    q_init = q_init.at[:, :3].set(
                        kp_chunk[:, root_kp_start : root_kp_start + 3]
                    )
            else:
                init_overlap = min(int(prev_overlap_q.shape[0]), solve_frames)
                q_init = jp.tile(prev_overlap_q[init_overlap - 1], (solve_frames, 1))
                q_init = q_init.at[:init_overlap].set(prev_overlap_q[:init_overlap])
                if self._root_kp_idx >= 0 and not self._fixed:
                    q_init = q_init.at[init_overlap:, :3].set(
                        kp_chunk[init_overlap:, root_kp_start : root_kp_start + 3]
                    )

            if self._freejoint:
                quat = q_init[:, 3:7]
                quat_norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
                q_init = q_init.at[:, 3:7].set(
                    quat / jp.where(quat_norm > 0, quat_norm, 1.0)
                )

            coarse_idx_np = np.arange(0, solve_frames, coarse_stride, dtype=np.int32)
            if coarse_idx_np[-1] != solve_frames - 1:
                coarse_idx_np = np.concatenate(
                    [coarse_idx_np, np.array([solve_frames - 1], dtype=np.int32)]
                )
            coarse_idx = jp.array(coarse_idx_np)
            coarse_frames = int(coarse_idx.shape[0])
            if coarse_frames not in coarse_problems:
                coarse_problems[coarse_frames] = stac_core.build_q_opt_problem(
                    coarse_frames,
                    mjx_model,
                    mjx_data,
                    joint_mask,
                    kp_mask,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    kp_data.shape[1],
                    joint_reg_weights,
                    acceleration_smoothness_weight=acceleration_smoothness_weight,
                )
            q_coarse = stac_core.q_opt(
                coarse_problems[coarse_frames],
                q_init[coarse_idx],
                kp_chunk[coarse_idx],
                n_solver_max_iters=q_opt_cfg.ik_max_iterations,
                initial_step_damping=q_opt_cfg.initial_step_damping,
            )
            frame_ids = jp.arange(solve_frames)
            right = jp.searchsorted(coarse_idx, frame_ids, side="right")
            right = jp.clip(right, 1, coarse_frames - 1)
            left = right - 1
            left_t = coarse_idx[left]
            right_t = coarse_idx[right]
            alpha = (
                (frame_ids - left_t).astype(q_coarse.dtype)
                / jp.maximum(right_t - left_t, 1).astype(q_coarse.dtype)
            )[:, None]
            q_left = q_coarse[left]
            q_right = q_coarse[right]
            if self._freejoint:
                xyz = (1.0 - alpha) * q_left[:, :3] + alpha * q_right[:, :3]
                left_wxyz = q_left[:, 3:7]
                right_wxyz = q_right[:, 3:7]
                left_wxyz = left_wxyz / jp.maximum(
                    jp.linalg.norm(left_wxyz, axis=-1, keepdims=True), 1e-12
                )
                right_wxyz = right_wxyz / jp.maximum(
                    jp.linalg.norm(right_wxyz, axis=-1, keepdims=True), 1e-12
                )
                left_rot = jaxlie.SO3(wxyz=left_wxyz)
                right_rot = jaxlie.SO3(wxyz=right_wxyz)
                rot = left_rot @ jaxlie.SO3.exp(
                    alpha * (left_rot.inverse() @ right_rot).log()
                )
                hinges = (1.0 - alpha) * q_left[:, 7:] + alpha * q_right[:, 7:]
                q_init = jp.concatenate([xyz, rot.wxyz, hinges], axis=-1)
                quat = q_init[:, 3:7]
                quat_norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
                q_init = q_init.at[:, 3:7].set(
                    quat / jp.where(quat_norm > 0, quat_norm, 1.0)
                )
            else:
                q_init = (1.0 - alpha) * q_left + alpha * q_right

            _, qpos, body_pos, body_quat, marker_pos, marker_error = (
                compute_stac.pose_optimization(
                    mjx_model,
                    mjx_data,
                    kp_chunk,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    q_init,
                    problem=problems[solve_frames],
                    acceleration_smoothness_weight=acceleration_smoothness_weight,
                    n_solver_max_iters=q_opt_cfg.ik_max_iterations,
                    initial_step_damping=q_opt_cfg.initial_step_damping,
                )
            )
            keep_start = context_frames
            q_keep = qpos[keep_start : keep_start + keep]
            all_qpos.append(q_keep)
            all_body_pos.append(body_pos[keep_start : keep_start + keep])
            all_body_quat.append(body_quat[keep_start : keep_start + keep])
            all_marker_pos.append(marker_pos[keep_start : keep_start + keep])
            all_errors.append(marker_error[keep_start : keep_start + keep])

            init_overlap = min(context_frames, int(q_keep.shape[0]))
            prev_overlap_q = q_keep[-init_overlap:] if init_overlap > 0 else None

            mjx_data = mjx_data.replace(qpos=q_keep[-1])
            mjx_data = utils.kinematics(mjx_model, mjx_data)
            mjx_data = utils.com_pos(mjx_model, mjx_data)

        qpos = jp.concatenate(all_qpos, axis=0)
        body_pos = jp.concatenate(all_body_pos, axis=0)
        body_quat = jp.concatenate(all_body_quat, axis=0)
        marker_pos = jp.concatenate(all_marker_pos, axis=0)
        frame_error = jp.concatenate(all_errors, axis=0)

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        return self._package_data(
            np.array(qpos),
            np.array(body_pos),
            np.array(body_quat),
            np.array(marker_pos),
            np.array(kp_data),
        )

    def _package_data(
        self,
        qpos: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        marker_pos: np.ndarray,
        kp_data: np.ndarray,
    ) -> io.StacData:
        """Package optimization results into a StacData structure.

        Args:
            qpos: Generalized coordinates per frame.
            body_pos: Body positions per frame.
            body_quat: Body quaternions per frame.
            marker_pos: Marker site positions per frame.
            kp_data: Keypoint data.

        Returns:
            Packaged STAC output data.
        """
        offsets = self._offsets
        offsets = np.array(offsets)
        kp_data = kp_data.reshape(-1, kp_data.shape[-1])

        return io.StacData(
            qpos=qpos,
            xpos=body_pos,
            xquat=body_quat,
            marker_sites=marker_pos,
            offsets=offsets,
            names_qpos=self._part_names,
            names_xpos=self._body_names,
            kp_data=kp_data,
            kp_names=self._kp_names,
        )

    def _build_render_model(
        self,
        offsets: Float[Array, "n_keypoints 3"],
        show_marker_error: bool,
    ) -> tuple[mujoco.MjModel, list[int]]:
        """Create a rendering model with keypoint and marker sites.

        Args:
            offsets: Marker offsets to apply to the rendering model.
            show_marker_error: Whether to add tendons showing marker-keypoint distance.

        Returns:
            Tuple of (compiled render model, keypoint site indices).
        """
        render_spec = self._build_body_spec()
        keypoint_site_names = []
        # set up keypoint rendering by adding the kp sites to the root body
        for name in self.cfg.model.KEYPOINT_MODEL_PAIRS:
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.cfg.model.KEYPOINT_COLOR_PAIRS[name]

            if isinstance(rgba, str):
                rgba = [float(c) for c in rgba.split(" ")]
            site_name = name + "_kp"
            keypoint_site_names.append(site_name)
            render_spec.worldbody.add_site(
                name=site_name,
                size=[self._marker_size] * 3,
                rgba=rgba,
                pos=start,
                group=2,
            )

        # Add body sites for new offsets
        offsets = np.asarray(offsets).reshape((-1, 3))
        for (key, v), pos in zip(self.cfg.model.KEYPOINT_MODEL_PAIRS.items(), offsets):
            parent = render_spec.body(v)
            parent.add_site(
                name=key + "_new",
                size=[self._marker_size] * 3,
                rgba=[0, 0, 0, 1],
                pos=pos,
                group=2,
            )

        # Tendons from new marker sites to kp
        if show_marker_error:
            for key, v in self.cfg.model.KEYPOINT_MODEL_PAIRS.items():
                tendon = render_spec.add_tendon(
                    name=key + "-" + v,
                    width=0.001,
                    rgba=[1.0, 0.0, 0.0, 1.0],
                    limited=0,
                )
                tendon.wrap_site(key + "_kp")
                tendon.wrap_site(key + "_new")

        render_mj_model = render_spec.compile()
        keypoint_site_idxs = [
            mujoco.mj_name2id(render_mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for site_name in keypoint_site_names
        ]
        return render_mj_model, keypoint_site_idxs

    def render(
        self,
        qpos: np.ndarray,
        kp_data: np.ndarray,
        offsets: np.ndarray,
        n_frames: int,
        save_path: str | Path,
        start_frame: int = 0,
        camera: int | str = 0,
        height: int = 1200,
        width: int = 1920,
        show_marker_error: bool = False,
    ) -> list[np.ndarray]:
        """Render fitted results as a video.

        Args:
            qpos: Joint angles per frame.
            kp_data: Mocap keypoint data per frame.
            offsets: Marker offsets.
            n_frames: Number of frames to render.
            save_path: Output video file path.
            start_frame: First frame to render.
            camera: MuJoCo camera name or index.
            height: Render height in pixels.
            width: Render width in pixels.
            show_marker_error: Whether to show marker-keypoint distance.

        Returns:
            List of rendered RGB frames.

        Raises:
            ValueError: If qpos/kp_data lengths mismatch or frame range is invalid.
        """
        if qpos.shape[0] != kp_data.shape[0]:
            raise ValueError(
                f"Length of qpos ({qpos.shape[0]}) is not equal to the length of kp_data({kp_data.shape[0]})"
            )
        if start_frame < 0 or start_frame > kp_data.shape[0]:
            raise ValueError(
                f"start_frame ({start_frame}) must be non-negative and less than the length of kp_data ({kp_data.shape[0]})"
            )
        if start_frame + n_frames > kp_data.shape[0]:
            raise ValueError(
                f"start_frame + n_frames ({start_frame} + {n_frames}) must be less than the length of given qpos and kp_data ({kp_data.shape[0]})"
            )

        render_mj_model, keypoint_site_idxs = self._build_render_model(
            offsets, show_marker_error
        )

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[1] = 0
        scene_option.geomgroup[2] = 1

        scene_option.sitegroup[2] = 1

        scene_option.sitegroup[3] = 0
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        scene_option.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
        mj_data = mujoco.MjData(render_mj_model)

        mujoco.mj_kinematics(render_mj_model, mj_data)

        renderer = mujoco.Renderer(render_mj_model, height=height, width=width)

        kp_data = kp_data[: qpos.shape[0]]

        kp_data = kp_data[start_frame : start_frame + n_frames]
        qpos = qpos[start_frame : start_frame + n_frames]

        frames = []
        with imageio.get_writer(save_path, fps=self.cfg.model.RENDER_FPS) as video:
            for qpos, kps in tqdm(zip(qpos, kp_data)):
                # Set keypoints--they're in cartesian space, but since they're attached to the worldbody they're the same as offsets
                render_mj_model.site_pos[keypoint_site_idxs] = np.reshape(kps, (-1, 3))
                mj_data.qpos = qpos

                mujoco.mj_fwdPosition(render_mj_model, mj_data)

                renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        return frames
