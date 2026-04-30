"""Stac class handling high level functionality of stac-mjx."""

import jax
from jax import Array
from jax import numpy as jp

import numpy as np

import mujoco
from mujoco import mjx

from stac_mjx import utils, rescale, compute_stac, io, stac_core

from omegaconf import DictConfig
from pathlib import Path
import imageio
from tqdm import tqdm

from jaxtyping import Float, Int, Bool
from jaxtyping import jaxtyped
from beartype import beartype

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
        self.stac_core_obj = None

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

        self._indiv_parts = self.part_opt_setup()

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

        self.stac_core_obj = stac_core.StacCore(
            self.cfg.model.FTOL, self.cfg.model.N_ITER_Q
        )

    def part_opt_setup(self) -> list[Bool[Array, " n_qpos"]]:
        """Set up joint masks for individual part optimization.

        Returns:
            List of boolean masks, one per part group.
        """

        def get_part_ids(parts: list[str]) -> Bool[Array, " n_qpos"]:
            return jp.array(
                [any(part in name for part in parts) for name in self._part_names]
            )

        if "INDIVIDUAL_PART_OPTIMIZATION" not in self.cfg.model:
            indiv_parts = []
        else:
            indiv_parts = jp.array(
                [
                    get_part_ids(parts)
                    for parts in self.cfg.model.INDIVIDUAL_PART_OPTIMIZATION.values()
                ]
            )

        return indiv_parts

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
    ) -> tuple[mujoco.MjModel, Int[Array, " n_keypoints"], Float[Array, "n_keypoints 3"]]:
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

    def _get_error_stats(
        self, errors: list
    ) -> tuple[np.ndarray, float, float]:
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

    @jaxtyped(typechecker=beartype)
    def fit_offsets(
        self, kp_data: Float[Array, "n_frames n_keypoints_xyz"]
    ) -> io.StacData:
        """Alternate between pose and offset optimization.

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

        if self._root_kp_idx == -1:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT not specified, skipping Root Optimization."
            )
        elif not self._fixed:
            mjx_data = compute_stac.root_optimization(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                kp_data,
                self._root_kp_idx,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._trunk_kps,
            )
        else:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT specified but model has fixed root, skipping Root Optimization"
            )

        for n_iter in range(self.cfg.model.N_ITERS):
            print(f"Calibration iteration: {n_iter + 1}/{self.cfg.model.N_ITERS}")
            mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error = (
                compute_stac.pose_optimization(
                    self.stac_core_obj,
                    mjx_model,
                    mjx_data,
                    kp_data,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    self._indiv_parts,
                )
            )

            flattened_errors, mean, std = self._get_error_stats(frame_error)
            print(f"Mean: {mean}")
            print(f"Standard deviation: {std}")

            mjx_model, mjx_data, self._offsets = compute_stac.offset_optimization(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                kp_data,
                self._offsets,
                qposes,
                self.cfg.model.N_SAMPLE_FRAMES,
                self._is_regularized,
                self._body_site_idxs,
                self.cfg.model.M_REG_COEF,
            )

        print("Final pose optimization", flush=True)
        mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error = (
            compute_stac.pose_optimization(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                kp_data,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._indiv_parts,
            )
        )

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")
        return self._package_data(
            mjx_model,
            np.array(qposes),
            np.array(xposes),
            np.array(xquats),
            np.array(marker_sites),
            np.array(kp_data),
        )

    @jaxtyped(typechecker=beartype)
    def ik_only(
        self,
        kp_data: Float[Array, "n_frames n_keypoints_xyz"],
        offsets: Float[Array, "n_keypoints 3"],
    ) -> io.StacData:
        """Run inverse kinematics only, using pre-fitted offsets.

        Stand-alone IK step for use after marker offsets have been determined
        by fit_offsets(). Useful when running IK on a different dataset than
        was used during fitting.

        Args:
            kp_data: Flattened keypoint data in meters.
            offsets: Marker offsets from a previous fit_offsets() run.

        Returns:
            Packaged STAC output data.
        """
        batched_kp_data = utils.batch_kp_data(
            kp_data,
            self.cfg.stac.n_frames_per_clip,
            continuous=self.cfg.stac.continuous,
        )

        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        def mjx_setup(kp_data, mj_model):
            """Create MJX model/data and set offsets."""
            mjx_model, mjx_data = utils.mjx_load(mj_model)

            mjx_model = utils.set_site_pos(mjx_model, offsets, self._body_site_idxs)

            mjx_data = mjx.kinematics(mjx_model, mjx_data)
            mjx_data = mjx.com_pos(mjx_model, mjx_data)

            return mjx_model, mjx_data

        mjx_model, mjx_data = jax.vmap(mjx_setup, in_axes=(0, None))(
            batched_kp_data, self._mj_model
        )

        if self._root_kp_idx == -1:
            print(
                "Missing or invalid ROOT_OPTIMIZATION_KEYPOINT, skipping root_optimization()"
            )
        elif self._mj_model.jnt_type[0] in (
            mujoco.mjtJoint.mjJNT_FREE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            vmap_root_opt = jax.vmap(
                compute_stac.root_optimization,
                in_axes=(None, 0, 0, 0, None, None, None, None, None),
            )
            mjx_data = vmap_root_opt(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                batched_kp_data,
                self._root_kp_idx,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._trunk_kps,
            )
        else:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT specified but model has fixed root, skipping root_optimization()"
            )

        vmap_pose_opt = jax.vmap(
            compute_stac.pose_optimization,
            in_axes=(None, 0, 0, 0, None, None, None, None),
        )
        mjx_data, qposes, xposes, xquats, marker_sites, frame_time, frame_error = (
            vmap_pose_opt(
                self.stac_core_obj,
                mjx_model,
                mjx_data,
                batched_kp_data,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._indiv_parts,
            )
        )

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        return self._package_data(
            mjx_model,
            np.array(qposes),
            np.array(xposes),
            np.array(xquats),
            np.array(marker_sites),
            np.array(batched_kp_data),
            batched=True,
        )

    def _package_data(
        self,
        mjx_model: mjx.Model,
        qposes: np.ndarray,
        xposes: np.ndarray,
        xquats: np.ndarray,
        marker_sites: np.ndarray,
        kp_data: np.ndarray,
        batched: bool = False,
    ) -> io.StacData:
        """Package optimization results into a StacData structure.

        Args:
            mjx_model: MJX model (used to extract offsets when batched).
            qposes: Generalized coordinates per frame.
            xposes: Body positions per frame.
            xquats: Body quaternions per frame.
            marker_sites: Marker site positions per frame.
            kp_data: Keypoint data (may be batched).
            batched: Whether the data has a batch dimension.

        Returns:
            Packaged STAC output data.
        """
        if batched:
            get_batch_offsets = jax.vmap(utils.get_site_pos, in_axes=(0, None))
            offsets = get_batch_offsets(mjx_model, self._body_site_idxs)[0]
            qposes = qposes.reshape(-1, qposes.shape[-1])
            xposes = xposes.reshape(-1, *xposes.shape[2:], order="F")
            xquats = xquats.reshape(-1, *xquats.shape[2:], order="F")
            marker_sites = marker_sites.reshape(-1, *marker_sites.shape[2:])
        else:
            offsets = self._offsets

        offsets = np.array(offsets)
        kp_data = kp_data.reshape(-1, kp_data.shape[-1])

        return io.StacData(
            qpos=qposes,
            xpos=xposes,
            xquat=xquats,
            marker_sites=marker_sites,
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

    @jaxtyped(typechecker=beartype)
    def render(
        self,
        qposes: Float[Array, "n_frames n_qpos"],
        kp_data: Float[Array, "n_frames n_keypoints_xyz"],
        offsets: Float[Array, "n_keypoints 3"],
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
            qposes: Joint angles per frame.
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
            ValueError: If qposes/kp_data lengths mismatch or frame range is invalid.
        """
        if qposes.shape[0] != kp_data.shape[0]:
            raise ValueError(
                f"Length of qposes ({qposes.shape[0]}) is not equal to the length of kp_data({kp_data.shape[0]})"
            )
        if start_frame < 0 or start_frame > kp_data.shape[0]:
            raise ValueError(
                f"start_frame ({start_frame}) must be non-negative and less than the length of kp_data ({kp_data.shape[0]})"
            )
        if start_frame + n_frames > kp_data.shape[0]:
            raise ValueError(
                f"start_frame + n_frames ({start_frame} + {n_frames}) must be less than the length of given qposes and kp_data ({kp_data.shape[0]})"
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

        kp_data = kp_data[: qposes.shape[0]]

        kp_data = kp_data[start_frame : start_frame + n_frames]
        qposes = qposes[start_frame : start_frame + n_frames]

        frames = []
        with imageio.get_writer(save_path, fps=self.cfg.model.RENDER_FPS) as video:
            for qpos, kps in tqdm(zip(qposes, kp_data)):
                # Set keypoints--they're in cartesian space, but since they're attached to the worldbody they're the same as offsets
                render_mj_model.site_pos[keypoint_site_idxs] = np.reshape(kps, (-1, 3))
                mj_data.qpos = qpos

                mujoco.mj_fwdPosition(render_mj_model, mj_data)

                renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        return frames
