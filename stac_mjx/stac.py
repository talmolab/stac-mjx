"""Stac class handling high level functionality of stac-mjx."""

import jax
from jax import numpy as jp

import numpy as np

import mujoco
from mujoco import mjx

from stac_mjx import utils, rescale, compute_stac, io, stac_core

from omegaconf import DictConfig
from typing import List, Union
from pathlib import Path
from copy import deepcopy
import imageio
from tqdm import tqdm

# """Stac class handling high level functionality of stac-mjx."""

_ROOT_QPOS_LB = jp.concatenate([-jp.inf * jp.ones(3), -1.0 * jp.ones(4)])
_ROOT_QPOS_UB = jp.concatenate([jp.inf * jp.ones(3), 1.0 * jp.ones(4)])

# mujoco jnt_type enums: https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtjoint
_MUJOCO_JOINT_TYPE_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


def _align_joint_dims(types, ranges, names):
    """Creates bounds and joint names aligned with qpos dimensions."""
    lb = []
    ub = []
    part_names = []
    for type, range, name in zip(types, ranges, names):
        dims = _MUJOCO_JOINT_TYPE_DIMS[type]
        # Set inf bounds for freejoint
        if type == mujoco.mjtJoint.mjJNT_FREE:
            lb.append(_ROOT_QPOS_LB)
            ub.append(_ROOT_QPOS_UB)
            part_names += [name] * dims
        else:
            lb.append(range[0] * jp.ones(dims))
            ub.append(range[1] * jp.ones(dims))
            part_names += [name] * dims

    return jp.minimum(jp.concatenate(lb), 0.0), jp.concatenate(ub), part_names


class Stac:
    """Main class with key functionality for skeletal registration and rendering."""

    def __init__(self, xml_path: str, cfg: DictConfig, kp_names: List[str]):
        """Init stac class, taking values from configs and creating values needed for stac.

        Args:
            xml_path (str): Path to model MJCF.
            cfg (DictConfig): Configs for this run.
            kp_names (List[str]): Ordered list of mocap keypoint names.
        """
        self.cfg = cfg
        self._kp_names = kp_names
        self._spec = mujoco.MjSpec.from_file(str(xml_path))
        self.stac_core_obj = None

        (
            self._mj_model,
            self._body_site_idxs,
            self._is_regularized,
        ) = self._create_body_sites(self._spec)

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

        # Runs faster on GPU with this
        self._mj_model.opt.jacobian = 0  # dense
        self._freejoint = bool(self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)

        self.stac_core_obj = stac_core.StacCore(self.cfg.model.FTOL)

    def part_opt_setup(self):
        """Set up the lists of indices for part optimization."""

        def get_part_ids(parts: List) -> jp.ndarray:
            """Get the part ids for a given list of parts."""
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

    def _create_body_sites(self, spec: mujoco.MjSpec):
        """Create body site elements using dmcontrol mjcf for each keypoint.

        Args:
            spec (mujoco.MjSpec):

        Returns:
            mujoco.Model, list of marker site indices, boolean mask for offset
            regularization, lists for part names and body names.
        """
        for key, v in self.cfg.model.KEYPOINT_MODEL_PAIRS.items():
            parent = spec.body(v)
            pos = self.cfg.model.KEYPOINT_INITIAL_OFFSETS[key]

            if isinstance(pos, str):
                pos = [float(p) for p in pos.split(" ")]

            parent.add_site(
                name=key,
                size=[0.005, 0.005, 0.005],
                rgba=(0, 0, 0, 0.8),
                pos=pos,
                group=3,
            )

        rescale.dm_scale_spec(spec, self.cfg.model.SCALE_FACTOR)
        model = self._spec.compile()

        site_index_map = {
            site.name: i
            for i, site in enumerate(self._spec.sites)
            if site.name in self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()
        }

        # Define which offsets to regularize
        is_regularized = []
        for k in site_index_map.keys():
            if any(n == k for n in self.cfg.model.get("SITES_TO_REGULARIZE", [])):
                is_regularized.append(jp.array([1.0, 1.0, 1.0]))
            else:
                is_regularized.append(jp.array([0.0, 0.0, 0.0]))
        is_regularized = jp.stack(is_regularized).flatten()
        body_site_idxs = jp.array(list(site_index_map.values()))
        return (
            model,
            body_site_idxs,
            is_regularized,
        )

    def _get_error_stats(self, errors: list):
        """Compute error stats."""
        flattened_errors = np.array(errors).reshape(-1)

        # Calculate mean and standard deviation
        mean = np.mean(flattened_errors)
        std = np.std(flattened_errors)

        return flattened_errors, mean, std

    def fit_offsets(self, kp_data):
        """Alternate between pose and offset optimization for a set number of iterations.

        Args:
            kp_data (jp.ndarray): Mocap keypoints to fit to

        Returns:
            Dict: Output data packaged in a dictionary.
        """
        # Create mjx model and data
        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        # Get and set the offsets of the markers
        self._offsets = jp.copy(utils.get_site_pos(mjx_model, self._body_site_idxs))

        mjx_model = utils.set_site_pos(mjx_model, self._offsets, self._body_site_idxs)

        # Calculate initial xpos and such
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)

        # Begin optimization steps
        # Skip root optimization if model is fixed (no free joint at root)
        if self._root_kp_idx == -1:
            print(
                "ROOT_OPTIMIZATION_KEYPOINT not specified, skipping Root Optimization."
            )
        elif self._freejoint:
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

            # for i, (t, e) in enumerate(zip(frame_time, frame_error)):
            #     print(f"Frame {i+1} done in {t} with a final error of {e}")

            flattened_errors, mean, std = self._get_error_stats(frame_error)
            # Print the results
            print(f"Mean: {mean}")
            print(f"Standard deviation: {std}")

            print("starting offset optimization", flush=True)
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

        # Optimize the pose for the whole sequence
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

        # for i, (t, e) in enumerate(zip(frame_time, frame_error)):
        #     print(f"Frame {i+1} done in {t} with a final error of {e}")

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        # Print the results
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

    def ik_only(self, kp_data, offsets):
        """Do only inverse kinematics (no fitting) on motion capture data.

            ik_only is a stand-alone inverse kinematics step to be used after the marker offsets
            have been determined by fit_offsets(). This is most useful when it is desired or necessary
            to run the fit on a different data set than was used during fit. (Otherwise, the output of fit_offsets()
            will contain identical data.)

        Args:
            mj_model (mujoco.Model): Physics model.
            kp_data (jp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.
            offsets (jp.ndarray): offsets loaded from offset.p after fit()
        """
        # Create batches of kp_data
        # TODO: add continuous option
        batched_kp_data = utils.batch_kp_data(
            kp_data,
            self.cfg.stac.n_frames_per_clip,
            continuous=self.cfg.stac.continuous,
        )

        # Create mjx model and data
        mjx_model, mjx_data = utils.mjx_load(self._mj_model)

        def mjx_setup(kp_data, mj_model):
            """Create mjxmodel and mjxdata and set offet.

            Args:
                kp_data (_type_): _description_

            Returns:
                _type_: _description_
            """
            # Create mjx model and data
            mjx_model, mjx_data = utils.mjx_load(mj_model)

            # Set the offsets.
            mjx_model = utils.set_site_pos(mjx_model, offsets, self._body_site_idxs)

            # forward is used to calculate xpos and such
            mjx_data = mjx.kinematics(mjx_model, mjx_data)
            mjx_data = mjx.com_pos(mjx_model, mjx_data)

            return mjx_model, mjx_data

        mjx_model, mjx_data = jax.vmap(mjx_setup, in_axes=(0, None))(
            batched_kp_data, self._mj_model
        )

        # q_phase - root
        if self._root_kp_idx == -1:
            print(
                "Missing or invalid ROOT_OPTIMIZATION_KEYPOINT, skipping root_optimization()"
            )
        elif self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
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

        # q_phase - pose
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
        # Print the results
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
        self, mjx_model, qposes, xposes, xquats, marker_sites, kp_data, batched=False
    ):
        """Extract pose, offsets, data, and all parameters.

        marker_sites is the marker positions for each frame--the rodent model's kp_data equivalent
        """
        if batched:
            # prepare batched data to be packaged
            get_batch_offsets = jax.vmap(utils.get_site_pos, in_axes=(0, None))
            offsets = get_batch_offsets(mjx_model, self._body_site_idxs)[0]
            qposes = qposes.reshape(-1, qposes.shape[-1])
            xposes = xposes.reshape(-1, *xposes.shape[2:], order="F")
            xquats = xquats.reshape(-1, *xquats.shape[2:], order="F")
            marker_sites = marker_sites.reshape(-1, *marker_sites.shape[2:])
        else:
            offsets = self._offsets.reshape((-1, 3))

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

    def _create_render_sites(self):
        """Create sites for keypoints (used for rendering only).

        Returns:
            (mujoco.Model, List, List): Mj_model for rendering, list of keypoint site indices, and list of body site indices
        """
        keypoint_sites = []
        keypoint_site_names = []
        # set up keypoint rendering by adding the kp sites to the root body
        for id, name in enumerate(self.cfg.model.KEYPOINT_MODEL_PAIRS):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.cfg.model.KEYPOINT_COLOR_PAIRS[name]

            if isinstance(rgba, str):
                rgba = [float(c) for c in rgba.split(" ")]
            site_name = name + "_kp"
            keypoint_site_names.append(site_name)
            site = self._spec.worldbody.add_site(
                name=site_name,
                size=[0.005, 0.005, 0.005],
                rgba=rgba,
                pos=start,
                group=2,
            )
            keypoint_sites.append(site)

        model = self._spec.compile()

        # Combine the two lists of site names and create the index map
        site_index_map = {
            site.name: i
            for i, site in enumerate(self._spec.sites)
            if site.name
            in list(self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()) + keypoint_site_names
        }
        body_site_idxs = [
            site_index_map[n] for n in self.cfg.model.KEYPOINT_MODEL_PAIRS.keys()
        ]
        keypoint_site_idxs = [site_index_map[n] for n in keypoint_site_names]

        self._body_site_idxs = body_site_idxs
        self._keypoint_site_idxs = keypoint_site_idxs
        return (deepcopy(model), body_site_idxs, keypoint_site_idxs)

    def render(
        self,
        qposes: jp.ndarray,
        kp_data: jp.ndarray,
        offsets: jp.ndarray,
        n_frames: int,
        save_path: Union[str, Path],
        start_frame: int = 0,
        camera: Union[int, str] = 0,
        height: int = 1200,
        width: int = 1920,
        show_marker_error: bool = False,
    ):
        """Creates rendering using the instantiated model, given the user's qposes and kp_data.

        Args:
            qposes (jp.ndarray): Set of model joint angles corresponding to kp_data.
            kp_data (jp.ndarray): Set of motion capture keypoints.
            offsets (jp.ndarray): array of marker offsets.
            n_frames (int): Number of frames to render.
            save_path (str): Path to save.
            start_frame (int, optional): Starting frame of qposes/kp_data to render at. Defaults to 0.
            camera (Union[int, str], optional): Mujoco camera name. Defaults to 0.
            height (int, optional): Height in pixels. Defaults to 1200.
            width (int, optional): Width in pixels. Defaults to 1920.
            show_marker_error (bool, optional): Show distance between marker and keypoint. Defaults to False.

        Raises:
            ValueError: qposes and kp_data must have same length (shape[0])
            ValueError: start_frame must be a non-negative value and within the length of kp_data/qposes
            ValueError: start_frame + n_frames must be within the length of kp_data/qposes

        Returns:
            List: List of rendered frames.
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

        render_mj_model, body_site_idxs, keypoint_site_idxs = (
            self._create_render_sites()
        )

        # Add body sites for new offsets
        for (key, v), pos in zip(
            self.cfg.model.KEYPOINT_MODEL_PAIRS.items(), offsets.reshape((-1, 3))
        ):
            parent = self._spec.body(v)
            parent.add_site(
                name=key + "_new",
                size=[0.005, 0.005, 0.005],
                rgba=[0, 0, 0, 1],
                pos=pos,
                group=2,
            )

        # Tendons from new marker sites to kp
        if show_marker_error:
            for key, v in self.cfg.model.KEYPOINT_MODEL_PAIRS.items():
                tendon = self._spec.add_tendon(
                    name=key + "-" + v,
                    width="0.001",
                    rgba=[255, 0, 0, 1],  # Red
                    limited=False,
                )
                tendon.wrap_site(key + "_kp")
                tendon.wrap_site(key + "_new")

        render_mj_model = deepcopy(self._spec.compile())

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

        # Slice kp_data to match qposes length
        kp_data = kp_data[: qposes.shape[0]]

        # Slice arrays to be the range that is being rendered
        kp_data = kp_data[start_frame : start_frame + n_frames]
        qposes = qposes[start_frame : start_frame + n_frames]

        frames = []
        # Render while stepping using mujoco
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
