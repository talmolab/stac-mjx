"""STAC class handling high level functionality of stac-mjx."""

import jax
from jax import numpy as jp

import numpy as np

import mujoco
from mujoco import mjx

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import enums

from stac_mjx import utils as utils
from stac_mjx import compute_stac
from stac_mjx import operations as op

from omegaconf import DictConfig
from typing import List, Union, Dict
from pathlib import Path
from copy import deepcopy

import imageio
from tqdm import tqdm

# Root = position (3) + quaternion (4)
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


class STAC:
    """Main class with key functionality for skeletal registration and rendering."""

    def __init__(
        self, xml_path: str, stac_cfg: DictConfig, model_cfg: Dict, kp_names: List[str]
    ):
        """Init STAC class, taking values from configs and creating values needed for stac.

        Args:
            xml_path (str): Path to model MJCF.
            stac_cfg (DictConfig): Stac config file.
            model_cfg (Dict): Model config file.
            kp_names (List[str]): Ordered list of mocap keypoint names.
        """
        self.stac_cfg = stac_cfg
        self.model_cfg = model_cfg
        self._kp_names = kp_names
        self._root = mjcf.from_path(xml_path)
        (
            self._mj_model,
            self._body_site_idxs,
            self._is_regularized,
        ) = self._create_body_sites(self._root)

        self._body_names = [
            self._mj_model.body(i).name for i in range(self._mj_model.nbody)
        ]

        joint_names = [self._mj_model.joint(i).name for i in range(self._mj_model.njnt)]

        # Set up bounds and part_names based on joint ranges, taking into account the dimensionality of parameters
        self._lb, self._ub, self._part_names = _align_joint_dims(
            self._mj_model.jnt_type, self._mj_model.jnt_range, joint_names
        )

        self._indiv_parts = self.part_opt_setup()

        self._trunk_kps = jp.array(
            [n in self.model_cfg["TRUNK_OPTIMIZATION_KEYPOINTS"] for n in kp_names],
        )

        self._mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[stac_cfg.mujoco.solver.lower()]

        self._mj_model.opt.iterations = stac_cfg.mujoco.iterations
        self._mj_model.opt.ls_iterations = stac_cfg.mujoco.ls_iterations

        # Runs faster on GPU with this
        self._mj_model.opt.jacobian = 0  # dense

    def part_opt_setup(self):
        """Set up the lists of indices for part optimization.

        Args:
            physics (dmcontrol.Physics): (See Mujoco dm_control docs)[https://github.com/google-deepmind/dm_control/blob/bdb1ab54c4c24cd89283fb18f06a6a54b6c0803b/dm_control/mjcf/physics.py#L434]
        """

        def get_part_ids(parts: List) -> jp.ndarray:
            """Get the part ids for a given list of parts."""
            return jp.array(
                [any(part in name for part in parts) for name in self._part_names]
            )

        if self.model_cfg["INDIVIDUAL_PART_OPTIMIZATION"] is None:
            indiv_parts = []
        else:
            indiv_parts = jp.array(
                [
                    get_part_ids(parts)
                    for parts in self.model_cfg["INDIVIDUAL_PART_OPTIMIZATION"].values()
                ]
            )

        return indiv_parts

    def _create_body_sites(self, root: mjcf.Element):
        """Create body site elements using dmcontrol mjcf for each keypoint.

        Args:
            root (mjcf.Element):

        Returns:
            dmcontrol.Physics, mujoco.Model:
        """
        for key, v in self.model_cfg["KEYPOINT_MODEL_PAIRS"].items():
            parent = root.find("body", v)
            pos = self.model_cfg["KEYPOINT_INITIAL_OFFSETS"][key]
            parent.add(
                "site",
                name=key,
                type="sphere",
                size=[0.005],
                rgba="0 0 0 1",
                pos=pos,
                group=3,
            )

        rescale.rescale_subtree(
            root,
            self.model_cfg["SCALE_FACTOR"],
            self.model_cfg["SCALE_FACTOR"],
        )
        physics = mjcf.Physics.from_mjcf_model(root)

        axis = physics.named.model.site_pos._axes[0]
        site_index_map = {
            key: int(axis.convert_key_item(key))
            for key in self.model_cfg["KEYPOINT_MODEL_PAIRS"].keys()
        }

        # Define which offsets to regularize
        is_regularized = []
        for k in site_index_map.keys():
            if any(n == k for n in self.model_cfg.get("SITES_TO_REGULARIZE", [])):
                is_regularized.append(jp.array([1.0, 1.0, 1.0]))
            else:
                is_regularized.append(jp.array([0.0, 0.0, 0.0]))
        is_regularized = jp.stack(is_regularized).flatten()

        return (
            # physics,
            physics.model.ptr,
            jp.array(list(site_index_map.values())),
            is_regularized,
        )

    def _chunk_kp_data(self, kp_data):
        """Reshape data for parallel processing."""
        n_frames = self.model_cfg["N_FRAMES_PER_CLIP"]
        total_frames = kp_data.shape[0]

        n_chunks = int(total_frames / n_frames)

        kp_data = kp_data[: int(n_chunks) * n_frames]

        # Reshape the array to create chunks
        kp_data = kp_data.reshape((n_chunks, n_frames) + kp_data.shape[1:])

        return kp_data

    def _get_error_stats(self, errors: jp.ndarray):
        """Compute error stats."""
        flattened_errors = errors.reshape(-1)

        # Calculate mean and standard deviation
        mean = jp.mean(flattened_errors)
        std = jp.std(flattened_errors)

        return flattened_errors, mean, std

    # TODO: pmap fit and transform if you want to use it with multiple gpus
    def fit(self, kp_data):
        """Alternate between pose and offset optimization for a set number of iterations.

        Args:
            kp_data (jp.ndarray): Mocap keypoints to fit to

        Returns:
            Dict: Output data packaged in a dictionary.
        """
        # Create mjx model and data
        mjx_model = mjx.put_model(self._mj_model)
        mjx_data = mjx.make_data(mjx_model)

        # Get and set the offsets of the markers
        self._offsets = jp.copy(op.get_site_pos(mjx_model, self._body_site_idxs))

        mjx_model = op.set_site_pos(mjx_model, self._offsets, self._body_site_idxs)

        # Calculate initial xpos and such
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)

        # Begin optimization steps
        # Skip root optimization if model is fixed (no free joint at root)
        if self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            mjx_data = compute_stac.root_optimization(
                mjx_model,
                mjx_data,
                kp_data,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._trunk_kps,
            )

        for n_iter in range(self.model_cfg["N_ITERS"]):
            print(f"Calibration iteration: {n_iter + 1}/{self.model_cfg['N_ITERS']}")
            mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
                compute_stac.pose_optimization(
                    mjx_model,
                    mjx_data,
                    kp_data,
                    self._lb,
                    self._ub,
                    self._body_site_idxs,
                    self._indiv_parts,
                )
            )

            for i, (t, e) in enumerate(zip(frame_time, frame_error)):
                print(f"Frame {i+1} done in {t} with a final error of {e}")

            flattened_errors, mean, std = self._get_error_stats(frame_error)
            # Print the results
            print(f"Flattened array shape: {flattened_errors.shape}")
            print(f"Mean: {mean}")
            print(f"Standard deviation: {std}")

            print("starting offset optimization")
            mjx_model, mjx_data = compute_stac.offset_optimization(
                mjx_model,
                mjx_data,
                kp_data,
                self._offsets,
                q,
                self.model_cfg["N_SAMPLE_FRAMES"],
                self._is_regularized,
                self._body_site_idxs,
                self.model_cfg["M_REG_COEF"],
            )

        # Optimize the pose for the whole sequence
        print("Final pose optimization")
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
            compute_stac.pose_optimization(
                mjx_model,
                mjx_data,
                kp_data,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._indiv_parts,
            )
        )

        for i, (t, e) in enumerate(zip(frame_time, frame_error)):
            print(f"Frame {i+1} done in {t} with a final error of {e}")

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")
        return self._package_data(mjx_model, q, x, walker_body_sites, kp_data)

    def transform(self, kp_data, offsets):
        """Register skeleton to keypoint data.

            Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Args:
            mj_model (mujoco.Model): Physics model.
            kp_data (jp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.
            offsets (jp.ndarray): offsets loaded from offset.p after fit()
        """
        # Create batches of kp_data
        batched_kp_data = self._chunk_kp_data(kp_data)

        # Create mjx model and data
        mjx_model = mjx.put_model(self._mj_model)
        mjx_data = mjx.make_data(mjx_model)

        def mjx_setup(kp_data, mj_model):
            """Create mjxmodel and mjxdata and set offet.

            Args:
                kp_data (_type_): _description_

            Returns:
                _type_: _description_
            """
            # Create mjx model and data
            mjx_model = mjx.put_model(mj_model)
            mjx_data = mjx.make_data(mjx_model)

            # Set the offsets.
            mjx_model = op.set_site_pos(mjx_model, offsets, self._body_site_idxs)

            # forward is used to calculate xpos and such
            mjx_data = mjx.kinematics(mjx_model, mjx_data)
            mjx_data = mjx.com_pos(mjx_model, mjx_data)

            return mjx_model, mjx_data

        mjx_model, mjx_data = jax.vmap(mjx_setup, in_axes=(0, None))(
            batched_kp_data, self._mj_model
        )

        # Vmap optimize functions
        vmap_root_opt = jax.vmap(
            compute_stac.root_optimization,
            in_axes=(0, 0, 0, None, None, None, None),
        )
        vmap_pose_opt = jax.vmap(
            compute_stac.pose_optimization,
            in_axes=(0, 0, 0, None, None, None, None),
        )

        # q_phase
        if self._mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            mjx_data = vmap_root_opt(
                mjx_model,
                mjx_data,
                batched_kp_data,
                self._lb,
                self._ub,
                self._body_site_idxs,
                self._trunk_kps,
            )
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = vmap_pose_opt(
            mjx_model,
            mjx_data,
            batched_kp_data,
            self._lb,
            self._ub,
            self._body_site_idxs,
            self._indiv_parts,
        )

        flattened_errors, mean, std = self._get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        return self._package_data(
            mjx_model, q, x, walker_body_sites, batched_kp_data, batched=True
        )

    def _package_data(self, mjx_model, q, x, walker_body_sites, kp_data, batched=False):
        """Extract pose, offsets, data, and all parameters.

        walker_body_sites is the marker positions for each frame--the rodent model's kp_data equivalent
        """
        if batched:
            # prepare batched data to be packaged
            get_batch_offsets = jax.vmap(op.get_site_pos, in_axes=(0, None))
            offsets = get_batch_offsets(mjx_model, self._body_site_idxs).copy()[0]
            x = x.reshape(-1, x.shape[-1])
            q = q.reshape(-1, q.shape[-1])
        else:
            offsets = op.get_site_pos(mjx_model, self._body_site_idxs).copy()

        kp_data = kp_data.reshape(-1, kp_data.shape[-1])

        data = {}

        for k, v in self.model_cfg.items():
            data[k] = v

        data.update(
            {
                "qpos": q,
                "xpos": x,
                "walker_body_sites": walker_body_sites,
                "offsets": offsets,
                "names_qpos": self._part_names,
                "names_xpos": self._body_names,
                "kp_data": jp.copy(kp_data),
                "kp_names": self._kp_names,
            }
        )

        return data

    def _create_keypoint_sites(self):
        """Create sites for keypoints (used for rendering only).

        Returns:
            (mujoco.Model, List, List): Mj_model for rendering, list of keypoint site indices, and list of body site indices
        """
        keypoint_sites = []
        keypoint_site_names = []
        # set up keypoint rendering by adding the kp sites to the root body
        for id, name in enumerate(self.model_cfg["KEYPOINT_MODEL_PAIRS"]):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.model_cfg["KEYPOINT_COLOR_PAIRS"][name]
            site_name = name + "_kp"
            keypoint_site_names.append(site_name)
            site = self._root.worldbody.add(
                "site",
                name=site_name,
                type="sphere",
                size=[0.005],
                rgba=rgba,
                pos=start,
                group=2,
            )
            keypoint_sites.append(site)

        physics = mjcf.Physics.from_mjcf_model(self._root)

        axis = physics.named.model.site_pos._axes[0]
        # Combine the two lists of site names and create the index map
        site_index_map = {
            key: int(axis.convert_key_item(key))
            for key in list(self.model_cfg["KEYPOINT_MODEL_PAIRS"].keys())
            + keypoint_site_names
        }
        body_site_idxs = [
            site_index_map[n] for n in self.model_cfg["KEYPOINT_MODEL_PAIRS"].keys()
        ]
        keypoint_site_idxs = [site_index_map[n] for n in keypoint_site_names]
        self._body_site_idxs = body_site_idxs
        self._keypoint_site_idxs = keypoint_site_idxs

        return deepcopy(physics.model.ptr), body_site_idxs, keypoint_site_idxs

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
            self._create_keypoint_sites()
        )
        render_mj_model.site_pos[body_site_idxs] = offsets

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = 1
        scene_option.sitegroup[2] = 1

        scene_option.sitegroup[3] = 1
        scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
        scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
        scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
        scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
        scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
        scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
        scene_option.flags[enums.mjtRndFlag.mjRND_FOG] = False

        mj_data = mujoco.MjData(render_mj_model)

        mujoco.mj_kinematics(render_mj_model, mj_data)

        renderer = mujoco.Renderer(render_mj_model, height=height, width=width)

        # slice kp_data to match qposes length
        kp_data = kp_data[: qposes.shape[0]]

        # Slice arrays to be the range that is being rendered
        kp_data = kp_data[start_frame : start_frame + n_frames]
        qposes = qposes[start_frame : start_frame + n_frames]

        frames = []
        # render while stepping using mujoco
        with imageio.get_writer(save_path, fps=self.model_cfg["RENDER_FPS"]) as video:
            for qpos, kps in tqdm(zip(qposes, kp_data)):
                # Set keypoints
                render_mj_model.site_pos[keypoint_site_idxs] = np.reshape(kps, (-1, 3))
                mj_data.qpos = qpos
                mujoco.mj_forward(render_mj_model, mj_data)

                renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        return frames
