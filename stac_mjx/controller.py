"""Utilities for mapping between mocap model and physics model."""

from jax import vmap
from jax import numpy as jp

import mujoco
from mujoco import mjx

import numpy as np

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

from stac_mjx import utils as utils
from stac_mjx import compute_stac
from stac_mjx import operations as op
from typing import List


class STAC:
    def __init__(self, xml_path: str, stac_cfg, model_cfg, kp_names: List[str]):
        self.stac_cfg = stac_cfg
        self.model_cfg = model_cfg

        root = mjcf.from_path(xml_path)
        physics, mj_model, self._site_index_map, self._part_names, self._body_names = (
            self.create_body_sites(root)
        )
        self._indiv_parts = self.part_opt_setup(physics)

        self._site_idxs = jp.array(list(self._site_index_map.values()))
        self._trunk_kps = jp.array(
            [n in self.model_cfg["TRUNK_OPTIMIZATION_KEYPOINTS"] for n in kp_names],
        )

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[stac_cfg.mujoco.solver.lower()]

        mj_model.opt.iterations = stac_cfg.mujoco.iterations
        mj_model.opt.ls_iterations = stac_cfg.mujoco.ls_iterations

        # Runs faster on GPU with this
        mj_model.opt.jacobian = 0  # dense

        # Define which offsets to regularize
        is_regularized = []
        for k in self._site_index_map.keys():
            if any(n == k for n in model_cfg.get("SITES_TO_REGULARIZE", [])):
                is_regularized.append(jp.array([1.0, 1.0, 1.0]))
            else:
                is_regularized.append(jp.array([0.0, 0.0, 0.0]))
        self._is_regularized = jp.stack(is_regularized).flatten()

        # Create mjx model and data
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # Get and set the offsets of the markers
        self._offsets = jp.copy(op.get_site_pos(self.mjx_model, self._site_idxs))
        self._offsets *= self.model_cfg["SCALE_FACTOR"]

        self.mjx_model = op.set_site_pos(self.mjx_model, self._offsets, self._site_idxs)

        # Calculate initial xpos and such
        self.mjx_data = mjx.kinematics(self.mjx_model, self.mjx_data)
        self.mjx_data = mjx.com_pos(self.mjx_model, self.mjx_data)

        # Set joint bounds
        self._lb = jp.minimum(
            jp.concatenate([-jp.inf * jp.ones(7), self.mjx_model.jnt_range[1:][:, 0]]),
            0.0,
        )
        self._ub = jp.concatenate(
            [jp.inf * jp.ones(7), self.mjx_model.jnt_range[1:][:, 1]]
        )

    def initialize_part_names(self, physics):
        """Get the ids of the limbs, accounting for quaternion and position."""
        part_names = physics.named.data.qpos.axes.row.names
        for _ in range(6):
            part_names.insert(0, part_names[0])
        return part_names

    def part_opt_setup(self, physics):
        """Set up the lists of indices for part optimization.

        Args:
            physics (dmcontrol.Physics): _description_
        """

        def get_part_ids(physics, parts: List) -> jp.ndarray:
            """Get the part ids for a given list of parts."""
            part_names = physics.named.data.qpos.axes.row.names
            return np.array(
                [any(part in name for part in parts) for name in part_names]
            )

        if self.model_cfg["INDIVIDUAL_PART_OPTIMIZATION"] is None:
            indiv_parts = []
        else:
            indiv_parts = jp.array(
                [
                    get_part_ids(physics, parts)
                    for parts in self.model_cfg["INDIVIDUAL_PART_OPTIMIZATION"].values()
                ]
            )

        return indiv_parts

    def create_keypoint_sites(self, root):
        """Create sites for keypoints (used for rendering).

        Args:
            root (mjcf.Element): root element of mjcf

        Returns:
            (dmcontrol.Physics, mujoco.Model, [mjcf.Element]): physics, mjmodel, and list of the created sites
        """
        keypoint_sites = []
        # set up keypoint rendering by adding the kp sites to the root body
        for id, name in enumerate(self.model_cfg["KEYPOINT_MODEL_PAIRS"]):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.model_cfg["KEYPOINT_COLOR_PAIRS"][name]
            site = root.worldbody.add(
                "site",
                name=name + "_kp",
                type="sphere",
                size=[0.005],
                rgba=rgba,
                pos=start,
                group=2,
            )
            keypoint_sites.append(site)

        physics = mjcf.Physics.from_mjcf_model(root)

        # return physics, mj_model, and sites (to use in bind())
        return physics, physics.model.ptr, keypoint_sites

    def set_keypoint_sites(self, physics, sites, kps):
        """Bind keypoint sites to physics model.

        Args:
            physics (_type_): dmcontrol physics object
            sites (_type_): _description_
            kps (_type_): _description_

        Returns:
            (dmcontrol.Physics, mujoco.Model): update physics and model with update site pos
        """
        physics.bind(sites).pos[:] = np.reshape(kps.T, (-1, 3))
        return physics, physics.model.ptr

    def create_body_sites(self, root: mjcf.Element):
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

        part_names = self.initialize_part_names(physics)

        body_names = physics.named.data.xpos.axes.row.names

        return physics, physics.model.ptr, site_index_map, part_names, body_names

    def chunk_kp_data(self, kp_data):
        """Reshape data for parallel processing."""
        n_frames = self.model_cfg["N_FRAMES_PER_CLIP"]
        total_frames = kp_data.shape[0]

        n_chunks = int(total_frames / n_frames)

        kp_data = kp_data[: int(n_chunks) * n_frames]

        # Reshape the array to create chunks
        kp_data = kp_data.reshape((n_chunks, n_frames) + kp_data.shape[1:])

        return kp_data

    def get_error_stats(self, errors: jp.ndarray):
        """Compute error stats."""
        flattened_errors = errors.reshape(-1)

        # Calculate mean and standard deviation
        mean = jp.mean(flattened_errors)
        std = jp.std(flattened_errors)

        return flattened_errors, mean, std

    # TODO: pmap fit and transform if you want to use it with multiple gpus
    def fit(self, kp_data):
        """Do pose optimization."""
        # Begin optimization steps
        self.mjx_data = compute_stac.root_optimization(
            self.mjx_model,
            self.mjx_data,
            kp_data,
            self._lb,
            self._ub,
            self._site_idxs,
            self._trunk_kps,
        )

        for n_iter in range(self.model_cfg["N_ITERS"]):
            print(f"Calibration iteration: {n_iter + 1}/{self.model_cfg['N_ITERS']}")
            self.mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
                compute_stac.pose_optimization(
                    self.mjx_model,
                    self.mjx_data,
                    kp_data,
                    self._lb,
                    self._ub,
                    self._site_idxs,
                    self._indiv_parts,
                )
            )

            for i, (t, e) in enumerate(zip(frame_time, frame_error)):
                print(f"Frame {i+1} done in {t} with a final error of {e}")

            flattened_errors, mean, std = self.get_error_stats(frame_error)
            # Print the results
            print(f"Flattened array shape: {flattened_errors.shape}")
            print(f"Mean: {mean}")
            print(f"Standard deviation: {std}")

            print("starting offset optimization")
            self.mjx_model, self.mjx_data = compute_stac.offset_optimization(
                self.mjx_model,
                self.mjx_data,
                kp_data,
                self._offsets,
                q,
                self.model_cfg["N_SAMPLE_FRAMES"],
                self._is_regularized,
                self._site_idxs,
                self.model_cfg["M_REG_COEF"],
            )

        # Optimize the pose for the whole sequence
        print("Final pose optimization")
        self.mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
            compute_stac.pose_optimization(
                self.mjx_model,
                self.mjx_data,
                kp_data,
                self._lb,
                self._ub,
                self._site_idxs,
                self._indiv_parts,
            )
        )

        for i, (t, e) in enumerate(zip(frame_time, frame_error)):
            print(f"Frame {i+1} done in {t} with a final error of {e}")

        flattened_errors, mean, std = self.get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")
        return q, x, walker_body_sites, kp_data

    def transform(self, mj_model, kp_data, offsets):
        """Register skeleton to keypoint data.

            Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Args:
            mj_model (mujoco.Model): Physics model.
            kp_data (jp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.
            offsets (jp.ndarray): offsets loaded from offset.p after fit()
        """
        # Set joint bounds
        lb = jp.concatenate([-jp.inf * jp.ones(7), self.mjx_model.jnt_range[1:][:, 0]])
        lb = jp.minimum(lb, 0.0)
        ub = jp.concatenate([jp.inf * jp.ones(7), self.mjx_model.jnt_range[1:][:, 1]])

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
            mjx_model = op.set_site_pos(mjx_model, offsets, self._site_idxs)

            # forward is used to calculate xpos and such
            mjx_data = mjx.kinematics(mjx_model, mjx_data)
            mjx_data = mjx.com_pos(mjx_model, mjx_data)

            return mjx_model, mjx_data

        vmap_mjx_setup = vmap(mjx_setup, in_axes=(0, None))

        # Create batch mjx model and data where batch_size = kp_data.shape[0]
        self.mjx_model, mjx_data = vmap_mjx_setup(kp_data, mj_model)

        # Vmap optimize functions
        vmap_root_opt = vmap(compute_stac.root_optimization)
        vmap_pose_opt = vmap(compute_stac.pose_optimization)

        # q_phase
        mjx_data = vmap_root_opt(
            self.mjx_model, self.mjx_data, kp_data, self._lb, self._ub, self._site_idxs
        )
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = vmap_pose_opt(
            mjx_model, mjx_data, kp_data, lb, ub, self._site_idxs, self._indiv_parts
        )

        flattened_errors, mean, std = self.get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        return self.mjx_model, q, x, walker_body_sites, kp_data

    def package_data(self, q, x, walker_body_sites, kp_data, batched=False):
        """Extract pose, offsets, data, and all parameters.
        walker_body_sites is the marker positions for each frame--
            the rodent model's kp_data equivalent
        """
        if batched:
            # prepare batched data to be packaged
            get_batch_offsets = vmap(op.get_site_pos)
            offsets = get_batch_offsets(self.mjx_model).copy()[0]
            x = x.reshape(-1, x.shape[-1])
            q = q.reshape(-1, q.shape[-1])
        else:
            offsets = op.get_site_pos(self.mjx_model, self._site_idxs).copy()

        kp_data = kp_data.reshape(-1, kp_data.shape[-1])
        data = {
            "qpos": q,
            "xpos": x,
            "walker_body_sites": walker_body_sites,
            "offsets": offsets,
            "names_qpos": self._part_names,
            "names_xpos": self._body_names,
            "kp_data": jp.copy(kp_data),
        }

        for k, v in self.model_cfg.items():
            data[k] = v

        return data
