"""Utilities for mapping between mocap model and physics model."""

from jax import vmap
from jax import numpy as jnp

from mujoco import mjx

import numpy as np

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

from stac_mjx import utils as utils
from stac_mjx import compute_stac
from stac_mjx import operations as op
from typing import List


class STAC:
    def __init__(self, xml_path, stac_cfg, model_cfg):
        self.stac_cfg = stac_cfg
        self.model_cfg = model_cfg

        root = mjcf.from_path(xml_path)
        physics, mj_model, self._site_index_map, self._part_names = (
            self.create_body_sites(root)
        )
        self._indiv_parts = self.part_opt_setup(physics)

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
                is_regularized.append(jnp.array([1.0, 1.0, 1.0]))
            else:
                is_regularized.append(jnp.array([0.0, 0.0, 0.0]))
        self._is_regularized = jnp.stack(is_regularized).flatten()

        # Create mjx model and data
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # Get and set the offsets of the markers
        self._offsets = jnp.copy(op.get_site_pos(self.mjx_model))
        self._offsets *= self.model_cfg["SCALE_FACTOR"]

        self.mjx_model = op.set_site_pos(self.mjx_model, self._offsets)

        # Calculate initial xpos and such
        self.mjx_data = mjx.kinematics(self.mjx_model, self.mjx_data)
        self.mjx_data = mjx.com_pos(self.mjx_model, self.mjx_data)

        # Set joint bounds
        self._lb = jnp.minimum(
            jnp.concatenate(
                [-jnp.inf * jnp.ones(7), self.mjx_model.jnt_range[1:][:, 0]]
            ),
            0.0,
        )
        self._ub = jnp.concatenate(
            [jnp.inf * jnp.ones(7), self.mjx_model.jnt_range[1:][:, 1]]
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

        def get_part_ids(physics, parts: List) -> jnp.ndarray:
            """Get the part ids for a given list of parts.

            Args:
                env (TYPE): Environment
                parts (List): List of part names

            Returns:
                jnp.ndarray: an array of idxs
            """
            part_names = physics.named.data.qpos.axes.row.names
            return np.array(
                [any(part in name for part in parts) for name in part_names]
            )

        if self.model_cfg["INDIVIDUAL_PART_OPTIMIZATION"] is None:
            indiv_parts = []
        else:
            indiv_parts = jnp.array(
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
        for id, name in enumerate(utils.params["KEYPOINT_MODEL_PAIRS"]):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = utils.params["KEYPOINT_COLOR_PAIRS"][name]
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

        return physics, physics.model.ptr, site_index_map, part_names

    def chunk_kp_data(self, kp_data):
        """Reshape data for parallel processing."""
        n_frames = self.model_cfg["N_FRAMES_PER_CLIP"]
        total_frames = kp_data.shape[0]

        n_chunks = int(total_frames / n_frames)

        kp_data = kp_data[: int(n_chunks) * n_frames]

        # Reshape the array to create chunks
        kp_data = kp_data.reshape((n_chunks, n_frames) + kp_data.shape[1:])

        return kp_data

    def get_error_stats(self, errors: jnp.ndarray):
        """Compute error stats."""
        flattened_errors = errors.reshape(-1)

        # Calculate mean and standard deviation
        mean = jnp.mean(flattened_errors)
        std = jnp.std(flattened_errors)

        return flattened_errors, mean, std

    # TODO: pmap fit and transform if you want to use it with multiple gpus
    def fit(self, kp_data):
        """Do pose optimization."""
        # Begin optimization steps
        mjx_data = compute_stac.root_optimization(
            mjx_model, mjx_data, kp_data, self._lb, self._ub
        )

        for n_iter in range(self.model_cfg["N_ITERS"]):
            print(f"Calibration iteration: {n_iter + 1}/{self.model_cfg["N_ITERS"]}")
            mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
                compute_stac.pose_optimization(
                    mjx_model, mjx_data, kp_data, self._lb, self._ub
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
            mjx_model, mjx_data = compute_stac.offset_optimization(
                mjx_model,
                mjx_data,
                kp_data,
                self._offsets,
                q,
                self.model_cfg["N_SAMPLE_FRAMES"],
                self._is_regularized,
            )

        # Optimize the pose for the whole sequence
        print("Final pose optimization")
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
            compute_stac.pose_optimization(
                mjx_model, mjx_data, kp_data, self._lb, self._ub
            )
        )

        for i, (t, e) in enumerate(zip(frame_time, frame_error)):
            print(f"Frame {i+1} done in {t} with a final error of {e}")

        flattened_errors, mean, std = self.get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")
        return mjx_model, q, x, walker_body_sites, kp_data

    def transform(self, mj_model, kp_data, offsets):
        """Register skeleton to keypoint data.

            Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Args:
            mj_model (mujoco.Model): Physics model.
            kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.
            offsets (jnp.ndarray): offsets loaded from offset.p after fit()
        """
        # Set joint bounds
        lb = jnp.concatenate([-jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 0]])
        lb = jnp.minimum(lb, 0.0)
        ub = jnp.concatenate([jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 1]])

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
            mjx_model = op.set_site_pos(mjx_model, offsets)

            # forward is used to calculate xpos and such
            mjx_data = mjx.kinematics(mjx_model, mjx_data)
            mjx_data = mjx.com_pos(mjx_model, mjx_data)

            return mjx_model, mjx_data

        vmap_mjx_setup = vmap(mjx_setup, in_axes=(0, None))

        # Create batch mjx model and data where batch_size = kp_data.shape[0]
        mjx_model, mjx_data = vmap_mjx_setup(kp_data, mj_model)

        # Vmap optimize functions
        vmap_root_opt = vmap(compute_stac.root_optimization)
        vmap_pose_opt = vmap(compute_stac.pose_optimization)

        # q_phase
        mjx_data = vmap_root_opt(mjx_model, mjx_data, kp_data, lb, ub)
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = vmap_pose_opt(
            mjx_model, mjx_data, kp_data, lb, ub
        )

        flattened_errors, mean, std = self.get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        return mjx_model, q, x, walker_body_sites, kp_data

    def package_data(
        self, mjx_model, physics, q, x, walker_body_sites, kp_data, batched=False
    ):
        """Extract pose, offsets, data, and all parameters."""
        if batched:
            # prepare batched data to be packaged
            get_batch_offsets = vmap(op.get_site_pos)
            offsets = get_batch_offsets(mjx_model).copy()[0]
            x = x.reshape(-1, x.shape[-1])
            q = q.reshape(-1, q.shape[-1])
        else:
            offsets = op.get_site_pos(mjx_model).copy()

        names_xpos = physics.named.data.xpos.axes.row.names

        kp_data = kp_data.reshape(-1, kp_data.shape[-1])
        data = {
            "qpos": q,
            "xpos": x,
            "walker_body_sites": walker_body_sites,
            "offsets": offsets,
            "names_qpos": self._part_names,
            "names_xpos": names_xpos,
            "kp_data": jnp.copy(kp_data),
        }

        for k, v in utils.params.items():
            data[k] = v

        return data
