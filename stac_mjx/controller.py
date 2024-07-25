"""Utilities for mapping between mocap model and physics model."""

from jax import vmap
from jax import numpy as jnp

import mujoco
from mujoco import mjx

import numpy as np

from typing import Text

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

import utils
from compute_stac import *
import operations as op

import pickle
import logging
import os
from statistics import fmean, pstdev


def initialize_part_names(physics):
    """Get the ids of the limbs, accounting for quaternion and position."""
    part_names = physics.named.data.qpos.axes.row.names
    for _ in range(6):
        part_names.insert(0, part_names[0])
    return part_names


def part_opt_setup(physics) -> None:
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
        return np.array([any(part in name for part in parts) for name in part_names])

    if utils.params["INDIVIDUAL_PART_OPTIMIZATION"] is None:
        indiv_parts = []
    else:
        indiv_parts = jnp.array(
            [
                get_part_ids(physics, parts)
                for parts in utils.params["INDIVIDUAL_PART_OPTIMIZATION"].values()
            ]
        )

    utils.params["indiv_parts"] = indiv_parts


def create_keypoint_sites(root):
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


def set_keypoint_sites(physics, sites, kps):
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


def create_body_sites(root: mjcf.Element):
    """Create body site elements using dmcontrol mjcf for each keypoint.

    Args:
        root (mjcf.Element):

    Returns:
        dmcontrol.Physics, mujoco.Model:
    """
    body_sites = []
    for key, v in utils.params["KEYPOINT_MODEL_PAIRS"].items():
        parent = root.find("body", v)
        pos = utils.params["KEYPOINT_INITIAL_OFFSETS"][key]
        site = parent.add(
            "site",
            name=key,
            type="sphere",
            size=[0.005],
            rgba="0 0 0 1",
            pos=pos,
            group=3,
        )
        body_sites.append(site)

    rescale.rescale_subtree(
        root,
        utils.params["SCALE_FACTOR"],
        utils.params["SCALE_FACTOR"],
    )
    physics = mjcf.Physics.from_mjcf_model(root)
    # Usage of physics: binding = physics.bind(body_sites)

    axis = physics.named.model.site_pos._axes[0]
    utils.params["site_index_map"] = {
        key: int(axis.convert_key_item(key))
        for key in utils.params["KEYPOINT_MODEL_PAIRS"].keys()
    }

    utils.params["part_names"] = initialize_part_names(physics)

    return physics, physics.model.ptr


def chunk_kp_data(kp_data):
    """Reshape data for parallel processing."""
    n_frames = utils.params["N_FRAMES_PER_CLIP"]
    total_frames = kp_data.shape[0]

    n_chunks = int(total_frames / n_frames)

    kp_data = kp_data[: int(n_chunks) * n_frames]

    # Reshape the array to create chunks
    kp_data = kp_data.reshape((n_chunks, n_frames) + kp_data.shape[1:])

    return kp_data


def get_error_stats(errors: jnp.ndarray):
    """Compute error stats."""
    flattened_errors = errors.reshape(
        -1
    )  # -1 infers the size based on other dimensions
    # Calculate mean and standard deviation
    mean = jnp.mean(flattened_errors)
    std = jnp.std(flattened_errors)

    return flattened_errors, mean, std


# TODO: pmap fit and transform if you want to use it with multiple gpus
def fit(mj_model, kp_data):
    """Do pose optimization."""
    # Create mjx model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    # Get and set the offsets of the markers
    offsets = jnp.copy(op.get_site_pos(mjx_model))
    offsets *= utils.params["SCALE_FACTOR"]

    # print(mjx_model.site_pos, mjx_model.site_pos.shape)
    mjx_model = op.set_site_pos(mjx_model, offsets)

    # forward is used to calculate xpos and such
    mjx_data = mjx.kinematics(mjx_model, mjx_data)
    mjx_data = mjx.com_pos(mjx_model, mjx_data)

    # Set joint bounds
    lb = jnp.concatenate([-jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 0]])
    lb = jnp.minimum(lb, 0.0)
    ub = jnp.concatenate([jnp.inf * jnp.ones(7), mjx_model.jnt_range[1:][:, 1]])
    utils.params["lb"] = lb
    utils.params["ub"] = ub

    # Begin optimization steps
    mjx_data = root_optimization(mjx_model, mjx_data, kp_data)

    for n_iter in range(utils.params["N_ITERS"]):
        print(f"Calibration iteration: {n_iter + 1}/{utils.params['N_ITERS']}")
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = pose_optimization(
            mjx_model, mjx_data, kp_data
        )

        for i, (t, e) in enumerate(zip(frame_time, frame_error)):
            print(f"Frame {i+1} done in {t} with a final error of {e}")

        flattened_errors, mean, std = get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        print("starting offset optimization")
        mjx_model, mjx_data = offset_optimization(
            mjx_model, mjx_data, kp_data, offsets, q
        )

    # Optimize the pose for the whole sequence
    print("Final pose optimization")
    mjx_data, q, walker_body_sites, x, frame_time, frame_error = pose_optimization(
        mjx_model, mjx_data, kp_data
    )

    for i, (t, e) in enumerate(zip(frame_time, frame_error)):
        print(f"Frame {i+1} done in {t} with a final error of {e}")

    flattened_errors, mean, std = get_error_stats(frame_error)
    # Print the results
    print(f"Flattened array shape: {flattened_errors.shape}")
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")
    return mjx_model, q, x, walker_body_sites, kp_data


def transform(mj_model, kp_data, offsets):
    """Register skeleton to keypoint data.

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

    Args:
        mj_model (mujoco.Model): Physics model.
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
        offsets (jnp.ndarray): offsets loaded from offset.p after fit()
    """
    # physics, mj_model = set_body_sites(root)
    # utils.params["mj_model"] = mj_model
    # part_opt_setup(physics)

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
        # do initial get_site stuff inside mjx_setup

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
    vmap_root_opt = vmap(root_optimization)
    vmap_pose_opt = vmap(pose_optimization)

    # q_phase
    mjx_data = vmap_root_opt(mjx_model, mjx_data, kp_data)
    mjx_data, q, walker_body_sites, x, frame_time, frame_error = vmap_pose_opt(
        mjx_model, mjx_data, kp_data
    )

    flattened_errors, mean, std = get_error_stats(frame_error)
    # Print the results
    print(f"Flattened array shape: {flattened_errors.shape}")
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")

    return mjx_model, q, x, walker_body_sites, kp_data
