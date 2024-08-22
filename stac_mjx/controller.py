"""Utilities for mapping between mocap model and physics model."""

from jax import vmap
from jax import numpy as jp
from mujoco import mjx

import numpy as np

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

from stac_mjx import utils as utils
from stac_mjx import compute_stac
from stac_mjx import operations as op
from typing import List
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

    def get_part_ids(physics, parts: List) -> jp.ndarray:
        """Get the part ids for a given list of parts.

        Args:
            env (TYPE): Environment
            parts (List): List of part names

        Returns:
            jp.ndarray: an array of idxs
        """
        part_names = physics.named.data.qpos.axes.row.names
        return np.array([any(part in name for part in parts) for name in part_names])

    if utils.params["INDIVIDUAL_PART_OPTIMIZATION"] is None:
        indiv_parts = []
    else:
        indiv_parts = jp.array(
            [
                get_part_ids(physics, parts)
                for parts in utils.params["INDIVIDUAL_PART_OPTIMIZATION"].values()
            ]
        )

    utils.params["indiv_parts"] = indiv_parts


def compute_keypoint_centroid(kps):
    import numpy as np
    # Reshape the array from (1, 3N) to (N, 3)
    pts = kps.reshape(-1, 3)

    # Compute the centroid
    centroid = np.mean(pts, axis=0)

    return centroid


def rotate_points(points, alpha, theta):
    theta *= 3.1415/180
    alpha *= 3.1415/180    
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])


    # # Rotation matrix around the Y-axis (pitch)
    # R_y = np.array([
    #     [np.cos(phi), 0, np.sin(phi)],
    #     [0, 1, 0],
    #     [-np.sin(phi), 0, np.cos(phi)]
    # ])

    # Rotation matrix around the Z-axis (yaw)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Combined rotation: first Y-axis, then Z-axis
    R_combined = np.dot(R_z, R_x)

    # Rotate the points
    rotated_points = np.dot(points, R_combined.T)
    #rotate_points = np.dot(R_combined, points)

    return rotated_points

def create_keypoint_sites_centroid(root, kps):
    # Rotation
    #theta = np.radians(-50)
    #c, s = np.cos(theta), np.sin(theta)
    #R = np.array(((c, -s, 0), (s, c, 0), (0,0,1)))    
    
    # Compute the centroid    
    pts = kps.reshape(-1, 3)
    print("pts shape", pts.shape)
    centroid = np.mean(pts, axis=0)
    print("centroid", centroid)

    # Subtract centroid & rotate
    pts = pts - centroid
    #pts = rotate_points(pts, -50, -45)
    pts = rotate_points(pts, -55, -35)
    # Translation
    pts = pts - np.array([0, .05, 0])
    
    
    #pts = pts @ R.T
    
    keypoint_sites = []
    # set up keypoint rendering by adding the kp sites to the root body
    for id, name in enumerate(utils.params["KEYPOINT_MODEL_PAIRS"]):
        #start = (np.random.rand(3) - 0.5) * 0.001
    
        rgba = utils.params["KEYPOINT_COLOR_PAIRS"][name]
        
        # Keypoints
        site = root.worldbody.add(
            "site",
            name=name + "_kp",
            type="sphere",
            size="0.002",
            rgba=rgba,
            pos=pts[id],
            group=2,
        )
        keypoint_sites.append(site)

    physics = mjcf.Physics.from_mjcf_model(root)

    # return physics, mj_model, and sites (to use in bind())
    return physics, physics.model.ptr, keypoint_sites


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
            size="0.002",
            rgba=rgba,
            pos=start,
            group=2,
        )
        keypoint_sites.append(site)

    physics = mjcf.Physics.from_mjcf_model(root)

    # return physics, mj_model, and sites (to use in bind())
    return physics, physics.model.ptr, keypoint_sites


def set_keypoint_sites_centroid(physics, sites, kps):
    """Bind keypoint sites to physics model.

    Args:
        physics (_type_): dmcontrol physics object
        sites (_type_): _description_
        kps (_type_): _description_

    Returns:
        (dmcontrol.Physics, mujoco.Model): update physics and model with update site pos
    """
    theta = np.radians(35)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0,0,1)))    
    
    
    pts = kps.reshape(-1, 3)
    # Compute the centroid
    centroid = np.mean(pts, axis=0)
    print("centroid", centroid)
    pts = pts - centroid
    pts = pts @ R.T

    kps = pts.reshape(-1)


    physics.bind(sites).pos[:] = np.reshape(kps.T, (-1, 3))
    return physics, physics.model.ptr

def create_tendons(root: mjcf.Element):
    tendon_sites = []
    for key, v in utils.params["KEYPOINT_MODEL_PAIRS"].items():
        #pos = utils.params["KEYPOINT_INITIAL_OFFSETS"][key]
        rgba = utils.params["KEYPOINT_COLOR_PAIRS"][key]
        tendon = root.tendon.add(
            "spatial",
            name = key+"-"+v,
            width="0.0002",
            rgba = rgba,
            limited=False,
        )
        tendon.add("site",
                 site = key + "_kp")
        tendon.add("site",
                 site = key)
            
    physics = mjcf.Physics.from_mjcf_model(root)
    return physics, physics.model.ptr

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
        print("Parent v",v)
        pos = utils.params["KEYPOINT_INITIAL_OFFSETS"][key]
        site = parent.add(
            "site",
            name=key,
            type="sphere",
            size="0.001",
            rgba="0 0 0 1",
            pos=pos,
            group=2,
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
        key: int(axis.convert_key_item(key)) for key in utils.params["KEYPOINT_MODEL_PAIRS"].keys()
    }

    #print("site_index_map", utils.params["site_index_map"])

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


def get_error_stats(errors: jp.ndarray):
    """Compute error stats."""
    flattened_errors = errors.reshape(
        -1
    )  # -1 infers the size based on other dimensions
    # Calculate mean and standard deviation
    mean = jp.mean(flattened_errors)
    std = jp.std(flattened_errors)

    return flattened_errors, mean, std


# TODO: pmap fit and transform if you want to use it with multiple gpus
def fit(mj_model, kp_data):
    """Do pose optimization."""
    # Create mjx model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    # Get and set the offsets of the markers
    offsets = jp.copy(op.get_site_pos(mjx_model))
    offsets *= utils.params["SCALE_FACTOR"]

    mjx_model = op.set_site_pos(mjx_model, offsets)

    # forward is used to calculate xpos and such
    mjx_data = mjx.kinematics(mjx_model, mjx_data)
    mjx_data = mjx.com_pos(mjx_model, mjx_data)

    # Set joint bounds
    lb = jp.concatenate([-jp.inf * jp.ones(7), mjx_model.jnt_range[1:][:, 0]])
    lb = jp.minimum(lb, 0.0)
    ub = jp.concatenate([jp.inf * jp.ones(7), mjx_model.jnt_range[1:][:, 1]])
    utils.params["lb"] = lb
    utils.params["ub"] = ub

    # Begin optimization steps
    mjx_data = compute_stac.root_optimization(mjx_model, mjx_data, kp_data)

    for n_iter in range(utils.params["N_ITERS"]):
        print(f"Calibration iteration: {n_iter + 1}/{utils.params['N_ITERS']}")
        mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
            compute_stac.pose_optimization(mjx_model, mjx_data, kp_data)
        )

        for i, (t, e) in enumerate(zip(frame_time, frame_error)):
            print(f"Frame {i+1} done in {t} with a final error of {e}")

        flattened_errors, mean, std = get_error_stats(frame_error)
        # Print the results
        print(f"Flattened array shape: {flattened_errors.shape}")
        print(f"Mean: {mean}")
        print(f"Standard deviation: {std}")

        print("starting offset optimization")
        mjx_model, mjx_data = compute_stac.offset_optimization(
            mjx_model, mjx_data, kp_data, offsets, q
        )

    # Optimize the pose for the whole sequence
    print("Final pose optimization")
    mjx_data, q, walker_body_sites, x, frame_time, frame_error = (
        compute_stac.pose_optimization(mjx_model, mjx_data, kp_data)
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
        kp_data (jp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
        offsets (jp.ndarray): offsets loaded from offset.p after fit()
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
    vmap_root_opt = vmap(compute_stac.root_optimization)
    vmap_pose_opt = vmap(compute_stac.pose_optimization)

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


def package_data(mjx_model, physics, q, x, walker_body_sites, kp_data, batched=False):
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
        "names_qpos": utils.params["part_names"],
        "names_xpos": names_xpos,
        "kp_data": jp.copy(kp_data),
    }

    for k, v in utils.params.items():
        data[k] = v

    return data
