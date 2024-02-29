from jax import vmap
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from typing import Text
import utils
from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
import numpy as np
from compute_stac import *
import operations as op
import pickle
import os

"""
This file should serve the same purpose as the logic for executing SLURM jobs. 
However, instead of executing SLURM jobs for individual clips, it:
1. creates the single mjxModel, N mjxDatas, N kp_datas, N body_sites 
   (kp_data and body_sites are both np arrays so are jax compatible. Only question is how to get then all in the cheapest way)
2. Executes the functions called in fit() and transform(). 
Esentially, we are moving the preprocessing functions and fit() and transform() here.
    Compute_stac.py retains the intermediary functions like root_optimization() and pose_optimization()
Unlike old stac, all data needs to be passed in as arguments to functions 
    since we want to have a vectorized set of multiple data instances to be passed into vmapped functions
"""

def part_opt_setup(physics):
    def get_part_ids(physics, parts: List) -> jnp.ndarray:
        """Get the part ids given a list of parts.
        This code creates a JAX NumPy-like Boolean array where each element 
        represents whether any of the strings in the parts list is found as a substring in 
        the corresponding name from the part_names list.
            Args:
                env (TYPE): Environment
                parts (List): List of part names

            Returns:
                jnp.ndarray: Array of part identifiers
        """
        part_names = physics.named.data.qpos.axes.row.names
        return np.array([any(part in name for part in parts) for name in part_names])

    if utils.params["INDIVIDUAL_PART_OPTIMIZATION"] is None:
        indiv_parts = []
    else:
        indiv_parts = jnp.array([
            get_part_ids(physics, parts)
            for parts in utils.params["INDIVIDUAL_PART_OPTIMIZATION"].values()
        ])
    
    utils.params["indiv_parts"] = indiv_parts


def create_keypoint_sites(root):
    keypoint_sites = []
    # set up keypoint rendering by adding the kp sites to the body
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
    """

    Args:
        physics (_type_): _description_
        sites (_type_): _description_
        kps (_type_): _description_
    """
    physics.bind(sites).pos[:] = np.reshape(kps.T, (-1,3))
    return physics, physics.model.ptr


def create_body_sites(root):
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
    utils.params["site_index_map"] = {key: int(axis.convert_key_item(key)) for key in utils.params["KEYPOINT_MODEL_PAIRS"].keys()}
    
    utils.params["part_names"] = initialize_part_names(physics)

    return physics, physics.model.ptr

def prep_kp_data(kp_data, stac_keypoint_order):
    """Data preparation for kp_data: splits kp_data into 1k frame chunks.
        Makes sure that the total chunks is divisible by the number of gpus
        not vmapped but essentially vectorizes kpdata via chunking

    Returns:
        model: mjModel
        mjx_data: MjData with batch dimension; (batch, dim of attr)
        kp_data: kp_data chunks with a leading batch dimension
    """
    
    kp_data = jnp.array(kp_data[:, :, stac_keypoint_order])
    kp_data = jnp.transpose(kp_data, (0, 2, 1))
    kp_data = jnp.reshape(kp_data, (kp_data.shape[0], -1))
    
    return kp_data 

def chunk_kp_data(kp_data):
    n_frames = utils.params['N_FRAMES_PER_CLIP']
    total_frames = kp_data.shape[0]

    n_chunks = int((total_frames / n_frames) // utils.params['N_GPUS'] * utils.params['N_GPUS'])
    
    kp_data = kp_data[:int(n_chunks) * n_frames]
    
    # Reshape the array to create chunks
    kp_data = kp_data.reshape((n_chunks, n_frames) + kp_data.shape[1:])
    
    return kp_data

    
# TODO: pmap fit and transform if you want to use it with multiple gpus
def fit(mj_model, kp_data):
    
    # Create mjx model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)
    
    # Get and set the offsets of the markers
    offsets = jnp.copy(op.get_site_pos(mjx_model))
    offsets *= utils.params['SCALE_FACTOR']
    
    # print(mjx_model.site_pos, mjx_model.site_pos.shape)
    mjx_model = op.set_site_pos(mjx_model, offsets)

    # forward is used to calculate xpos and such
    mjx_data = mjx.kinematics(mjx_model, mjx_data)
    mjx_data = mjx.com_pos(mjx_model, mjx_data)
    
    # Begin optimization steps
    mjx_data = root_optimization(mjx_model, mjx_data, kp_data)

    for n_iter in range(utils.params['N_ITERS']):
        print(f"Calibration iteration: {n_iter + 1}/{utils.params['N_ITERS']}")
        mjx_data, q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, kp_data)
        print("starting offset optimization")
        mjx_model, mjx_data = offset_optimization(mjx_model, mjx_data, kp_data, offsets, q)

    # Optimize the pose for the whole sequence
    print("Final pose optimization")
    mjx_data, q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, kp_data)
       
    return mjx_model, q, x, walker_body_sites, kp_data


def transform(mj_model, kp_data, offsets):
    """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

    Args:
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
        offsets (jnp.ndarray): offsets loaded from offset.p after fit()
    """
    
    # physics, mj_model = set_body_sites(root)
    # utils.params["mj_model"] = mj_model
    # part_opt_setup(physics)

    def mjx_setup(kp_data, mj_model):
        """creates mjxmodel and mjxdata, setting offets 

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
    mjx_data, q, walker_body_sites, x = vmap_pose_opt(mjx_model, mjx_data, kp_data)

    return mjx_model, q, x, walker_body_sites, kp_data

    
def save(fit_data, save_path: Text):
    """Save data.

    Args:
        save_path (Text): Path to save data. Defaults to None.
    """
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _, file_extension = os.path.splitext(save_path)
    if file_extension == ".p":
        with open(save_path, "wb") as output_file:
            pickle.dump(fit_data, output_file, protocol=2)
    else:
        with open(save_path + ".p", "wb") as output_file:
            pickle.dump(fit_data, output_file, protocol=2)