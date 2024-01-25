import jax 
from jax import jit, vmap
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from typing import Text, Dict
import utils
from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
import numpy as np
import functools
from compute_stac import *
import stac_base
import pickle
import os
import time

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
    
# root is modified in place
def set_body_sites(root):
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

    # Set params fields instead of returning
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
    
    return kp_data, n_chunks

# TODO: pmap fit and transform if you want to use it with multiple gpus
# May not be necessary?
def fit(root, kp_data):
    """Calibrate and fit the model to keypoints.
    Performs three rounds of alternating marker and quaternion optimization. Optimal
    results with greater than 200 frames of data in which the subject is moving.
    
    Args:
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
    
    Returns: fitted model props?? find relevant props from stac object

    """    
    
    physics, mj_model = set_body_sites(root)
    # utils.params["mj_model"] = mj_model
    part_opt_setup(physics)
    
    @vmap
    def mjx_setup(kp_data):
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
        
        # Get and set the offsets of the markers
        offsets = np.copy(stac_base.get_site_pos(mjx_model))
        offsets *= utils.params['SCALE_FACTOR']
        
        # print(mjx_model.site_pos, mjx_model.site_pos.shape)
        mjx_model = stac_base.set_site_pos(mjx_model, offsets) 

        # forward is used to calculate xpos and such
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)
        return mjx_model, mjx_data, offsets

    # Create batch mjx model and data where batch_size = kp_data.shape[0]
    # mjx_model, mjx_data, offsets = jax.vmap(lambda x: mjx_setup(x, mj_model))(kp_data)
    mjx_model, mjx_data, offsets = mjx_setup(kp_data)
    # for n_site, p in enumerate(physics.bind(body_sites).pos):
    #     body_sites[n_site].pos = p
    
    mjx_data = root_optimization(mjx_model, mjx_data, kp_data)
    for n_iter in range(utils.params['N_ITERS']):
        print(f"Calibration iteration: {n_iter + 1}/{utils.params['N_ITERS']}")
        mjx_data, q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, kp_data)
        print("starting offset optimization")
        mjx_model, mjx_data = offset_optimization(mjx_model, mjx_data, kp_data, offsets, q)

    # Optimize the pose for the whole sequence
    print("Final pose optimization")
    q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data)
    
    data = package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data
    )
    return data

def test_opt(root, kp_data):
    physics, mj_model = set_body_sites(root)
    utils.params["mj_model"] = mj_model
    part_opt_setup(physics)
    
    @vmap
    def mjx_setup(kp_data):
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
        
        # Get and set the offsets of the markers
        offsets = jnp.copy(stac_base.get_site_pos(mjx_model))
        offsets *= utils.params['SCALE_FACTOR']
        
        # print(mjx_model.site_pos, mjx_model.site_pos.shape)
        mjx_model = stac_base.set_site_pos(mjx_model, offsets) 

        # forward is used to calculate xpos and such
        mjx_data = mjx.kinematics(mjx_model, mjx_data)
        mjx_data = mjx.com_pos(mjx_model, mjx_data)

        return mjx_model, mjx_data, offsets
    
    # Create batch mjx model and data where batch_size = kp_data.shape[0]
    mjx_model, mjx_data, offsets = mjx_setup(kp_data)
    # This used to set the walker body sites based on the body site positions in 'physics'. 
    # Do we still need to do this? 
    # for n_site, p in enumerate(physics.bind(body_sites).pos):
    #     body_sites[n_site].pos = p
    
    # Optimize
    mjx_data = root_optimization(mjx_model, mjx_data, kp_data)
    mjx_data, q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, kp_data)
    
    # Saving the post_opt data in a pickle
    data = {
        "kp_data": kp_data,
        "mjx_model": mjx_model,
        "mjx_data": mjx_data,
        "q": q,
        "offsets": offsets,
        "walker_body_sites": walker_body_sites,
        "x": x,
        "physics": physics,
        "site_index_map": utils.params["site_index_map"],
    }
    save_path = "pose_opt_qs.p"
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as output_file:
        pickle.dump(data, output_file, protocol=2)
        
    mjx_model, mjx_data = offset_optimization(mjx_model, mjx_data, kp_data, offsets, q)
    data = package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data
    )
    return data
    # return None
def transform(kp_data, offset_path):
    """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

    Args:
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
        offset_path (Text): Path to offset file saved after .fit()
    """
    # kp_data = self._prepare_data(kp_data)
    # self.n_frames = kp_data.shape[0]
    # env = build_env(kp_data, self._properties)
    # part_names = initialize_part_names(env)

    # # Set the offsets.
    # self.offset_path = offset_path
    # with open(self.offset_path, "rb") as f:
    #     in_dict = pickle.load(f)
    # sites = env.task._walker.body_sites
    # env.physics.bind(sites).pos[:] = in_dict["offsets"]
    # for n_site, p in enumerate(env.physics.bind(sites).pos):
    #     sites[n_site].pos = p

    # # Optimize the root position
    # root_optimization(env, self._properties)

    # # Optimize the pose for the whole sequence
    # q, walker_body_sites, x = pose_optimization(env, self._properties)

    # # Extract pose, offsets, data, and all parameters
    # data = package_data(
    #     env, q, x, walker_body_sites, part_names, kp_data, self._properties
    # )
    # return data
    return

def end_to_end():
    """this function runs fit and transform end to end for conceptualizing purposes
    """
    # params = utils.load_params("params/params.yaml")
    utils.init_params("params/params.yaml")
    model = mujoco.MjModel.from_xml_path(utils.params["XML_PATH"])
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 1
    model.opt.ls_iterations = 1
    
    offset_path = "offset.p"

    root = mjcf.from_path(utils.params["XML_PATH"])

    n_frames = None

    # Default ordering of mj sites is alphabetical, so we reorder to match
    kp_names = utils.loadmat(utils.params["SKELETON_PATH"])["joint_names"]
    utils.params["kp_names"] = kp_names
    
    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)

    # kp_data
    # TODO: store kp_data used in fit in another variable (small slice of kpdata)
    kp_data = prep_kp_data(kp_data, stac_keypoint_order)
    # chunk it to pass int vmapped functions
    kp_data, n_envs = chunk_kp_data(kp_data)
    # fit
    fit_data = fit(root, kp_data)

    save(fit_data, offset_path)
    # transform
    # transform_data = transform()
    return
    
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
    elif file_extension == ".mat":
        savemat(save_path, fit_data)