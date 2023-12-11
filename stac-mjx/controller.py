import jax 
from jax import jit, vmap
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from typing import Text, Dict
import utils
from dm_control import mjcf
from dm_control.locomotion.walkers import rescale

from compute_stac import root_optimization, pose_optimization, initialize_part_names
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

# root is modified in place
def set_body_sites(root, params):
    body_sites = []
    for key, v in params["KEYPOINT_MODEL_PAIRS"].items():
        parent = root.find("body", v)
        pos = params["KEYPOINT_INITIAL_OFFSETS"][key]
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
        params["SCALE_FACTOR"],
        params["SCALE_FACTOR"],
    )
    physics = mjcf.Physics.from_mjcf_model(root)
    # Usage of physics: binding = physics.bind(body_sites)
    # TODO: is getting new mjmodel using mjcf_root necessary?
    
    
    part_names = initialize_part_names(mjmodel, mjdata)

    return physics, body_sites, part_names, mjx_model

# TODO mjmodel cant be vectorized so fix (vectorize a lamba func of smth)
@vmap
def prep_data(model, kp_data, stac_keypoint_order, params):
    """Data preparation for kp_data and mjxdata: 1. prepare_data() 2. build_env()
        This function is vmapped so the usage will be to pass in the args with a leading batch dim.
        i.e: kp_data.shape = (batch_size, kp_data.shape)
             stac_keypoint_order.shape = (batch_size, stac_keypoint_order.shape)

    Returns:
        model: mjModel
        mjx_data: MjData with batch dimension; (batch, dim of attr)
        kp_data:
        body_sites: 
    """
    
    kp_data = kp_data[:, :, stac_keypoint_order]
    kp_data = jnp.transpose(kp_data, (0, 2, 1))
    kp_data = jnp.reshape(kp_data, (kp_data.shape[0], -1))
    
    n_frames = kp_data.shape[0]
    
    mjx_model = mjx.device_put(model)
    mjx_data = mjx.make_data(mjx_model)

    return mjx_data, kp_data, n_frames

# pmap fit and transform if you want to use it with multiple gpus
def fit(root, kp_data, params):
    """Calibrate and fit the model to keypoints.
    Performs three rounds of alternating marker and quaternion optimization. Optimal
    results with greater than 200 frames of data in which the subject is moving.
    
    Args:
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
    
    Returns: fitted model props?? find relevant props from stac object

    """    
    
    physics, body_sites, part_names, mjx_model = set_body_sites(root, params)
    # Get and set the offsets of the markers
    offsets = jnp.copy(physics.bind(body_sites).pos[:])
    offsets *= params['SCALE_FACTOR']
    physics.bind(body_sites).pos[:] = offsets
    
    mjx_data = mjx.forward(mjx_model, mjx_data)

    for n_site, p in enumerate(physics.bind(body_sites).pos):
        body_sites[n_site].pos = p
    
    root_optimization(mjx_model, mjx_data, params)
    for n_iter in range(params['N_ITERS']):
        print(f"Calibration iteration: {n_iter + 1}/{params['N_ITERS']}")
        q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, params)
        offset_optimization(mjx_model, mjx_data, offsets, q, params)

    # Optimize the pose for the whole sequence
    print("Final pose optimization")
    q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, params)
    data = package_data(
        mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, params
    )
    return data

def transform(kp_data, offset_path):
    """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

    Args:
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
        offset_path (Text): Path to offset file saved after .fit()
    """
    return

def end_to_end():
    """this function runs fit and transform end to end for conceptualizing purposes
    """
    params = utils.load_params("params/params.yaml")
    model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
    root = mjcf.from_path(params["XML_PATH"])
    mjx_model = mjx.device_put(model)

    n_frames = None

    # Default ordering of mj sites is alphabetical, so we reorder to match
    kp_names = utils.loadmat(params["SKELETON_PATH"])["joint_names"]
    # argsort returns the indices that would sort the array
    stac_keypoint_order = jnp.argsort(params["kp_names"])
    
    # Call vmapped prep data
    mjx_data, kp_data, n_frames = prep_data(kp_data)

    # fit
    data = fit(kp_data, params)
    
    # transform
    # transform()