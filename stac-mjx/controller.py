import jax 
from jax import jit, vmap
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from typing import Text, Dict
import utils

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

@vmap
def prep_data(kp_data):
    """data prep

    Returns:
        mjx_data: MjData with batch dimension; (batch, dim of attr)
    """
    kp_data = self._prepare_data(kp_data)
    n_frames = kp_data.shape[0]
    mjx_model, mjx_data = build_env(kp_data, self._properties)
    part_names = initialize_part_names(mjx_model, mjx_data)

    # Get and set the offsets of the markers
    offsets = jnp.copy(env.physics.bind(env.task._walker.body_sites).pos[:])
    offsets *= self.SCALE_FACTOR
    env.physics.bind(env.task._walker.body_sites).pos[:] = offsets
    
    mjx_data = mjx.forward(mjx_model, mjx_data)

    for n_site, p in enumerate(env.physics.bind(env.task._walker.body_sites).pos):
        env.task._walker.body_sites[n_site].pos = p
    # call a reset function
    return mjx_data

@vmap
def vmap_kp_data(batch_arr):
    """returns a batch of kpdatas

    Args:
        batch_arr (jnp.array): an arbitrary(?) array of length x, where x is the batchsize u want
    """
    return kp_data


# pmap fit and transform if you want to use it with multiple gpus
def fit(kp_data):
    """Calibrate and fit the model to keypoints.
    Performs three rounds of alternating marker and quaternion optimization. Optimal
    results with greater than 200 frames of data in which the subject is moving.
    
    Args:
        kp_data (jnp.ndarray): Keypoint data in meters (batch_size, n_frames, 3, n_keypoints).
            Keypoint order must match the order in the skeleton file.
    
    Returns: fitted model props?? find relevant props from stac object

    """
    mjx_data, y, z = prep_data(kp_data)
    
    root_optimization(mjx_model, mjx_data, self._properties)
    for n_iter in range(self.N_ITERS):
        print(f"Calibration iteration: {n_iter + 1}/{self.N_ITERS}")
        q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)
        offset_optimization(mjx_model, mjx_data, offsets, q, self._properties)

    # Optimize the pose for the whole sequence
    print("Final pose optimization")
    q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)
    self.data = package_data(
        mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, self._properties
    )
    return

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
    mjx_model = mjx.device_put(model)

    data = None
    n_frames = None

    # Default ordering of mj sites is alphabetical, so we reorder to match
    kp_names = utils.loadmat(params["SKELETON_PATH"])["joint_names"]
    # argsort returns the indices that would sort the array
    stac_keypoint_order = jnp.argsort(params["kp_names"])
