import utils
import mujoco
import os
import pickle
from scipy.io import savemat 
from dm_control import mjcf
import numpy as np
import jax
from jax import numpy as jnp
import time
from controller import *

start_time = time.time()

# If your machine is low on ram:
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.6'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

def save(fit_data, save_path):
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

# relative pathing no working in notebook rn
utils.init_params("././params/params.yaml")
ratpath = "././models/rodent_stac.xml"
rat23path = "././models/rat23.mat"
model = mujoco.MjModel.from_xml_path(ratpath)
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
model.opt.iterations = 1
model.opt.ls_iterations = 1

# Need to download this data file and provide the path
data_path = "/home/charles/Desktop/save_data_AVG.mat"

# data_path = "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/DANNCE/predict03/save_data_AVG.mat" 

root = mjcf.from_path(ratpath)

# Default ordering of mj sites is alphabetical, so we reorder to match
kp_names = utils.loadmat(rat23path)["joint_names"]
utils.params["kp_names"] = kp_names

# argsort returns the indices that would sort the array
stac_keypoint_order = np.argsort(kp_names)
# Load kp_data
kp_data = utils.loadmat(data_path)["pred"][:] / 1000

kp_data = prep_kp_data(kp_data, stac_keypoint_order)

# print(f"total envs: {n_envs}")
# fit_data = test_opt(root, kp_data)
# Single clip optimization for first 500 frames
def test_single_clip_fit(root, kp_data):
    # returns fit_data
    fit_data = single_clip_opt(root, kp_data)
    offset_path = "offset_sing_clip2.p"
    print(f"saving data to {offset_path}")
    save(fit_data, offset_path)

def test_transform(offset_path, root, kp_data):
    print("Running transform()")
    with open(offset_path, "rb") as file:
        data = pickle.load(file)
    offsets = data["offsets"] 
    kp_data, n_envs = chunk_kp_data(kp_data)
    transform_data = transform(root, kp_data[500:], offsets)
    transform_path = "transform1.p"
    print(f"saving data to {transform_path}")
    save(transform_data, transform_path)


def test_m_phase():
    post_pose_opt_path = "pose_opt_qs.p"
    with open(post_pose_opt_path, "rb") as file:
        in_dict = pickle.load(file)

    mjx_model = in_dict["mjx_model"]
    mjx_data = in_dict["mjx_data"]
    kp_data = in_dict["kp_data"]
    q = in_dict["q"]
    physics = in_dict["physics"]
    x = in_dict["x"]
    walker_body_sites = in_dict["walker_body_sites"]
    utils.params["site_index_map"] = in_dict["site_index_map"]

    @jax.vmap
    def get_offsets(mjx_model):
        offsets = jnp.copy(stac_base.get_site_pos(mjx_model))
        offsets *= utils.params['SCALE_FACTOR']
        return offsets
    offsets = get_offsets(mjx_model)

    mjx_model, mjx_data = vmap(offset_optimization)(
        mjx_model, 
        mjx_data, 
        kp_data, 
        offsets, 
        q
        )
    mjx_model, mjx_data = vmap(offset_optimization)(
        mjx_model, 
        mjx_data, 
        kp_data, 
        offsets, 
        q
        )
    mjx_model, mjx_data = vmap(offset_optimization)(
        mjx_model, 
        mjx_data, 
        kp_data, 
        offsets, 
        q
        )

    fit_data = package_data(
            mjx_model, physics, q, x, walker_body_sites, kp_data
        )
    offset_path = "m_phase_test.p"
    print(f"saving data to {offset_path}")
    save(fit_data, offset_path)


# test_transform("offset_sing_clip1.p", root, kp_data)
# test_single_clip_fit(root, kp_data[500:])
# test_m_phase()

print(f"Job complete in {time.time()-start_time}")

