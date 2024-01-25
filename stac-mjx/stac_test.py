# %%
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

start_time = time.time()

# %%
# If youre machine is low on ram:
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.6'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# %%
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

# %%
from controller import *

# %%
# relative pathing no working in notebook rn
utils.init_params("/home/charles/github/stac-mjx/params/params.yaml")
ratpath = "/home/charles/github/stac-mjx/models/rodent.xml"
rat23path = "/home/charles/github/stac-mjx/models/rat23.mat"
model = mujoco.MjModel.from_xml_path(ratpath)
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
model.opt.iterations = 1
model.opt.ls_iterations = 1

# Need to download this data file and provide the path
data_path = "/home/charles/Desktop/save_data_AVG.mat"
offset_path = "offset.p"

root = mjcf.from_path(ratpath)

# Default ordering of mj sites is alphabetical, so we reorder to match
kp_names = utils.loadmat(rat23path)["joint_names"]
utils.params["kp_names"] = kp_names

# argsort returns the indices that would sort the array
stac_keypoint_order = np.argsort(kp_names)
# Load kp_data
kp_data = utils.loadmat(data_path)["pred"][:] / 1000


# %%
# kp_data
# TODO: store kp_data used in fit in another variable (small slice of kpdata)
kp_data = prep_kp_data(kp_data, stac_keypoint_order)
# chunk it to pass int vmapped functions
kp_data, n_envs = chunk_kp_data(kp_data)
# %%
fit_kp_data = kp_data[:100]
fit_kp_data.shape

# %%
# fit
# fit_data = test_opt(root, fit_kp_data)
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

mjx_model, mjx_data = offset_optimization(
    mjx_model, 
    mjx_data, 
    kp_data, 
    offsets, 
    q
    )

fit_data = package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data
    )
save(fit_data, offset_path)

print(f"Job complete in {time.time()-start_time}")

