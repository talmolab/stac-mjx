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
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

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
fit_data = test_opt(root, fit_kp_data)
save(fit_data, offset_path)

print(f"Job complete in {time.time()-start_time}")

