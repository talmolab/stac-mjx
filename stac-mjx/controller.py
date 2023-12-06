import jax 
from jax import jit, vmap
import mujoco
from mujoco import mjx

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

# need a function to set mjdata to the starting pos? 
# check what goes into creating the environments to see what needs to be done before doing mjx.make_data(mjxModel),
# then check what needs to be done that is unique to each mjxData

def load_params(param_path: Text) -> Dict:
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params

params = load_params("params/params.yaml")
model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
mjx_model = mjx.device_put(model)

@vmap
def getMjxData():
    mjx_data = mjx.make_data(mjx_model)
    # What else needs to be done before returning?
    return mjx_data