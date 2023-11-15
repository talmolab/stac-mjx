import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put
import numpy as np 
import yaml
from typing import List, Dict, Text


def load_params(param_path: Text) -> Dict:
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params

params = load_params("params/params.yaml")
model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
mjx_model = mjx.device_put(model)

# minimal example code--this is supposed to work
# @jax.vmap
# def batched_step(vel):
#     mjx_data = mjx.make_data(mjx_model)
#     qvel = mjx_data.qvel.at[0].set(vel)
#     mjx_data = mjx_data.replace(qvel=qvel)
#     pos = mjx.step(mjx_model, mjx_data).qpos[0]
#     return pos

# vel = jax.numpy.linspace(0.0, 0.2, 5)
# pos = jax.jit(batched_step)(vel)
def serial_step(vel):
    data = mujoco.MjData(model)
    print(data.xpos)
    data.qvel[0] = 0
    # qvel[0] = vel
    # data = data.replace(qvel=qvel)
    mujoco.mj_step(model, data)
    
    return data.xpos

pos = serial_step(.1)

print(pos)