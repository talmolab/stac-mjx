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
# def batched_step(vel, mjx_model):
#     mjx_data = mjx.make_data(mjx_model)
#     qvel = mjx_data.qvel.at[0].set(vel)
#     mjx_data = mjx_data.replace(qvel=qvel)
#     pos = mjx.step(mjx_model, mjx_data).qpos[0]
#     return pos

def serial_step(vel):
    data = mujoco.MjData(model)
    print(data.qpos)
    data.qvel[0] = 0
    # qvel[0] = vel
    # data = data.replace(qvel=qvel)
    mujoco.mj_step(model, data)
    
    return data.qpos

def serial_step_mjx(vel):
    mjx_data = mjx.make_data(mjx_model)    
    print(mjx_data.qpos)
    qvel = mjx_data.qvel.at[0].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    mjx_data = mjx.step(mjx_model, mjx_data)
    # mjx.forward(model, mjx_data)
    
    return mjx_data.qpos

# jit compile and simulate one step 5 times parallely!
# vel = jax.numpy.linspace(0.0, 0.2, 5)
# pos = jax.jit(batched_step)(vel)
#12:56
import time
print("Compilation done: " + str(time.time()))
def step_function(vel, mjx_model):
    mjx_data = mjx.make_data(mjx_model)
    qvel = mjx_data.qvel.at[0].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    mjx.step(mjx_model, mjx_data)

# Wrap the step function with jax.vmap for vectorization over vel
batched_step = jax.vmap(step_function, in_axes=(0, None))
start_time = prev_time = time.time()
vel = jnp.linspace(0.0, 0.2, 2048)
for i in range(100):    
    jax.jit(lambda v: batched_step(v, mjx_model), backend="gpu")(vel)
    print(f"batch {i}: {time.time() - prev_time}")
    prev_time = time.time()

end_time = time.time()
print(f"Time to complete {100 * len(vel)} steps: {end_time - start_time}")
