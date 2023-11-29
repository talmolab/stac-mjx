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
import time

start_time = time.time()

def load_params(param_path: Text) -> Dict:
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params

params = load_params("params/params.yaml")
model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
mjx_model = mjx.device_put(model)

# minimal example code--this is supposed to work
@jax.vmap
def single_batch_step(ctrl, mjx_model):
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(ctrl=ctrl)
    qpos = mjx.step(mjx_model, mjx_data).qpos
    return qpos

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

# jit compile and simulate one step 5 times parallely
# vel = jax.numpy.linspace(0.0, 0.2, 5)
# pos = jax.jit(batched_step)(vel)

def take_steps(ctrl, steps, mjx_model):
    # ctrl = network(obs)
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(ctrl=ctrl)
    def f(data, _):
      return (
          mjx.step(mjx_model, data),
          None,
      )
      
    mjx_data, _ = jax.lax.scan(f, mjx_data, (), steps)
    return mjx_data.qpos

steps = 1
n_envs_small = 1
n_envs_large = 512
batched_steps = vmap(lambda ctrl: take_steps(ctrl, steps, mjx_model), in_axes=0)

key = random.PRNGKey(0)
small_ctrl = random.uniform(key, shape=(n_envs_small, mjx_model.nu))

jit_batch_step = jit(batched_steps)

batch_end_data = jit_batch_step(small_ctrl)
initial_time = time.time() - start_time
print(f"compilation (first execution) done in: {initial_time}")
two = time.time()
print("starting second run")
large_ctrl = random.uniform(key, shape=(n_envs_large, mjx_model.nu))

batch_end_data = jit_batch_step(large_ctrl)

end_time = time.time()
print(f"second run duration: {end_time - two}")
print(f"Time to complete {steps * n_envs_large} steps: {end_time - start_time}")
