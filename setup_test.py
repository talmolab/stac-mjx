import mujoco
from mujoco import mjx
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put
import numpy as np 

print("mujoco-mjx and jax successfully imported")

size = 3000

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)

print(x)

XML=r"""
 <mujoco>
   <worldbody>
     <body>
       <freejoint/>
       <geom size=".15" mass="1" type="sphere"/>
     </body>
   </worldbody>
 </mujoco>
 """

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.device_put(model)

@jax.vmap
def batched_step(vel):
    mjx_data = mjx.make_data(mjx_model)
    qvel = mjx_data.qvel.at[0].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    pos = mjx.step(mjx_model, mjx_data).qpos[0]
    return pos

vel = jax.numpy.arange(0.0, 1.0, 0.01)
pos = jax.jit(batched_step)(vel)
print(pos)