import mujoco 
from mujoco import mjx 
import jax
from jax import vmap, jit
from jax import numpy as jp
from typing import Any, Dict, Tuple, Union
import yaml

from environment import MjxEnv

class ThousandRatDeathSwarm():

  def __init__(
      self,
      n_envs: int = 1,
      **kwargs,
  ):
      
    param_path = "../../params/params.yaml"
    with open(param_path, "rb") as file:
            params = yaml.safe_load(file)
    xml_path = params["XML_PATH"]
    
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    physics_steps_per_control_step = 5
    kwargs['physics_steps_per_control_step'] = kwargs.get(
        'physics_steps_per_control_step', physics_steps_per_control_step)
    
    def init():
        rat = MjxEnv(mj_model)
        rat.pipeline_init(jp.zeros(rat.model.nq), jp.zeros(rat.model.nv))
        return rat
    # make list of n_envs initialized rats
    self.rats = [init for _ in range(n_envs)]

  def batch_step(self):
    """Runs one timestep of the each environment's dynamics."""
    # for rat in self.rats:
    #     data0 = mjx.make_data(rat.sys)
    #     data = self.pipeline_step(data0, action)

    @vmap
    def f(rat):
        rat.data = self.pipeline_step(rat.sys_data, jp.zeros(rat.model.nu))
        return rat
    
    jit_batch_step = jit(f)
    self.rats = jit_batch_step(self.rats)

swarm = ThousandRatDeathSwarm(10)
swarm.batch_step()
