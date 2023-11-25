import mujoco 
from mujoco import mjx 
import jax
from jax import numpy as jp
from typing import Any, Dict, Tuple, Union
import yaml

class MjxEnv():
  """API for driving an MJX system for training and inference in brax."""

  def __init__(
      self,
      mj_model: mujoco.MjModel,
      physics_steps_per_control_step: int = 1,
  ):
    """Initializes MjxEnv.

    Args:
      mj_model: mujoco.MjModel
      physics_steps_per_control_step: the number of times to step the physics
        pipeline for each environment step
    """
    self.model = mj_model
    self.data = mujoco.MjData(mj_model)
    self.sys = mjx.device_put(mj_model)
    self.sys_data = mjx.device_put(self.data)
    self._physics_steps_per_control_step = physics_steps_per_control_step

  def pipeline_init(
      self, qpos: jax.Array, qvel: jax.Array
  ) -> mjx.Data:
    """Initializes the physics state."""
    self.sys_data = self.sys_data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
    self.sys_data = mjx.forward(self.sys, self.sys_data)
    return self.sys_data

  def pipeline_step(
      self, data: mjx.Data, ctrl: jax.Array
  ) -> mjx.Data:
    """Takes a physics step using the physics pipeline."""
    def f(data, _):
      data = data.replace(ctrl=ctrl)
      return (
          mjx.step(self.sys, data),
          None,
      )
    data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
    return data

  @property
  def dt(self) -> jax.Array:
    """The timestep used for each env step."""
    return self.sys.opt.timestep * self._physics_steps_per_control_step

  @property
  def observation_size(self) -> int:
    rng = jax.random.PRNGKey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def action_size(self) -> int:
    return self.sys.nu

  @property
  def backend(self) -> str:
    return 'mjx'


class Rodent(MjxEnv):

  def __init__(
      self,
      **kwargs,
  ):
      
    param_path = "./params/params.yaml"
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

    super().__init__(mj_model=mj_model, **kwargs)

  def reset(self, rng: jp.ndarray) -> mjx.Data:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    return data

  def step(self, data: mjx.Data, action: jp.ndarray) -> mjx.Data:
    """Runs one timestep of the environment's dynamics."""
    data0 = mjx.Data
    data = self.pipeline_step(data0, action)
    return data
  