import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put
import numpy as np 
import yaml
from typing import List, Dict, Text, Any, Sequence, Union, Optional
import time
import functools
import copy
from flax import struct
import logging
import os
from jax.tree_util import tree_map

num_gpus = jax.local_device_count()

@struct.dataclass
class Base:
  """Base functionality extending all brax types.

  These methods allow for brax types to be operated like arrays/matrices.
  """

  def __add__(self, o: Any) -> Any:
    return tree_map(lambda x, y: x + y, self, o)

  def __sub__(self, o: Any) -> Any:
    return tree_map(lambda x, y: x - y, self, o)

  def __mul__(self, o: Any) -> Any:
    return tree_map(lambda x: x * o, self)

  def __neg__(self) -> Any:
    return tree_map(lambda x: -x, self)

  def __truediv__(self, o: Any) -> Any:
    return tree_map(lambda x: x / o, self)

  def reshape(self, shape: Sequence[int]) -> Any:
    return tree_map(lambda x: x.reshape(shape), self)

  def select(self, o: Any, cond: jax.Array) -> Any:
    return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

  def slice(self, beg: int, end: int) -> Any:
    return tree_map(lambda x: x[beg:end], self)

  def take(self, i, axis=0) -> Any:
    return tree_map(lambda x: jnp.take(x, i, axis=axis, mode='wrap'), self)

  def concatenate(self, *others: Any, axis: int = 0) -> Any:
    return tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

  def index_set(
      self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
  ) -> Any:
    return tree_map(lambda x, y: x.at[idx].set(y), self, o)

  def index_sum(
      self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
  ) -> Any:
    return tree_map(lambda x, y: x.at[idx].add(y), self, o)

  def vmap(self, in_axes=0, out_axes=0):
    """Returns an object that vmaps each follow-on instance method call."""

    # TODO: i think this is kinda handy, but maybe too clever?

    outer_self = self

    class VmapField:
      """Returns instance method calls as vmapped."""

      def __init__(self, in_axes, out_axes):
        self.in_axes = [in_axes]
        self.out_axes = [out_axes]

      def vmap(self, in_axes=0, out_axes=0):
        self.in_axes.append(in_axes)
        self.out_axes.append(out_axes)
        return self

      def __getattr__(self, attr):
        fun = getattr(outer_self.__class__, attr)
        # load the stack from the bottom up
        vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
        for in_axes, out_axes in vmap_order:
          fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
        fun = functools.partial(fun, outer_self)
        return fun

    return VmapField(in_axes, out_axes)

  def tree_replace(
      self, params: Dict[str, Optional[jax.typing.ArrayLike]]
  ) -> 'Base':
    """Creates a new object with parameters set.

    Args:
      params: a dictionary of key value pairs to replace

    Returns:
      data clas with new values

    Example:
      If a system has 3 links, the following code replaces the mass
      of each link in the System:
      >>> sys = sys.tree_replace(
      >>>     {'link.inertia.mass', jp.array([1.0, 1.2, 1.3])})
    """
    new = self
    for k, v in params.items():
      new = _tree_replace(new, k.split('.'), v)
    return new

  @property
  def T(self):  # pylint:disable=invalid-name
    return tree_map(lambda x: x.T, self)

def _tree_replace(
    base: Base,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Base:
  """Sets attributes in a struct.dataclass with values."""
  if not attr:
    return base

  # special case for List attribute
  if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
    lst = copy.deepcopy(getattr(base, attr[0]))

    for i, g in enumerate(lst):
      if not hasattr(g, attr[1]):
        continue
      v = val if not hasattr(val, '__iter__') else val[i]
      lst[i] = _tree_replace(g, attr[1:], v)

    return base.replace(**{attr[0]: lst})

  if len(attr) == 1:
    return base.replace(**{attr[0]: val})

  return base.replace(
      **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
  )

@struct.dataclass
class State(Base):
  """A minimal state class (only containing mjx.Data).

  Args:
    pipeline_state: the physics state, mjx.Data
  """

  data: mjx.Data

def load_params(param_path: Text) -> Dict:
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params

params = load_params("params/params.yaml")
model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
mjdata = mujoco.MjData(model)
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.iterations = 1
model.opt.ls_iterations = 1

mjx_model = mjx.device_put(model)

"""takes state class and ctrl (action) vector, returns next step's state"""
def single_step(state, ctrl):
    data0 = state.data
    data0 = data0.replace(ctrl=ctrl)
    data = mjx.step(mjx_model, data0)
    state = state.replace(data=data)
    return state

total_envs = 2048
n_envs_small = 1
key = random.PRNGKey(0)
small_ctrl = random.uniform(key, shape=(n_envs_small, mjx_model.nu))
large_ctrl = random.uniform(key, shape=(total_envs, mjx_model.nu))
per_gpu = random.uniform(key, shape=(int(total_envs / num_gpus), 1))
large_ctrl_split = random.uniform(key, shape=(num_gpus, int(total_envs / num_gpus), mjx_model.nu)) # large_ctrl.reshape(num_gpus, -1)

def reset(val: int) -> State:
    """Resets the environment to an initial state."""
    data = mjx.make_data(mjx_model)
    # data = data.replace(qvel=jnp.zer(mjx_model.nv, x))
    data = mjx.forward(mjx_model, data)
    return State(data)

reset_fn = jax.jit(jax.vmap(reset))
single_batch_step = jax.vmap(single_step)
# returns the state object with a batch axis for each attribute in data (batch_size=n_envs_large)
# env_state = reset_fn(per_gpu)
# print(env_state.data.qpos.shape)

print("Running single step scan")

jit_single_batch_step = jax.jit(single_batch_step)
start_time = time.time()
steps = 100

def take_steps(large_ctrl):
    env_state = reset_fn(per_gpu)
    def f(state ,_):
        return (jit_single_batch_step(state, large_ctrl), None)
    env_state, _ = jax.lax.scan(f, env_state, (), length=steps)

    # d = mujoco.MjData(model)
    # mjx.device_get_into(d, env_state.data)
    return 

parallel_take_steps = jax.pmap(take_steps)
parallel_take_steps(large_ctrl_split)
print(f"{steps * total_envs} steps completed in {time.time()-start_time} seconds")

print("Running single step for loop")

start_time = time.time()

# jit_single_batch_step(env_state, large_ctrl)
# prev = time.time()
# print(f"initial execution time: {prev - start_time}")
def loop_steps(large_ctrl):
    env_state = reset_fn(per_gpu)
    for _ in range(steps):
        env_state = jit_single_batch_step(env_state, large_ctrl)
        # print(f"{time.time()-prev}")
        # prev = time.time()
    # d = mujoco.MjData(model)
    # mjx.device_get_into(d, env_state.data)
    return 

parallel_loop_steps = jax.pmap(loop_steps)
parallel_loop_steps(large_ctrl_split)
print(f"{steps * total_envs} steps completed in {time.time()-start_time} seconds")
