"""Check enabled and disabled MuJoCo joint limits during IK."""

import jax
import mujoco
import numpy as np
import pytest
from jax import numpy as jp

from stac_mjx import stac_core
from stac_mjx.stac import _align_joint_dims
from tests.unit.test_joint_rotations import _build_problem, _markers


# shortcut: use float64 until JAXLS fixes float32 stalls at active joint limits.
@pytest.mark.parametrize("turns", [-2, 2])
@jax.enable_x64(True)
def test_disabled_limits_and_positive_lower_bounds(turns):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody>
      <body><joint type="hinge" limited="true" range="0.5 1"/>
        <geom size="0.1"/><site pos="1 0 0"/></body>
      <body><joint type="hinge" limited="false" range="-0.1 0.1"/>
        <geom size="0.1"/><site pos="1 0 0"/></body>
      <body><joint type="slide" limited="false" range="0.1 0.2"/>
        <geom size="0.1"/><site pos="1 0 0"/></body>
      <body><joint type="ball" limited="false" range="0 0.2"/>
        <geom size="0.1"/><site pos="1 0 0"/><site pos="0 1 0"/></body>
    </worldbody></mujoco>""")
    hinge_angle = turns * 2 * np.pi + 0.6
    target = np.array([[0.0, hinge_angle, -0.3, np.cos(0.5), np.sin(0.5), 0, 0]])
    initial = model.qpos0[None].copy()
    initial[0, 1] = hinge_angle - 0.1
    problem = _build_problem(model, 1)
    solved = np.array(
        stac_core.q_opt(
            problem,
            jp.array(initial),
            _markers(model, target),
            n_solver_max_iters=100,
            initial_step_damping=0.1,
        ).qpos
    )
    np.testing.assert_allclose(solved[0, :3], [0.5, hinge_angle, -0.3], atol=1e-4)
    np.testing.assert_allclose(solved[0, 3:], target[0, 3:], atol=1e-4)
    lb, ub, _ = _align_joint_dims(
        model.jnt_type, model.jnt_range, ["j"] * 4, model.jnt_limited
    )
    assert np.isneginf(lb[1]) and np.isposinf(ub[1])
    assert np.isneginf(lb[2]) and np.isposinf(ub[2])
