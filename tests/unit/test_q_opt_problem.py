"""Minimal tests for q_opt problem construction and solves."""

import numpy as np
import jax
from jax import numpy as jp
import mujoco

from stac_mjx import utils
from stac_mjx.stac_core import build_q_opt_problem, q_opt

FREE_ROOT_XML = """\
<mujoco>
  <worldbody>
    <body name="root">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1"/>
      <site name="s0" pos="0.1 0.0 0.0"/>
    </body>
  </worldbody>
</mujoco>
"""


FREE_ROOT_WITH_HINGE_XML = """\
<mujoco>
  <worldbody>
    <body name="root">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1"/>
      <body name="child" pos="0.2 0.0 0.0">
        <joint name="j0" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.05"/>
        <site name="s0" pos="0.1 0.0 0.0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


FREE_ROOT_THREE_SITES_XML = """\
<mujoco>
  <worldbody>
    <body name="root">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1"/>
      <site name="s0" pos="0.2 0.0 0.0"/>
      <site name="s1" pos="0.0 0.3 0.0"/>
      <site name="s2" pos="0.0 0.0 0.4"/>
    </body>
  </worldbody>
</mujoco>
"""


HINGE_ONLY_XML = """\
<mujoco>
  <worldbody>
    <body name="hinge">
      <joint name="j0" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size="0.1"/>
      <site name="s0" pos="1.0 0.0 0.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _build_model(xml):
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mjx_model, mjx_data = utils.mjx_load(mj_model)
    site_idxs = jp.arange(mj_model.nsite, dtype=jp.int32)
    return mj_model, mjx_model, mjx_data, site_idxs


def _build_problem(
    mjx_model, mjx_data, site_idxs, joint_mask=None, velocity_smoothness_weight=0.05
):
    if joint_mask is None:
        joint_mask = jp.ones(mjx_model.nq, dtype=bool)
    return build_q_opt_problem(
        n_frames=2,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        joint_mask=joint_mask,
        kp_mask=jp.ones(3, dtype=bool),
        joint_lb=-jp.ones(mjx_model.nq),
        joint_ub=jp.ones(mjx_model.nq),
        site_idxs=site_idxs,
        n_kp_coords=3,
        joint_reg_weights=jp.zeros(mjx_model.nq),
        velocity_smoothness_weight=velocity_smoothness_weight,
    )


def _keypoints_from_qpos(mjx_model, mjx_data, qpos, site_idxs):
    def fk_frame(q):
        data = mjx_data.replace(qpos=q)
        data = utils.kinematics(mjx_model, data)
        data = utils.com_pos(mjx_model, data)
        return utils.get_site_xpos(data, site_idxs).flatten()

    return jax.vmap(fk_frame)(qpos)


def test_q_opt_problem_construction_modes():
    """Check the minimal q_opt problem construction branches."""
    _, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_XML)
    problem = _build_problem(mjx_model, mjx_data, site_idxs)
    assert problem.se3_mode is True
    assert problem.freejoint_root is True
    assert problem.n_frames == 2
    assert problem.n_kp_coords == 3
    assert problem._SE3Var is not None
    assert problem._JointVar is not None
    assert problem._QVar is None

    _, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_WITH_HINGE_XML)
    site_offsets = jp.array([[0.2, 0.3, 0.4]])
    problem = build_q_opt_problem(
        n_frames=2,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        joint_mask=jp.ones(mjx_model.nq, dtype=bool),
        kp_mask=jp.ones(3, dtype=bool),
        joint_lb=-jp.ones(mjx_model.nq),
        joint_ub=jp.ones(mjx_model.nq),
        site_idxs=site_idxs,
        n_kp_coords=3,
        joint_reg_weights=jp.zeros(mjx_model.nq),
        site_offsets=site_offsets,
    )

    assert problem.se3_mode is True
    np.testing.assert_allclose(np.array(problem.site_offsets), np.array(site_offsets))

    _, mjx_model, mjx_data, site_idxs = _build_model(HINGE_ONLY_XML)
    problem = _build_problem(mjx_model, mjx_data, site_idxs)
    assert problem.se3_mode is False
    assert problem.freejoint_root is False
    assert problem._QVar is not None
    assert problem._SE3Var is None
    assert problem._JointVar is None

    _, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_XML)
    joint_mask = jp.ones(mjx_model.nq, dtype=bool).at[0].set(False)
    problem = _build_problem(mjx_model, mjx_data, site_idxs, joint_mask=joint_mask)
    assert problem.se3_mode is False
    assert problem.freejoint_root is True
    assert problem._QVar is not None


def test_q_opt_flat_hinge_recovers_known_qpos():
    """Known hinge angles should reproduce FK-generated keypoints."""
    _, mjx_model, mjx_data, site_idxs = _build_model(HINGE_ONLY_XML)
    q_target = jp.array([[0.35], [-0.4]], dtype=jp.float32)
    keypoints = _keypoints_from_qpos(mjx_model, mjx_data, q_target, site_idxs)
    problem = _build_problem(
        mjx_model, mjx_data, site_idxs, velocity_smoothness_weight=0.0
    )

    q_out = q_opt(
        problem,
        jp.zeros_like(q_target),
        keypoints,
        n_solver_max_iters=30,
        initial_step_damping=0.1,
    )

    np.testing.assert_allclose(np.array(q_out), np.array(q_target), atol=1e-5)


def test_q_opt_se3_root_recovers_known_qpos():
    """A free-root solve should recover translation and orientation."""
    _, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_THREE_SITES_XML)
    q_target = jp.array(
        [
            [0.1, -0.2, 0.3, 0.9921977, 0.0, 0.0, 0.12467473],
            [-0.15, 0.05, 0.2, 0.98472655, 0.0, 0.0, -0.17410813],
        ],
        dtype=jp.float32,
    )
    keypoints = _keypoints_from_qpos(mjx_model, mjx_data, q_target, site_idxs)
    problem = build_q_opt_problem(
        n_frames=2,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        joint_mask=jp.ones(mjx_model.nq, dtype=bool),
        kp_mask=jp.ones(9, dtype=bool),
        joint_lb=-10.0 * jp.ones(mjx_model.nq),
        joint_ub=10.0 * jp.ones(mjx_model.nq),
        site_idxs=site_idxs,
        n_kp_coords=9,
        joint_reg_weights=jp.zeros(mjx_model.nq),
    )
    q_init = jp.tile(jp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (2, 1))

    q_out = q_opt(
        problem,
        q_init,
        keypoints,
        n_solver_max_iters=50,
        initial_step_damping=0.1,
    )

    np.testing.assert_allclose(np.array(q_out), np.array(q_target), atol=1e-5)
