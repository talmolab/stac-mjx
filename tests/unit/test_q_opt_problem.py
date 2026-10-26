"""Minimal tests for q_opt problem construction and solves."""

import jax
import jaxlie
import jaxls
import mujoco
import numpy as np
import pytest
from jax import numpy as jp

from stac_mjx import utils
from stac_mjx.stac_core import QOptResult, build_q_opt_problem, q_opt

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


def test_q_opt_problem_construction():
    """One builder preserves the free-root block and handles scalar/masked roots."""
    _, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_XML)
    problem = _build_problem(mjx_model, mjx_data, site_idxs)
    assert problem.n_frames == 2
    assert problem.n_kp_coords == 3
    assert problem._RootPoseVar.tangent_dim == 6
    assert problem._JointPoseVar.tangent_dim == mjx_model.nv - 6

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

    np.testing.assert_allclose(np.array(problem.site_offsets), np.array(site_offsets))

    _, mjx_model, mjx_data, site_idxs = _build_model(HINGE_ONLY_XML)
    problem = _build_problem(mjx_model, mjx_data, site_idxs)
    assert problem._RootPoseVar is None
    assert problem._JointPoseVar.tangent_dim == mjx_model.nv

    _, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_XML)
    joint_mask = jp.ones(mjx_model.nq, dtype=bool).at[0].set(False)
    problem = _build_problem(mjx_model, mjx_data, site_idxs, joint_mask=joint_mask)
    assert problem._RootPoseVar is None
    np.testing.assert_array_equal(problem.joint_mask, joint_mask)


def test_q_opt_hinge_recovers_known_qpos():
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

    assert isinstance(q_out, QOptResult)
    assert isinstance(q_out.summary, jaxls.SolveSummary)
    np.testing.assert_allclose(np.array(q_out.qpos), np.array(q_target), atol=1e-5)


def test_q_opt_free_root_recovers_known_qpos():
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

    assert isinstance(q_out, QOptResult)
    assert isinstance(q_out.summary, jaxls.SolveSummary)
    np.testing.assert_allclose(np.array(q_out.qpos), np.array(q_target), atol=1e-5)


@pytest.mark.parametrize("freeze_root_z", [False, True])
def test_root_smoothing_and_posture_costs(freeze_root_z):
    """Preserve the old root cost policy through the shared optimizer."""
    model, mjx_model, mjx_data, site_idxs = _build_model(FREE_ROOT_WITH_HINGE_XML)
    angles = np.array([0.2, 1.2])
    poses = np.zeros((2, model.nq))
    poses[:, :2] = [[0.1, 0.2], [0.3, -0.1]]
    poses[:, 3] = np.cos(angles / 2)
    poses[:, 6] = np.sin(angles / 2)
    poses[:, 7] = [0.2, -0.3]
    mask = jp.ones(model.nq, dtype=bool).at[2].set(not freeze_root_z)
    smoothness, posture_weight = 0.7, 0.4
    problem = build_q_opt_problem(
        2,
        mjx_model,
        mjx_data,
        mask,
        jp.zeros(3, dtype=bool),
        jp.full(model.nq, -jp.inf),
        jp.full(model.nq, jp.inf),
        site_idxs,
        3,
        jp.full(model.nq, posture_weight),
        velocity_smoothness_weight=smoothness,
    )
    offset = 0 if freeze_root_z else 7
    entries = [
        problem._JointPoseVar(jp.arange(2)).with_value(jp.array(poses[:, offset:])),
        problem._ObservedKeypointsVar(jp.arange(2)).with_value(jp.zeros((2, 3))),
    ]
    if not freeze_root_z:
        roots = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jp.array(poses[:, 3:7])), jp.array(poses[:, :3])
        )
        entries.append(problem._RootPoseVar(jp.arange(2)).with_value(roots))
    values = jaxls.VarValues.make(entries)
    residual = problem.analyzed.compute_residual_vector(values)

    displacement = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, displacement, 1.0, poses[0], poses[1])
    expected_posture = posture_weight * np.sum(poses[:, 7:] ** 2)
    if freeze_root_z:
        expected_posture += posture_weight * (
            np.sum(poses[:, :2] ** 2) + np.sum(angles**2)
        )
    else:
        roots = [
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jp.array(pose[3:7])), jp.array(pose[:3])
            )
            for pose in poses
        ]
        displacement[:6] = (roots[0].inverse() @ roots[1]).log()
    expected_cost = smoothness**2 * np.sum(displacement**2) + expected_posture
    np.testing.assert_allclose(jp.sum(residual**2), expected_cost, rtol=1e-6)


def test_free_root_keeps_coupled_se3_update_with_ball_joint():
    xml = FREE_ROOT_WITH_HINGE_XML.replace(
        'type="hinge" axis="0 0 1" range="-1 1"', 'type="ball"'
    )
    _, mjx_model, mjx_data, site_idxs = _build_model(xml)
    problem = _build_problem(mjx_model, mjx_data, site_idxs)
    # Keep separate six- and three-dimensional blocks for CG preconditioning.
    assert problem._RootPoseVar.tangent_dim == 6
    assert problem._JointPoseVar.tangent_dim == 3

    root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_z_radians(jp.pi / 2), jp.array([0.1, 0.2, 0.3])
    )
    delta = jp.array([1.0, 0.0, 0.0, 0.0, 0.0, jp.pi / 2])
    updated = problem._RootPoseVar.retract_fn(root, delta)
    # A simultaneous local translation/rotation follows an arc in world space.
    np.testing.assert_allclose(
        updated.translation(), [0.1 - 2 / np.pi, 0.2 + 2 / np.pi, 0.3], atol=1e-6
    )
    np.testing.assert_allclose(
        updated.rotation().as_matrix(), np.diag([-1.0, -1.0, 1.0]), atol=1e-6
    )

    ball_delta = jp.array([0.2, -0.3, 0.1])
    ball = problem._JointPoseVar.retract_fn(jp.array([1.0, 0.0, 0.0, 0.0]), ball_delta)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_quatIntegrate(expected, np.array(ball_delta, dtype=np.float64), 1.0)
    np.testing.assert_allclose(ball, expected, atol=1e-6)
