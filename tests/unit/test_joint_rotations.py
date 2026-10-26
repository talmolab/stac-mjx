"""Compare joint-aware trajectory operations and IK with native MuJoCo."""

import jax
import mujoco
import numpy as np
import pytest
from jax import numpy as jp

from stac_mjx import stac_core, utils
from stac_mjx.joints import JointLayout
from stac_mjx.stac import _align_joint_dims


def _model(joint_types):
    bodies = "".join(
        f'<body pos="{i} 0 0"><joint type="{kind}"/>'
        '<geom type="sphere" size="0.1"/></body>'
        for i, kind in enumerate(joint_types)
    )
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    )


@pytest.mark.parametrize(
    "joint_types",
    [
        ("ball",),
        ("free", "hinge", "ball", "slide", "ball", "free", "hinge"),
    ],
)
def test_velocities_and_interpolation_match_mujoco(joint_types):
    model = _model(joint_types)
    layout = JointLayout(model.jnt_type)
    rng = np.random.default_rng(17)
    start = model.qpos0.copy()
    mujoco.mj_integratePos(model, start, rng.normal(size=model.nv), 0.4)
    end = start.copy()
    mujoco.mj_integratePos(model, end, rng.normal(size=model.nv), 0.2)
    end[layout.quaternion_qpos] *= -1
    keyframes = jp.array([start, end])

    trajectory = utils.interpolate_qpos_from_keyframes(
        keyframes, jp.array([0, 4]), 5, False, joint_types=model.jnt_type
    )
    displacement = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, displacement, 1.0, start, end)
    for i, actual in enumerate(np.asarray(trajectory)):
        expected = start.copy()
        mujoco.mj_integratePos(model, expected, displacement, i / 4)
        error = np.empty(model.nv)
        mujoco.mj_differentiatePos(
            model, error, 1.0, expected, actual.astype(np.float64)
        )
        np.testing.assert_allclose(error, 0, atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(trajectory[:, layout.quaternion_qpos], axis=-1), 1, atol=1e-6
    )

    # Alternating quaternion signs must not create angular-velocity spikes.
    trajectory = np.array(trajectory, dtype=np.float64)
    trajectory[1::2, layout.quaternion_qpos] *= -1
    expected_velocity = np.empty((5, model.nv))
    dt = 0.01
    for i in range(4):
        mujoco.mj_differentiatePos(
            model, expected_velocity[i], dt, trajectory[i], trajectory[i + 1]
        )
    expected_velocity[-1] = expected_velocity[-2]
    velocity = jax.jit(
        lambda q: utils.compute_velocity_from_kinematics(
            q, dt, joint_types=model.jnt_type
        )
    )(jp.asarray(trajectory))
    assert velocity.shape == (5, model.nv)
    np.testing.assert_allclose(velocity, expected_velocity, atol=2e-5, rtol=2e-5)

    single = utils.compute_velocity_from_kinematics(
        trajectory[:1], dt, joint_types=model.jnt_type
    )
    np.testing.assert_array_equal(single, np.zeros((1, model.nv)))
    held = utils.interpolate_qpos_from_keyframes(
        keyframes[:1], jp.array([0]), 3, False, joint_types=model.jnt_type
    )
    np.testing.assert_allclose(held, np.tile(start, (3, 1)), atol=1e-6)


def _ball_model(free_root, limited=True):
    sites = '<site pos="0.3 0 0"/><site pos="0 0.3 0"/><site pos="0 0 0.3"/>'
    child = (
        f'<body><joint type="ball" limited="{str(limited).lower()}" range="0 1.7"/>'
        f'<geom type="sphere" size="0.1"/>{sites}</body>'
    )
    if free_root:
        child = (
            f'<body><freejoint/><geom type="sphere" size="0.1"/>{sites}{child}</body>'
        )
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><worldbody>{child}</worldbody></mujoco>'
    )


def _build_problem(model, frames, mask=None, smoothness=0.0):
    mjx_model, data = utils.mjx_load(model)
    lb, ub, _ = _align_joint_dims(
        model.jnt_type, model.jnt_range, ["joint"] * model.njnt, model.jnt_limited
    )
    problem = stac_core.build_q_opt_problem(
        frames,
        mjx_model,
        data,
        jp.ones(model.nq, dtype=bool) if mask is None else mask,
        jp.ones(model.nsite * 3, dtype=bool),
        lb,
        ub,
        jp.arange(model.nsite),
        model.nsite * 3,
        jp.zeros(model.nq),
        velocity_smoothness_weight=smoothness,
    )
    return problem


def _markers(model, qpos):
    data = mujoco.MjData(model)
    markers = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        markers.append(data.site_xpos.flatten().copy())
    return jp.array(markers)


@pytest.mark.parametrize("free_root", [False, True])
def test_ball_ik_respects_angular_limit(free_root):
    model = _ball_model(free_root)
    problem = _build_problem(model, 3)
    ball_start = int(model.jnt_qposadr[-1])
    angles = np.array([-0.3, 0.3, 2.0])
    target = np.tile(model.qpos0, (3, 1))
    target[:, ball_start] = np.cos(angles / 2)
    target[:, ball_start + 1] = np.sin(angles / 2)
    initial = np.tile(model.qpos0, (3, 1))
    initial[1, ball_start : ball_start + 4] *= -1
    solved = np.array(
        stac_core.q_opt(
            problem,
            jp.array(initial),
            _markers(model, target),
            n_solver_max_iters=100,
            initial_step_damping=0.1,
        ).qpos
    )
    quat = solved[:, ball_start : ball_start + 4]
    np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-6)
    angle = 2 * np.arctan2(np.linalg.norm(quat[:, 1:], axis=-1), np.abs(quat[:, 0]))
    np.testing.assert_allclose(angle[:2], 0.3, atol=3e-4)
    assert angle[2] <= 1.7 + 2e-4
    assert angle[2] > 1.69
    np.testing.assert_allclose(
        _markers(model, solved[:2]), _markers(model, target[:2]), atol=1e-4
    )


def test_smoothing_is_invariant_to_ball_quaternion_sign():
    model = _ball_model(False)
    problem = _build_problem(model, 2, smoothness=0.1)
    target = np.array([[np.cos(0.2), np.sin(0.2), 0, 0]] * 2)
    initial = target.copy()
    initial[1] *= -1
    solved = np.array(
        stac_core.q_opt(
            problem, jp.array(initial), _markers(model, target), n_solver_max_iters=30
        ).qpos
    )
    np.testing.assert_allclose(
        _markers(model, solved), _markers(model, target), atol=1e-6
    )
    np.testing.assert_allclose(
        JointLayout(model.jnt_type).difference(
            jp.array(solved[0]), jp.array(solved[1])
        ),
        0,
        atol=1e-6,
    )
