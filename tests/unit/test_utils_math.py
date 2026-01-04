import os
import numpy as np
from jax import numpy as jp

from stac_mjx import utils


def test_quat_mul_identity():
    q = jp.array([0.5, 0.5, 0.5, 0.5])
    ident = jp.array([1.0, 0.0, 0.0, 0.0])
    out = utils.quat_mul(ident, q)
    assert np.allclose(np.array(out), np.array(q))


def test_quat_conj_and_diff():
    q = jp.array([0.5, 0.5, 0.5, 0.5])
    conj = utils.quat_conj(q)
    assert np.allclose(np.array(conj), np.array([0.5, -0.5, -0.5, -0.5]))
    diff = utils.quat_diff(q, q)
    assert np.allclose(np.array(diff), np.array([1.0, 0.0, 0.0, 0.0]))


def test_quat_to_axisangle():
    angle = np.pi / 2
    q = jp.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    axis_angle = utils.quat_to_axisangle(q)
    assert np.allclose(np.array(axis_angle), np.array([angle, 0.0, 0.0]))


def test_compute_velocity_from_kinematics_no_freejoint():
    qpos = jp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    qvel = utils.compute_velocity_from_kinematics(
        qpos, dt=1.0, freejoint=False, max_qvel=100.0
    )
    assert qvel.shape == (3, 3)
    assert np.allclose(np.array(qvel[0]), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.array(qvel[1]), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.array(qvel[2]), np.array([0.0, 0.0, 0.0]))


def test_batch_kp_data_non_continuous():
    kp_data = jp.zeros((10, 6))
    batched = utils.batch_kp_data(kp_data, n_frames_per_clip=4, continuous=False)
    assert batched.shape == (2, 4, 6)


def test_batch_kp_data_continuous():
    kp_data = jp.zeros((30, 6))
    batched = utils.batch_kp_data(kp_data, n_frames_per_clip=10, continuous=True)
    assert batched.shape == (3, 20, 6)


def test_enable_xla_flags_sets_env_on_gpu(monkeypatch):
    class FakeBackend:
        platform = "gpu"

    monkeypatch.setattr(utils, "get_backend", lambda: FakeBackend())
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    utils.enable_xla_flags()
    assert "XLA_FLAGS" in os.environ
