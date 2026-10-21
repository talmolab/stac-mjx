import os
import numpy as np
import pytest
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


@pytest.mark.parametrize("angle", [0.0, 1e-7, 1e-4, np.pi / 2, np.pi - 1e-4])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_compute_velocity_from_kinematics_rotation(angle, sign):
    qpos = np.zeros((3, 7), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[1, 3:7] = sign * np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    dt = 1 / 30
    qvel = utils.compute_velocity_from_kinematics(qpos, dt=dt)
    expected = np.zeros((3, 6))
    expected[:, 3] = np.array([angle, -angle, -angle]) / dt
    np.testing.assert_allclose(qvel, expected, rtol=1e-6, atol=1e-8)


def test_compute_velocity_from_kinematics_no_freejoint():
    qpos = jp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    qvel = utils.compute_velocity_from_kinematics(qpos, dt=0.1, freejoint=False)
    np.testing.assert_allclose(
        qvel, [[10.0, 20.0, 30.0], [30.0, 40.0, 50.0], [30.0, 40.0, 50.0]]
    )


def test_enable_xla_flags_sets_env_on_gpu(monkeypatch):
    class FakeBackend:
        platform = "gpu"

    monkeypatch.setattr(utils, "get_backend", lambda: FakeBackend())
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    utils.enable_xla_flags()
    assert "XLA_FLAGS" in os.environ
