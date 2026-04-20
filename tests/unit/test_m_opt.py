"""Numerical correctness tests for closed-form m_opt.

Uses a minimal MuJoCo model (3 bodies in a chain, 3 hinge joints on
different axes, 3 marker sites) to verify the closed-form offset solver
against known ground-truth offsets and a reference loss implementation.
"""

import pytest
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from stac_mjx import utils
from stac_mjx.stac_core import _m_opt


MINIMAL_XML = """\
<mujoco>
  <worldbody>
    <body name="b1" pos="1 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1"/>
      <site name="s1" pos="0.1 0.2 0.3"/>
      <body name="b2" pos="0 1 0">
        <joint name="j2" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.1"/>
        <site name="s2" pos="0.4 0.5 0.6"/>
        <body name="b3" pos="0 0 1">
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1"/>
          <site name="s3" pos="0.15 0.25 0.35"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_and_sites():
    """Minimal MuJoCo/MJX model: 3 bodies, 3 hinge joints, 3 marker sites."""
    mj_model = mujoco.MjModel.from_xml_string(MINIMAL_XML)
    mjx_model, mjx_data = utils.mjx_load(mj_model)

    site_names = ["s1", "s2", "s3"]
    site_idxs = jp.array(
        [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in site_names
        ],
        dtype=jp.int32,
    )
    K = len(site_names)
    return mjx_model, mjx_data, site_idxs, K


def _generate_keypoints(mjx_model, mjx_data, q_trajectory, offsets, site_idxs):
    """Run FK with given offsets and return world-frame markers (T, 3*K)."""
    mjx_model = utils.set_site_pos(mjx_model, offsets, site_idxs)

    def fk_one(q_t):
        d = mjx_data.replace(qpos=q_t)
        d = utils.kinematics(mjx_model, d)
        d = utils.com_pos(mjx_model, d)
        return utils.get_site_xpos(d, site_idxs).flatten()

    return jax.vmap(fk_one)(q_trajectory)


GT_OFFSETS_A = jp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.15, 0.25, 0.35]])
GT_OFFSETS_B = jp.array([[0.2, -0.1, 0.4], [0.3, 0.4, -0.2], [-0.1, 0.3, 0.1]])


def test_identity_pose_recovers_offsets_and_error(model_and_sites):
    mjx_model, mjx_data, site_idxs, K = model_and_sites
    q = jp.zeros((5, mjx_model.nq))
    keypoints = _generate_keypoints(mjx_model, mjx_data, q, GT_OFFSETS_A, site_idxs)

    result = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        jp.zeros((K, 3)), jp.zeros((K, 3)), 0.0, site_idxs,
    )

    np.testing.assert_allclose(result.params, GT_OFFSETS_A, atol=1e-5)
    assert float(result.error) < 1e-8


def test_varied_random_poses(model_and_sites):
    mjx_model, mjx_data, site_idxs, K = model_and_sites
    rng = np.random.RandomState(42)
    q = jp.array(rng.randn(10, int(mjx_model.nq)).astype(np.float32) * 0.5)
    keypoints = _generate_keypoints(mjx_model, mjx_data, q, GT_OFFSETS_A, site_idxs)

    result = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        jp.zeros((K, 3)), jp.zeros((K, 3)), 0.0, site_idxs,
    )

    np.testing.assert_allclose(result.params, GT_OFFSETS_A, atol=1e-5)


def test_sweeping_single_joint(model_and_sites):
    """Sweep joint-1 from 0 to pi/4 across 8 frames."""
    mjx_model, mjx_data, site_idxs, K = model_and_sites
    q = jp.zeros((8, mjx_model.nq))
    q = q.at[:, 0].set(jp.linspace(0.0, jp.pi / 4, 8))
    keypoints = _generate_keypoints(mjx_model, mjx_data, q, GT_OFFSETS_B, site_idxs)

    result = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        GT_OFFSETS_B, jp.zeros((K, 3)), 0.0, site_idxs,
    )

    np.testing.assert_allclose(result.params, GT_OFFSETS_B, atol=1e-5)


def test_large_rotations(model_and_sites):
    """Angles up to ~±1.5 rad on all three joints."""
    mjx_model, mjx_data, site_idxs, K = model_and_sites
    rng = np.random.RandomState(99)
    q = jp.array(rng.randn(15, int(mjx_model.nq)).astype(np.float32) * 1.5)
    keypoints = _generate_keypoints(mjx_model, mjx_data, q, GT_OFFSETS_B, site_idxs)

    result = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        jp.zeros((K, 3)), jp.zeros((K, 3)), 0.0, site_idxs,
    )

    np.testing.assert_allclose(result.params, GT_OFFSETS_B, atol=1e-4)


def test_reg_coef_zero_vs_strong(model_and_sites):
    """Same data, two extremes: reg_coef=0 ignores initial, reg_coef=1e6 collapses to it."""
    mjx_model, mjx_data, site_idxs, K = model_and_sites
    rng = np.random.RandomState(42)
    q = jp.array(rng.randn(10, int(mjx_model.nq)).astype(np.float32) * 0.3)
    keypoints = _generate_keypoints(mjx_model, mjx_data, q, GT_OFFSETS_A, site_idxs)

    # reg_coef=0: even a wildly wrong initial should be ignored
    wrong_initial = jp.ones((K, 3)) * 99.0
    result_noreg = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        wrong_initial, jp.ones((K, 3)), 0.0, site_idxs,
    )
    np.testing.assert_allclose(result_noreg.params, GT_OFFSETS_A, atol=1e-5)

    # reg_coef=1e6: solution collapses to initial_offsets
    result_strong = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        jp.zeros((K, 3)), jp.ones((K, 3)), 1e6, site_idxs,
    )
    np.testing.assert_allclose(result_strong.params, jp.zeros((K, 3)), atol=1e-3)


def test_partial_regularization(model_and_sites):
    """Only the first site is regularized; others should still be exact."""
    mjx_model, mjx_data, site_idxs, K = model_and_sites
    q = jp.zeros((10, mjx_model.nq))
    gt = jp.array([[0.5, 0.5, 0.5]] * K)
    keypoints = _generate_keypoints(mjx_model, mjx_data, q, gt, site_idxs)

    is_reg = jp.zeros((K, 3)).at[0].set(1.0)
    initial_offsets = jp.zeros((K, 3))

    result_strong = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        initial_offsets, is_reg, 1e4, site_idxs,
    )
    result_noreg = _m_opt(
        mjx_model, mjx_data, keypoints, q,
        initial_offsets, is_reg, 0.0, site_idxs,
    )

    # First site pulled toward zero by regularization
    assert float(jp.linalg.norm(result_strong.params[0])) < float(
        jp.linalg.norm(result_noreg.params[0])
    )
    # Unregularized sites are unaffected
    np.testing.assert_allclose(result_strong.params[1:], gt[1:], atol=1e-5)
