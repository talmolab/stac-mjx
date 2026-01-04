import types

import numpy as np
from jax import numpy as jp

from stac_mjx import compute_stac, utils


class FakeMjxData:
    def __init__(self, qpos, site_xpos=None, xpos=None, xquat=None):
        self.qpos = qpos
        self.site_xpos = site_xpos if site_xpos is not None else jp.zeros((2, 3))
        self.xpos = xpos if xpos is not None else jp.zeros((2, 3))
        self.xquat = xquat if xquat is not None else jp.zeros((2, 4))

    def replace(self, **kwargs):
        return FakeMjxData(
            qpos=kwargs.get("qpos", self.qpos),
            site_xpos=kwargs.get("site_xpos", self.site_xpos),
            xpos=kwargs.get("xpos", self.xpos),
            xquat=kwargs.get("xquat", self.xquat),
        )


class FakeMjxModel:
    def __init__(self, nq, jnt_type, site_pos):
        self.nq = nq
        self.jnt_type = jnt_type
        self.site_pos = site_pos


class FakeStacCore:
    def __init__(self):
        self.q_calls = 0
        self.m_calls = 0

    def q_opt(self, *args, **kwargs):
        self.q_calls += 1
        q0 = args[5]
        res = types.SimpleNamespace(params=q0, state=types.SimpleNamespace(error=0.0))
        return args[1], res

    def m_opt(self, offset0, *args, **kwargs):
        self.m_calls += 1
        return types.SimpleNamespace(
            params=offset0, state=types.SimpleNamespace(error=0.0)
        )


def test_root_optimization_calls_q_opt_twice(monkeypatch):
    monkeypatch.setattr(compute_stac, "mujoco", types.SimpleNamespace(
        mjtJoint=types.SimpleNamespace(mjJNT_SLIDE=1, mjJNT_FREE=0)
    ))
    monkeypatch.setattr(utils, "kinematics", lambda model, data: data)
    monkeypatch.setattr(utils, "com_pos", lambda model, data: data)

    stac_core = FakeStacCore()
    mjx_model = FakeMjxModel(
        nq=7, jnt_type=jp.array([0]), site_pos=jp.zeros((2, 3))
    )
    mjx_data = FakeMjxData(qpos=jp.zeros(7))
    kp_data = jp.zeros((1, 6))
    lb = jp.zeros(7)
    ub = jp.ones(7)
    site_idxs = jp.array([0, 1])
    trunk_kps = jp.array([True, True])

    out = compute_stac.root_optimization(
        stac_core,
        mjx_model,
        mjx_data,
        kp_data,
        root_kp_idx=0,
        lb=lb,
        ub=ub,
        site_idxs=site_idxs,
        trunk_kps=trunk_kps,
    )

    assert isinstance(out, FakeMjxData)
    assert stac_core.q_calls == 2


def test_offset_optimization_updates_site_pos(monkeypatch):
    monkeypatch.setattr(utils, "kinematics", lambda model, data: data)

    def set_site_pos(model, offsets, site_idxs):
        model.site_pos = offsets
        return model

    monkeypatch.setattr(utils, "set_site_pos", set_site_pos)

    stac_core = FakeStacCore()
    mjx_model = FakeMjxModel(
        nq=7, jnt_type=jp.array([0]), site_pos=jp.zeros((2, 3))
    )
    mjx_data = FakeMjxData(qpos=jp.zeros(7))
    kp_data = jp.zeros((4, 6))
    offsets = jp.zeros((2, 3))
    q = jp.zeros((4, 7))

    mjx_model, mjx_data, offset_opt = compute_stac.offset_optimization(
        stac_core,
        mjx_model,
        mjx_data,
        kp_data,
        offsets,
        q,
        n_sample_frames=2,
        is_regularized=jp.zeros(6),
        site_idxs=jp.array([0, 1]),
        m_reg_coef=0.0,
    )

    assert stac_core.m_calls == 1
    assert np.allclose(np.array(offset_opt), np.array(offsets).flatten())
    assert np.allclose(np.array(mjx_model.site_pos), np.array(offsets))


def test_pose_optimization_runs_all_frames(monkeypatch):
    monkeypatch.setattr(utils, "kinematics", lambda model, data: data)
    monkeypatch.setattr(utils, "com_pos", lambda model, data: data)

    stac_core = FakeStacCore()
    mjx_model = FakeMjxModel(
        nq=7, jnt_type=jp.array([0]), site_pos=jp.zeros((2, 3))
    )
    mjx_data = FakeMjxData(qpos=jp.zeros(7))
    kp_data = jp.zeros((2, 6))

    result = compute_stac.pose_optimization(
        stac_core,
        mjx_model,
        mjx_data,
        kp_data,
        lb=jp.zeros(7),
        ub=jp.ones(7),
        site_idxs=jp.array([0, 1]),
        indiv_parts=[],
    )

    _, qposes, _, _, marker_sites, _, frame_error = result
    assert qposes.shape == (2, 7)
    assert len(marker_sites) == 2
    assert len(frame_error) == 2
