import types

import numpy as np
import pytest

from stac_mjx import main


class DummyData:
    def __init__(self, qpos=None, offsets=None):
        self.qpos = qpos if qpos is not None else np.zeros((4, 2))
        self.offsets = offsets if offsets is not None else np.zeros((2, 3))
        self.qvel = np.array([])

    def as_dict(self):
        return {}


class DummyStac:
    def __init__(self, xml_path, cfg, kp_names):
        self.xml_path = xml_path
        self.cfg = cfg
        self.kp_names = kp_names
        self._mj_model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.1))
        self._freejoint = False

    def fit_offsets(self, kps):
        return DummyData()

    def ik_only(self, kp_data, offsets):
        return DummyData(qpos=np.zeros((kp_data.shape[0], 2)))


def make_cfg(
    *,
    skip_fit_offsets,
    skip_ik_only,
    n_frames_per_clip=2,
    infer_qvels=False,
    continuous=False,
):
    stac = types.SimpleNamespace(
        fit_offsets_path="fit.h5",
        ik_only_path="ik.h5",
        skip_fit_offsets=skip_fit_offsets,
        skip_ik_only=skip_ik_only,
        n_fit_frames=2,
        n_frames_per_clip=n_frames_per_clip,
        infer_qvels=infer_qvels,
        continuous=continuous,
    )
    model = types.SimpleNamespace(MJCF_PATH="models/rodent.xml")
    return types.SimpleNamespace(stac=stac, model=model)


def test_run_stac_skip_fit_and_ik_only(monkeypatch):
    cfg = make_cfg(skip_fit_offsets=True, skip_ik_only=True)
    kp_data = np.zeros((4, 6))

    monkeypatch.setattr(main, "Stac", DummyStac)
    monkeypatch.setattr(main.utils, "enable_xla_flags", lambda: None)

    calls = {"save": 0}
    monkeypatch.setattr(
        main.io, "save_data_to_h5", lambda *args, **kwargs: calls.__setitem__("save", 1)
    )

    fit_path, ik_path = main.run_stac(cfg, kp_data, ["a", "b"])
    assert ik_path is None
    assert calls["save"] == 0


def test_run_stac_ik_only_path(monkeypatch):
    cfg = make_cfg(skip_fit_offsets=True, skip_ik_only=False, n_frames_per_clip=2)
    kp_data = np.zeros((4, 6))

    monkeypatch.setattr(main, "Stac", DummyStac)
    monkeypatch.setattr(main.utils, "enable_xla_flags", lambda: None)
    monkeypatch.setattr(main.io, "load_stac_data", lambda path: (cfg, DummyData()))

    calls = {"save": 0}
    monkeypatch.setattr(
        main.io,
        "save_data_to_h5",
        lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1),
    )

    fit_path, ik_path = main.run_stac(cfg, kp_data, ["a", "b"])
    assert ik_path is not None
    assert calls["save"] == 1


def test_run_stac_requires_divisible_frames(monkeypatch):
    cfg = make_cfg(skip_fit_offsets=True, skip_ik_only=False, n_frames_per_clip=3)
    kp_data = np.zeros((4, 6))

    monkeypatch.setattr(main, "Stac", DummyStac)
    monkeypatch.setattr(main.utils, "enable_xla_flags", lambda: None)

    with pytest.raises(ValueError):
        main.run_stac(cfg, kp_data, ["a", "b"])
