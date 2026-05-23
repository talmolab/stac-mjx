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

    def calibrate(self, kps):
        return DummyData()

    def run_ik(self, kp_data, offsets):
        return DummyData(qpos=np.zeros((kp_data.shape[0], 2)))


def make_cfg(
    *,
    skip_calibration,
    skip_ik,
    n_frames_per_clip=2,
    infer_qvels=False,
):
    stac = types.SimpleNamespace(
        calibration_path="calibration.h5",
        ik_path="ik.h5",
        skip_calibration=skip_calibration,
        skip_ik=skip_ik,
        n_calibration_frames=2,
        n_frames_per_clip=n_frames_per_clip,
        infer_qvels=infer_qvels,
    )
    model = types.SimpleNamespace(MJCF_PATH="models/rodent.xml")
    return types.SimpleNamespace(stac=stac, model=model)


def test_run_stac_skip_calibration_and_ik(monkeypatch):
    cfg = make_cfg(skip_calibration=True, skip_ik=True)
    kp_data = np.zeros((4, 6))

    monkeypatch.setattr(main, "Stac", DummyStac)

    calls = {"save": 0, "xla": 0}
    monkeypatch.setattr(
        main.utils,
        "enable_xla_flags",
        lambda: calls.__setitem__("xla", calls["xla"] + 1),
    )
    monkeypatch.setattr(
        main.io, "save_data_to_h5", lambda *args, **kwargs: calls.__setitem__("save", 1)
    )

    calibration_path, ik_path = main.run_stac(cfg, kp_data, ["a", "b"])
    assert ik_path is None
    assert calls["save"] == 0
    assert calls["xla"] == 0


def test_run_stac_ik_path(monkeypatch):
    cfg = make_cfg(skip_calibration=True, skip_ik=False, n_frames_per_clip=2)
    kp_data = np.zeros((4, 6))

    monkeypatch.setattr(main, "Stac", DummyStac)
    monkeypatch.setattr(main.io, "load_stac_data", lambda path: (cfg, DummyData()))

    calls = {"save": 0}
    monkeypatch.setattr(
        main.io,
        "save_data_to_h5",
        lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1),
    )

    calibration_path, ik_path = main.run_stac(cfg, kp_data, ["a", "b"])
    assert ik_path is not None
    assert calls["save"] == 1


def test_run_stac_allows_nondivisible_frames(monkeypatch):
    cfg = make_cfg(skip_calibration=True, skip_ik=False, n_frames_per_clip=3)
    kp_data = np.zeros((4, 6))

    monkeypatch.setattr(main, "Stac", DummyStac)
    monkeypatch.setattr(main.io, "load_stac_data", lambda path: (cfg, DummyData()))

    calls = {"save": 0}
    monkeypatch.setattr(
        main.io,
        "save_data_to_h5",
        lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1),
    )

    calibration_path, ik_path = main.run_stac(cfg, kp_data, ["a", "b"])
    assert calibration_path is not None
    assert ik_path is not None
    assert calls["save"] == 1


def test_run_stac_validates_kp_data_shape(monkeypatch):
    """kp_data.shape[1] must equal len(kp_names) * 3 (closes #42)."""
    cfg = make_cfg(skip_calibration=True, skip_ik=True)
    # 2 keypoint names → expect 6 columns; pass 9 to trigger the error
    kp_data = np.zeros((4, 9))

    monkeypatch.setattr(main, "Stac", DummyStac)

    with pytest.raises(ValueError, match="kp_data"):
        main.run_stac(cfg, kp_data, ["a", "b"])
