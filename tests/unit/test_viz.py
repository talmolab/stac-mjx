import types
from pathlib import Path

import numpy as np

from stac_mjx import viz


def test_viz_stac_calls_render(monkeypatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(MJCF_PATH="models/rodent.xml", RENDER_FPS=30)
    )
    data = types.SimpleNamespace(
        qpos=np.zeros((2, 3)),
        kp_data=np.zeros((2, 6)),
        kp_names=["a", "b"],
        offsets=np.zeros((2, 3)),
    )

    monkeypatch.setattr(viz.io, "load_stac_data", lambda path: (cfg, data))

    calls = {}

    class FakeStac:
        def __init__(self, xml_path, cfg_in, kp_names):
            calls["xml_path"] = xml_path
            calls["kp_names"] = kp_names

        def render(self, *args, **kwargs):
            calls["render"] = True
            return ["frame"]

    monkeypatch.setattr(viz, "Stac", FakeStac)

    cfg_out, frames = viz.viz_stac(
        tmp_path / "fake.h5",
        n_frames=1,
        save_path=tmp_path / "out.mp4",
        base_path=tmp_path,
    )

    assert cfg_out is cfg
    assert frames == ["frame"]
    assert calls["render"] is True
    assert calls["xml_path"] == Path(tmp_path) / "models/rodent.xml"
