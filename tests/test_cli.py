from pathlib import Path
import sys
import types

# Stub optional dependencies used during import
sys.modules.setdefault(
    "ndx_pose",
    types.SimpleNamespace(PoseEstimationSeries=object, PoseEstimation=object),
)

from stac_mjx import cli


def test_compose_config_loads_and_applies_overrides():
    cfg = cli.compose_config("tests/configs", "config", ["stac.n_fit_frames=5"])

    assert cfg.stac.n_fit_frames == 5
    assert cfg.model.MJCF_PATH.endswith("models/rodent.xml")
    # structured config merge should keep required fields
    assert cfg.model.KEYPOINT_MODEL_PAIRS


def test_run_pipeline_invokes_dependencies(monkeypatch, tmp_path):
    cfg = cli.compose_config("tests/configs", "config", [])

    calls = {"xla": 0}

    monkeypatch.setattr(
        cli.stac_mjx,
        "enable_xla_flags",
        lambda: calls.__setitem__("xla", calls["xla"] + 1),
    )
    monkeypatch.setattr(
        cli.stac_mjx, "load_mocap", lambda cfg, base_path=None: ("kp", "names")
    )
    monkeypatch.setattr(
        cli.stac_mjx,
        "run_stac",
        lambda cfg, kp_data, kp_names, base_path=None: ("fit_path", "ik_path"),
    )

    fit_path, ik_path = cli.run_pipeline(cfg, base_path=Path(tmp_path), enable_xla=True)

    assert calls["xla"] == 1
    assert fit_path == "fit_path"
    assert ik_path == "ik_path"
