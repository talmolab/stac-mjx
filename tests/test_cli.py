from pathlib import Path

from stac_mjx import cli
from stac_mjx.config import compose_config


def test_run_pipeline_invokes_dependencies(monkeypatch, tmp_path, config_dir):
    cfg = compose_config(config_dir, "config", [])

    monkeypatch.setattr(
        cli.stac_mjx, "load_keypoint_data", lambda cfg, base_path=None: ("kp", "names")
    )
    monkeypatch.setattr(
        cli.stac_mjx,
        "run_stac",
        lambda cfg, kp_data, kp_names, base_path=None: (
            "calibration_path",
            "ik_path",
        ),
    )

    calibration_path, ik_path = cli.run_pipeline(cfg, base_path=Path(tmp_path))

    assert calibration_path == "calibration_path"
    assert ik_path == "ik_path"


def test_parse_args_defaults():
    args, overrides = cli.parse_args([])
    assert args.config_path == "configs"
    assert args.config_name == "config"
    assert overrides == []
