from stac_mjx.config import compose_config


def test_compose_config_loads_defaults():
    cfg = compose_config("tests/configs", "config")
    assert cfg.model.MJCF_PATH.endswith("models/rodent.xml")
    assert cfg.stac.n_calibration_frames > 0
    assert cfg.stac.q_opt.initial_step_damping == 1.0
    assert cfg.stac.q_opt.acceleration_smoothness_weight == 0.9


def test_compose_config_applies_overrides():
    cfg = compose_config("tests/configs", "config", ["stac.n_calibration_frames=5"])
    assert cfg.stac.n_calibration_frames == 5
