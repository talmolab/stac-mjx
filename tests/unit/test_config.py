from omegaconf import DictConfig

from stac_mjx import load_stac_config


def test_load_stac_config_loads_defaults(config_dir):
    cfg = load_stac_config(config_dir)
    assert isinstance(cfg, DictConfig)
    assert cfg.model.MJCF_PATH == "models/rodent.xml"
    assert cfg.model.KEYPOINT_MODEL_PAIRS
    assert cfg.stac.calibration_path == "demo_calibration.h5"
    assert cfg.stac.n_calibration_frames == 10
    assert cfg.stac.q_opt.initial_step_damping == 1.0
    assert cfg.stac.q_opt.velocity_smoothness_weight == 0.05


def test_load_stac_config_applies_overrides(config_dir):
    cfg = load_stac_config(config_dir, overrides=["stac.n_calibration_frames=5"])
    assert cfg.stac.n_calibration_frames == 5
