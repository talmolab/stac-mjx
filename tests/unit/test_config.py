from stac_mjx.config import compose_config


def test_compose_config_loads_defaults():
    cfg = compose_config("tests/configs", "config")
    assert cfg.model.MJCF_PATH.endswith("models/rodent.xml")
    assert cfg.stac.n_fit_frames > 0


def test_compose_config_applies_overrides():
    cfg = compose_config("tests/configs", "config", ["stac.n_fit_frames=5"])
    assert cfg.stac.n_fit_frames == 5
