"""Tests for stac_mjx.replay_fit helper functions."""

from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from stac_mjx.replay_fit import (
    _detect_root_body,
    _load_h5_for_replay,
    _model_extent,
    _normalize_shapes,
    _resolve_xml_path,
)

RODENT_XML = Path("models/rodent.xml")


# --- _resolve_xml_path ---


def test_resolve_xml_path_relative_to_h5(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>")
    h5 = tmp_path / "fit.h5"

    result = _resolve_xml_path("model.xml", h5)
    assert result == xml.resolve()


def test_resolve_xml_path_nested_relative_to_h5(tmp_path):
    (tmp_path / "models").mkdir()
    xml = tmp_path / "models" / "robot.xml"
    xml.write_text("<mujoco/>")
    h5 = tmp_path / "fit.h5"

    result = _resolve_xml_path("models/robot.xml", h5)
    assert result == xml.resolve()


def test_resolve_xml_path_basename_fallback(tmp_path):
    xml = tmp_path / "robot.xml"
    xml.write_text("<mujoco/>")
    h5 = tmp_path / "fit.h5"

    result = _resolve_xml_path("some/other/path/robot.xml", h5)
    assert result == xml.resolve()


def test_resolve_xml_path_not_found(tmp_path):
    h5 = tmp_path / "fit.h5"
    with pytest.raises(FileNotFoundError, match="Could not find MJCF"):
        _resolve_xml_path("nonexistent.xml", h5)


def test_resolve_xml_path_absolute(tmp_path):
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>")

    result = _resolve_xml_path(str(xml), tmp_path / "fit.h5")
    assert result == xml


# --- _load_h5_for_replay ---


def _write_stac_h5(path, qpos, config_yaml=None, kp_data=None, marker_sites=None):
    with h5py.File(path, "w") as f:
        f.create_dataset("qpos", data=qpos)
        if config_yaml is not None:
            f.create_dataset("config", data=np.bytes_(config_yaml))
        if kp_data is not None:
            f.create_dataset("kp_data", data=kp_data)
        if marker_sites is not None:
            f.create_dataset("marker_sites", data=marker_sites)


def test_load_h5_basic(tmp_path):
    qpos = np.random.randn(10, 7)
    _write_stac_h5(tmp_path / "test.h5", qpos)

    data = _load_h5_for_replay(tmp_path / "test.h5")
    np.testing.assert_array_equal(data["qpos"], qpos)
    assert "config" not in data


def test_load_h5_with_config(tmp_path):
    qpos = np.random.randn(5, 7)
    cfg = "model:\n  MJCF_PATH: models/rodent.xml\n  SCALE_FACTOR: 0.9\n"
    _write_stac_h5(tmp_path / "test.h5", qpos, config_yaml=cfg)

    data = _load_h5_for_replay(tmp_path / "test.h5")
    assert data["config"]["model"]["MJCF_PATH"] == "models/rodent.xml"
    assert data["config"]["model"]["SCALE_FACTOR"] == 0.9


def test_load_h5_with_markers(tmp_path):
    n_frames, n_kps = 10, 5
    qpos = np.random.randn(n_frames, 7)
    kp_data = np.random.randn(n_frames, n_kps * 3)
    marker_sites = np.random.randn(n_frames, n_kps, 3)
    _write_stac_h5(
        tmp_path / "test.h5", qpos, kp_data=kp_data, marker_sites=marker_sites
    )

    data = _load_h5_for_replay(tmp_path / "test.h5")
    np.testing.assert_array_equal(data["kp_data"], kp_data)
    np.testing.assert_array_equal(data["marker_sites"], marker_sites)


def test_load_h5_missing_qpos(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("kp_data", data=np.zeros((5, 3)))

    with pytest.raises(ValueError, match="does not contain 'qpos'"):
        _load_h5_for_replay(path)


# --- _normalize_shapes ---


def test_normalize_shapes_flat():
    data = {
        "qpos": np.random.randn(20, 7),
        "kp_data": np.random.randn(20, 15),
        "marker_sites": np.random.randn(20, 5, 3),
    }
    result = _normalize_shapes(data)
    assert result["qpos"].shape == (20, 7)
    assert result["kp_data"].shape == (20, 5, 3)
    assert result["marker_sites"].shape == (20, 5, 3)


def test_normalize_shapes_batched():
    n_clips, n_per_clip, nq, n_kps = 3, 10, 7, 4
    data = {
        "qpos": np.random.randn(n_clips, n_per_clip, nq),
        "kp_data": np.random.randn(n_clips, n_per_clip, n_kps * 3),
        "marker_sites": np.random.randn(n_clips, n_per_clip, n_kps, 3),
    }
    result = _normalize_shapes(data)
    assert result["qpos"].shape == (30, nq)
    assert result["kp_data"].shape == (30, n_kps, 3)
    assert result["marker_sites"].shape == (30, n_kps, 3)


def test_normalize_shapes_no_markers():
    data = {"qpos": np.random.randn(10, 7)}
    result = _normalize_shapes(data)
    assert result["qpos"].shape == (10, 7)
    assert "kp_data" not in result


# --- _detect_root_body ---


@pytest.fixture
def rodent_model():
    return mujoco.MjModel.from_xml_path(str(RODENT_XML))


def test_detect_root_body_auto(rodent_model):
    bid = _detect_root_body(rodent_model)
    name = mujoco.mj_id2name(rodent_model, mujoco.mjtObj.mjOBJ_BODY, bid)
    assert name == "torso"


def test_detect_root_body_explicit(rodent_model):
    bid = _detect_root_body(rodent_model, "pelvis")
    name = mujoco.mj_id2name(rodent_model, mujoco.mjtObj.mjOBJ_BODY, bid)
    assert name == "pelvis"


def test_detect_root_body_invalid(rodent_model):
    with pytest.raises(ValueError, match="not found"):
        _detect_root_body(rodent_model, "nonexistent_body")


def test_detect_root_body_fallback():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="custom_root")
    model = spec.compile()
    bid = _detect_root_body(model)
    assert bid == 1


# --- _model_extent ---


def test_model_extent_rodent(rodent_model):
    extent = _model_extent(rodent_model)
    assert extent > 0.0


def test_model_extent_empty_model():
    spec = mujoco.MjSpec()
    model = spec.compile()
    extent = _model_extent(model)
    assert extent == 0.01  # fallback


# --- CLI argument parsing ---


def test_main_parse_args_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        from stac_mjx.replay_fit import main
        import sys

        sys.argv = ["stac-viewer", "--help"]
        main()
    assert exc_info.value.code == 0


# --- replay() input validation ---


def test_replay_missing_h5():
    from stac_mjx.replay_fit import replay

    with pytest.raises(FileNotFoundError, match="H5 file not found"):
        replay("nonexistent.h5")


def test_replay_non_stac_h5(tmp_path):
    from stac_mjx.replay_fit import replay

    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=np.zeros((5, 3)))

    with pytest.raises(ValueError, match="does not contain 'qpos'"):
        replay(path)


def test_replay_no_xml_no_config(tmp_path):
    from stac_mjx.replay_fit import replay

    path = tmp_path / "no_config.h5"
    _write_stac_h5(path, np.random.randn(5, 7))

    with pytest.raises(ValueError, match="No MJCF_PATH"):
        replay(path)
