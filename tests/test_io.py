import hydra
import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict

from stac_mjx import io

"""
Test data loaders for supported file types.
"""

# Expected keypoint order matches KEYPOINT_MODEL_PAIRS key order in test configs.
# This is critical: sorted_kp_names must stay in sync with the reordered kp_data
# columns, otherwise loss functions receive misaligned data.
EXPECTED_RODENT_KP_ORDER = [
    "AnkleL",
    "AnkleR",
    "EarL",
    "EarR",
    "ElbowL",
    "ElbowR",
    "FootL",
    "FootR",
    "HandL",
    "HandR",
    "HipL",
    "HipR",
    "KneeL",
    "KneeR",
    "ShoulderL",
    "ShoulderR",
    "Snout",
    "SpineF",
    "SpineL",
    "SpineM",
    "TailBase",
    "WristL",
    "WristR",
]

EXPECTED_MOUSE_KP_ORDER = [
    "Nose",
    "Ear_R",
    "Ear_L",
    "TTI",
    "Head",
    "Trunk",
    "Tail_0",
    "Tail_1",
    "Tail_2",
    "Tail_3",
    "Tail_4",
    "TailTip",
    "Shoulder_L",
    "Shoulder_R",
    "Knee_L",
    "Knee_R",
    "Neck",
    "Elbow_L",
    "Elbow_R",
    "Wrist_L",
    "Wrist_R",
    "Forepaw_L",
    "Forepaw_R",
    "Heel_L",
    "Heel_R",
    "Spine",
    "Eye_L",
    "Eye_R",
    "Hip_L",
    "Hip_R",
    "Lisfranc_L",
    "Lisfranc_R",
    "MTP_L",
    "MTP_R",
]


def load_config_with_overrides(
    config_dir, stac_data_path_override=None, model_override=None
):
    overrides = []
    if stac_data_path_override:
        overrides.append(f"stac.data_path={stac_data_path_override}")
    if model_override:
        overrides.append(f"model={model_override}")

    # Initialize Hydra and set the config path
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose the configuration by specifying the config name
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def test_load_nwb(config_dir, mocap_nwb):
    """
    Test loading data from .nwb file.
    """
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_nwb)
    data, sorted_kp_names = io.load_keypoint_data(cfg)
    assert data.shape == (1000, 69)
    assert sorted_kp_names == EXPECTED_RODENT_KP_ORDER


def test_load_mat_no_label3d(config_dir, mocap_mat):
    """
    Test loading data from .mat file.
    """
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_mat)
    data, sorted_kp_names = io.load_keypoint_data(cfg)
    assert data.shape == (1000, 69)
    assert sorted_kp_names == EXPECTED_RODENT_KP_ORDER


def test_load_mat_w_label3d(config_dir, mocap_mat):
    """
    Test loading data from a .mat file w/ labels file
    """
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_mat)
    with open_dict(cfg.model):
        del cfg.model.KP_NAMES
        cfg.model.KP_NAMES_LABEL3D_PATH = "tests/data/rat23.mat"
    data, sorted_kp_names = io.load_keypoint_data(cfg)
    assert data.shape == (1000, 69)
    assert sorted_kp_names == EXPECTED_RODENT_KP_ORDER


def test_load_mat_no_kp_names(config_dir, mocap_mat):
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_mat)
    cfg.model.KP_NAMES = None

    with pytest.raises(ValueError, match="Keypoint names not provided"):
        data, sorted_kp_names = io.load_keypoint_data(cfg)


def test_load_mat_less_kp_names(config_dir, mocap_mat):
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_mat)
    cfg.model.KP_NAMES = ["Snout"]

    with pytest.raises(ValueError, match="Number of keypoint names"):
        data, sorted_kp_names = io.load_keypoint_data(cfg)


def test_load_mat_more_kp_names(config_dir, mocap_mat):
    """ValueError when KP_NAMES has more entries than data keypoints (closes #42)."""
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_mat)
    cfg.model.KP_NAMES.extend(["ExtraKP1", "ExtraKP2"])

    with pytest.raises(ValueError, match="Number of keypoint names"):
        data, sorted_kp_names = io.load_keypoint_data(cfg)


def test_load_h5(config_dir, mocap_h5):
    """Load the shared rodent recording in HDF5 format."""
    cfg = load_config_with_overrides(config_dir, stac_data_path_override=mocap_h5)
    data, sorted_kp_names = io.load_keypoint_data(cfg)
    assert data.shape == (1000, 69)
    assert sorted_kp_names == EXPECTED_RODENT_KP_ORDER


def test_load_mouse_keypoint_order(config_dir, mouse_mocap_h5):
    """Preserve the mouse model's distinct 34-keypoint mapping."""
    cfg = load_config_with_overrides(
        config_dir,
        stac_data_path_override=mouse_mocap_h5,
        model_override="mouse",
    )

    data, sorted_kp_names = io.load_keypoint_data(cfg)

    assert data.shape == (3600, 102)
    assert sorted_kp_names == EXPECTED_MOUSE_KP_ORDER


def test_load_keypoint_data_reorders_coordinates_with_names(monkeypatch):
    cfg = OmegaConf.create(
        {
            "stac": {"data_path": "markers.mat"},
            "model": {
                "KEYPOINT_MODEL_PAIRS": {"a": "body_a", "b": "body_b"},
                "MOCAP_SCALE_FACTOR": 1.0,
            },
        }
    )
    data = np.array(
        [
            [[10, 70, 1], [20, 80, 2], [30, 90, 3]],
            [[40, 100, 4], [50, 110, 5], [60, 120, 6]],
        ]
    )
    monkeypatch.setattr(
        io, "load_dannce", lambda *args, **kwargs: (data, ["b", "extra", "a"])
    )

    actual, names = io.load_keypoint_data(cfg)

    assert names == ["a", "b"]
    np.testing.assert_array_equal(
        actual, [[1, 2, 3, 10, 20, 30], [4, 5, 6, 40, 50, 60]]
    )
