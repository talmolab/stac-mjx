"""Shared paths for repository configurations and test recordings."""

from pathlib import Path

import pytest


@pytest.fixture
def config_dir():
    return Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def mocap_mat():
    return "tests/data/test_rodent_mocap_1000_frames.mat"


@pytest.fixture
def mocap_nwb():
    return "tests/data/test_rodent_mocap_1000_frames.nwb"


@pytest.fixture
def mocap_h5():
    return "data/rodent_mocap_1000_frames.h5"


@pytest.fixture
def mouse_mocap_h5():
    return "data/mouse_mocap_3600_frames.h5"
