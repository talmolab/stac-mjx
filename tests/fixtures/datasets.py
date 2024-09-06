"""Fixtures that return paths to data files."""

import pytest


@pytest.fixture
def label3d_mat_file():
    """Typical nwb file."""
    return "tests/data/rat23.mat"


@pytest.fixture
def mocap_mat():
    """Typical mat file."""
    return "tests/data/test_rodent_mocap_1000_frames.mat"


@pytest.fixture
def mocap_nwb():
    """Typical nwb file."""
    return "tests/data/test_rodent_mocap_1000_frames.nwb"
