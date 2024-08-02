"""Fixtures that return paths to data files"""

import pytest

@pytest.fixture
def load3d_mat_file():
    """Typical nwb file"""
    return "tests/data/rat23.mat"

@pytest.fixture
def mocap_mat_file():
    """Typical mat file"""
    return "tests/data/test_mocap_1000_frames.mat"

@pytest.fixture
def mocap_nwb_file():
    """Typical nwb file"""
    return "tests/data/test_mocap_1000_frames.nwb"


@pytest.fixture
def rodent_load3d_yaml_file():
    """Typical nwb file"""
    return "tests/data/test_rodent_load3d.yaml"

@pytest.fixture
def rodent_yaml_file():
    """Typical nwb file"""
    return "tests/data/test_rodent.yaml"

