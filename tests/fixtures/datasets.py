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


@pytest.fixture
def rodent_config_label3d():
    """Rodent yaml file that refers to label3d data file."""
    return "tests/data/test_rodent_label3d.yaml"


@pytest.fixture
def rodent_config():
    """Typical model config file."""
    return "tests/data/test_rodent.yaml"


@pytest.fixture
def rodent_config_no_kp_names():
    """Typical model config file."""
    return "tests/data/test_rodent_no_kp_names.yaml"


@pytest.fixture
def rodent_config_less_kp_names():
    """Typical model config file."""
    return "tests/data/test_rodent_less_kp_names.yaml"


@pytest.fixture
def stac_config():
    """Typical model config file."""
    return "tests/data/test_stac.yaml"
