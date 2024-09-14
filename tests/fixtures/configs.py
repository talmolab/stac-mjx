"""Fixtures that return paths to data files."""

import pytest
from pathlib import Path

_BASE_PATH = Path.cwd()


@pytest.fixture
def rodent_config_label3d():
    """Rodent yaml file that refers to label3d data file."""
    return "test_rodent_label3d.yaml"


@pytest.fixture
def rodent_config():
    """Typical model config file."""
    return "test_rodent.yaml"


@pytest.fixture
def mouse_config():
    """Mouse config file w/ keypoint names."""
    return "test_mouse.yaml"


@pytest.fixture
def rodent_config_no_kp_names():
    """Typical model config file."""
    return "test_rodent_no_kp_names.yaml"


@pytest.fixture
def rodent_config_less_kp_names():
    """Typical model config file."""
    return "test_rodent_less_kp_names.yaml"


@pytest.fixture
def config():
    """Typical model config file."""
    return _BASE_PATH / "tests/configs"
