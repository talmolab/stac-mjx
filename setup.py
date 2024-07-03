"""Setup file for stac."""
from setuptools import setup, find_packages

setup(
    name="stac-mjx",
    version="0.0.1",
    python_requires="3.11",
    packages=find_packages(),
    install_requires=[
        "six >= 1.12.0",
        "clize >= 4.0.3",
        "absl-py >= 0.7.1",
        "mujoco-mjx >= 3.1.5",
        "dm_control",
        "jaxopt",
        "flax",
        "enum34",
        "future",
        "lxml",
        "numpy < 2.0",
        "pyopengl",
        "pyparsing",
        "h5py >= 2.9.0",
        "scipy >= 1.2.1",
        "pyyaml",
        "opencv-python",
        "imageio",
        "matplotlib",
        "hydra-core",
        "optax",
        "colorama"
        
    ],
)
