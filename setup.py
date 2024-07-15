"""Setup file for stac."""
from setuptools import setup, find_packages

''' WARNING: This file is deprecated, and is only included for the purpose of 
    debugging some conda env installation quirks: the uncommented code
    doesn't produce the ptxas error that can arise intermittently 
    from the environment.yaml install. However it's missing some includes
    to make viz_usage.py work.
    
    The commented version below is included because for the sake of 
    comparison only. 
'''

setup(
    name="stac-mjx",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "six >= 1.12.0",
        "clize >= 4.0.3",
        "absl-py >= 0.7.1",
        "mujoco-mjx >= 3.0.0",
        "enum34",
        "future",
        # 'futures',
        "glfw",
        "lxml",
        "numpy",
        "pyopengl",
        "pyparsing",
        "h5py >= 2.9.0",
        "scipy >= 1.2.1",
        "pyyaml",
        "opencv-python",
        
    ],
)

'''

"""Setup file for stac."""
from setuptools import setup, find_packages

setup(
    name="stac-mjx",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "six >= 1.12.0",
        "clize >= 4.0.3",
        "absl-py >= 0.7.1",
        "mujoco-mjx >= 3.0.0",
        "enum34",
        "future",
        # 'futures',
        "glfw",
        "lxml",
        "numpy",
        "pyopengl",
        "pyparsing",
        "h5py >= 2.9.0",
        "scipy >= 1.2.1",
        "pyyaml",
        "opencv-python",
        
    ],
)

'''
