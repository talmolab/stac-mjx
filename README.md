# stac-mjx
Implementation of STAC using MJX for GPU acceleration. Part of VNL project.

## Requirements and Installation
`stac-mjx` relies on `jax`, which has different versions based on your platform (CPU/GPU/TPU), so build within an environment with `jax` installed. 

* Install prerequisites using the included setup scripts.
```
python setup.py install
```
> **Note**
> currently doesn't work lol, getting `AttributeError: module 'mujoco' has no attribute 'mjtDisableBit'` when running setup_test.py
