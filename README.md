# stac-mjx :rat:
Implementation of STAC using MJX for GPU acceleration. Part of VNL project.

## Requirements and Installation
`stac-mjx` relies on `jax`, which has different versions based on your platform (CPU/GPU/TPU), so build within an environment with `jax` installed. Refer to https://jax.readthedocs.io/en/latest/installation.html. 

* Install the rest of the prerequisites using the included setup script with
```
pip install .
```

## Usage
1. Update the .yaml files in `config/` with the proper information (details WIP).

2. For new data, first run stac on just a small subset of the data with

    `python stac-mjx/main.py test.skip_transform=True`

3. Render the resulting data using `mujoco_viz()` (see `viz_usage.ipynb`)
4. After tuning parameters and confirming the small clip is processed well, run through the whole thing with
    `python stac-mjx/main.py` 