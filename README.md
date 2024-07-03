# stac-mjx :rat:
Implementation of [STAC](https://ieeexplore.ieee.org/document/7030016) using [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

This is part of the VNL project. 

## Installation

stac-mjx relies on many prerequisites, therefore we suggest installing in a new conda environment.

### Conda Environment (`environment.yaml`)

Create and activate the `stac-mjx-env` environment:

```
conda env create -f environment.yaml
conda activate stac-mjx-env
```

### `setup.py` script

In a terminal in a new conda environment, execute:

```
pip install .
```

## Usage
1. Update the .yaml files in `config/` with the proper information (details WIP).

2. For new data, first run stac on just a small subset of the data with

    `python core/main.py test.skip_transform=True`
    
    Note: this currently will fail w/o supplying a data file.


3. Render the resulting data using `mujoco_viz()` from within `viz_usage.ipynb`. Currently, this uses headless rendering on CPU via `mesalab`, which requires its own setup. To set up (currently on supported on Linux), execute the following commands sequentially:



We recommend creating a new Jupyter notebooks kernel with:

```
python -m ipykernel install --user --name stac-mjx-env --display-name "Python (stac-mjx-env)"
```

4. After tuning parameters and confirming the small clip is processed well, run through the whole thing with
    `python stac-mjx/main.py` 