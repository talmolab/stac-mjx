# stac-mjx :rat:
Implementation of STAC using [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) 

This is part of the VNL project. 

## Installation

stac-mjx relies on many prerequisites. Therefore we suggest installing in a new conda environment.

### Conda Environment (`environment.yaml`)

Create and activate the `stac-mjx-env` environment:

```
conda env create -f environment.yaml
conda activate stac-mjx-env
```

### `setup.py` script

Create a conda environment. Then, in a terminal, enter:

```
pip install .
```

## Usage
1. Update the .yaml files in `config/` with the proper information (details WIP).

2. For new data, first run stac on just a small subset of the data with

    `python core/main.py test.skip_transform=True`

3. Render the resulting data using `mujoco_viz()` (see `viz_usage.ipynb`). We recommend creating a Jupyter notebooks kernel with:

```
python -m ipykernel install --user --name stac-mjx-env --display-name "Python (stac-mjx-env)"
```

4. After tuning parameters and confirming the small clip is processed well, run through the whole thing with
    `python stac-mjx/main.py` 