# stac-mjx :rat:
Implementation of [STAC](https://ieeexplore.ieee.org/document/7030016) using [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). This is part of the VNL project. 

## Installation
stac-mjx relies on many prerequisites, therefore we suggest installing in a new conda environment. We currently provide two installation methods; 1. conda environment, 2. pip intall via `setup.py`.

1. Conda Environment (`environment.yaml`)
   Create and activate the `stac-mjx-env` environment:
   ```
   conda env create -f environment.yaml
   conda activate stac-mjx-env
   ```
2. `setup.py` script
   In a terminal in a new conda environment, from the `stac-mjx` directory, execute:
   ```
   pip install .
   ```

## Usage
1. Update the .yaml files in `config/` with the proper information (details WIP).

2. For new data, first run stac on just a small subset of the data with

    `python core/main.py test.skip_transform=True`
    
    Note: this currently will fail w/o supplying a data file.

3. Render the resulting data using `mujoco_viz()` from within `viz_usage.ipynb`. Currently, this uses headless rendering on CPU via `osmesa`, which requires its own setup. To set up (currently on supported on Linux), execute the following commands sequentially:
   ```
   sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6 
   conda install -c conda-forge glew 
   conda install -c conda-forge mesalib 
   conda install -c anaconda mesa-libgl-cos6-x86_64 
   conda install -c menpo glfw3
   ```
   Finally, set the following environment variables, and reactivate the conda environment:
   ```
   conda env config vars set MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
   conda deactivate && conda activate base
   ```
   To ensure all of the above changes are encapsulated in your Jupyter kernel, a create a new kernel with:
   ```
   python -m ipykernel install --user --name stac-mjx-env --display-name "Python (stac-mjx-env)"
   ```

4. After tuning parameters and confirming the small clip is processed well, run through the whole thing with
    `python stac-mjx/main.py` 