# stac-mjx :rat:
Implementation of [STAC](https://ieeexplore.ieee.org/document/7030016) using [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). This is part of the VNL project. 

## Installation
stac-mjx relies on many prerequisites, therefore we suggest installing in a new conda environment, using the provided `environment.yaml`:
[Local installation before package is officially published]
1. Clone the repository `git clone https://github.com/talmolab/stac-mjx.git` and `cd` into it
2. Create and activate the `stac-mjx-env` environment:

```
conda env create -f environment.yaml
conda activate stac-mjx-env
```

Our rendering functions support multiple backends: `egl`, `glfw`, and `osmesa`. We show `osmesa` setup as it supports headless rendering, which is common in remote/cluster setups. To set up (currently on supported on Linux), execute the following commands sequentially:
   ```bash
   sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6 
   conda install -c conda-forge glew 
   conda install -c conda-forge mesalib 
   conda install -c anaconda mesa-libgl-cos6-x86_64 
   conda install -c menpo glfw3
   ```
   Finally, set the following environment variables, and reactivate the conda environment:
   ```bash
   conda env config vars set MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
   conda deactivate && conda activate base
   ```
   To ensure all of the above changes are encapsulated in your Jupyter kernel, a create a new kernel with:
   ```bash
   conda install ipykernel
   python -m ipykernel install --user --name stac-mjx-env --display-name "Python (stac-mjx-env)"
   ```


## Usage
1. Update the .yaml files in `config/` with the proper information (details WIP).

2. Run stac-mjx with its basic api: `load_configs` for loading configs and `run_stac` for the keypoint registration. Below is an example script, found in `demos/use_api.ipynb`. 

   ```python
   from stac_mjx import main
   from stac_mjx import utils
   from pathlib import Path

   # Set base path to the parent directory of your config files
   base_path = Path.cwd()
   stac_config_path = base_path / "demos/demo_stac.yaml"
   model_config_path = base_path / "configs/rodent.yaml"

   # Load configs
   cfg = main.load_configs(stac_config_path, model_config_path)

   # Load data
   data_path = base_path / cfg.paths.data_path 
   kp_data = utils.load_data(data_path, utils.params)

   # Run stac
   fit_path, transform_path = main.run_stac(cfg, kp_data, base_path)
   ```

3. Render the resulting data using `mujoco_viz()` (example notebook found in `demos/viz_usage.ipynb`):
   ```python
   import os
   import mediapy as media

   from stac_mjx.viz import mujoco_viz
   from stac_mjx import main
   from stac_mjx import utils

   stac_config_path = "../configs/stac.yaml"
   model_config_path = "../configs/rodent.yaml"

   cfg = main.load_configs(stac_config_path, model_config_path)

   xml_path = "../models/rodent.xml"
   data_path = "../output.p"
   n_frames=250
   save_path="../videos/direct_render.mp4"

   # Call mujoco_viz
   frames = mujoco_viz(data_path, xml_path, n_frames, save_path, start_frame=0)

   # Show the video in the notebook (it is also saved to the save_path)
   media.show_video(frames, fps=utils.params["RENDER_FPS"])
   ```
   
4. If the rendering is poor, it's likely that some hyperparameter tuning is necessary. (details WIP)