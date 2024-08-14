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
   TODO: Use our dataloaders in this example

   ```python
   from stac_mjx import main
   from stac_mjx import utils
   from jax import numpy as jp
   import numpy as np

   stac_config_path = "../configs/stac.yaml"
   model_config_path = "../configs/rodent.yaml"

   cfg = main.load_configs(stac_config_path, model_config_path)
   data_path = "../tests/data/test_pred_only_1000_frames.mat"
   # Set up mocap data
   kp_names = utils.params["KP_NAMES"]
   # argsort returns the indices that would sort the array
   stac_keypoint_order = np.argsort(kp_names)
   data_path = cfg.paths.data_path

   # Load kp_data, /1000 to scale data (from mm to meters)
   kp_data = utils.loadmat(data_path)["pred"][:] / 1000

   # Preparing data by reordering and reshaping
   # Resulting kp_data is of shape (n_frames, n_keypoints)
   kp_data = jp.array(kp_data[:, :, stac_keypoint_order])
   kp_data = jp.transpose(kp_data, (0, 2, 1))
   kp_data = jp.reshape(kp_data, (kp_data.shape[0], -1))

   # Run stac
   main.run_stac(cfg, kp_data)
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