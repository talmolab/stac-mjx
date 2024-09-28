# stac-mjx :rat:
`stac-mjx` is an implementation of the [STAC](https://ieeexplore.ieee.org/document/7030016) algorithm for inverse kinematics on markerless motion capture data. It's written in [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) for hardware acceleration . 

This is part of the Virtual Neurosceince Lab (VNL) project.

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
   To ensure all of the above changes are encapsulated in your Jupyter kernel, create a new kernel with:
   ```bash
   conda install ipykernel
   python -m ipykernel install --user --name stac-mjx-env --display-name "Python (stac-mjx-env)"
   ```
## High 

## Usage
1. Update the .yaml files in `config/` with the proper information (details WIP).

2. Run stac-mjx with its basic api: `load_configs` for loading configs and `run_stac` for the keypoint registration. Below is an example script, found in `demos/use_api.ipynb`. A CLI script is also provided at `run_stac.py`. Refer to [hydra documention](https://hydra.cc/docs/advanced/override_grammar/basic/) for formatting args to override configs.

   ```python
   import stac_mjx 
   from pathlib import Path

   # Enable XLA flags if on GPU
   stac_mjx.enable_xla_flags()

   # Choose parent directory as base path for data files
   base_path = Path.cwd().parent

   # Load configs
   cfg = stac_mjx.load_configs(base_path / "configs")

   # Load data
   kp_data, sorted_kp_names = stac_mjx.load_data(cfg, base_path)

   # Run stac
   fit_path, ik_only_path = stac_mjx.run_stac(
    cfg,
    kp_data, 
    sorted_kp_names, 
    base_path=base_path
   )
   ```

3. Render the resulting data using `mujoco_viz()` (example notebook found in `demos/viz_usage.ipynb`):
   ```python
   import stac_mjx

   import mediapy as media
   from pathlib import Path
   import os

   base_path = Path.cwd()
   cfg = stac_mjx.load_configs(base_path / "configs")

   stac_cfg, model_cfg = main.load_configs(stac_config_path, model_config_path)

   data_path = base_path / "demo_fit.p"
   n_frames = 250
   save_path = base_path / "videos/direct_render.mp4"

   # Call mujoco_viz
   frames = viz_stac(data_path, cfg, n_frames, save_path, start_frame=0, camera="close_profile", base_path=Path.cwd().parent)

   # Show the video in the notebook (it is also saved to the save_path)
   media.show_video(frames, fps=cfg.model.RENDER_FPS)
   ```
   
4. If the rendering is poor, it's likely that some hyperparameter tuning is necessary. (details WIP)