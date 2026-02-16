# stac-mjx :rat:
`stac-mjx` is an implementation of the [STAC](https://ieeexplore.ieee.org/document/7030016) algorithm for inverse kinematics on markerless motion tracking data, using MuJoCo-compatible body models. It uses [MuJoCo XLA (MJX)](https://mujoco.readthedocs.io/en/stable/mjx.html) for GPU parallelization of the MuJoCo physics.

This is part of [MIMIC-MJX](https://mimic-mjx.talmolab.org/).

## Installation

### Option 1: `uv` (fastest)

**Prerequisites:** Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/) package manager, and optionally CUDA 12.x or 13.x for GPU support.

If you don't have uv installed:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

1. Clone the repository:
```bash
git clone https://github.com/talmolab/stac-mjx.git
```
2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install the package with optional dependencies based on your hardware. CUDA 12, CUDA 13, and CPU-only configurations are supported. This should take a few minutes:

For CUDA 12.x:
```bash
uv pip install -e ".[cuda12]"
```

For CUDA 13.x:
```bash
uv pip install -e ".[cuda13]"
```

For CPU-only:
```bash
uv pip install -e .
```

For development, include the `[dev]` extras in addition to the hardware optional dependencies:
```bash
uv pip install -e ".[cuda13,dev]"
```
4. Verify the installation:
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Available devices: {jax.devices()}')"
```
5. Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=stac-mjx --display-name="Python (stac-mjx)"
```

### Option 2: `conda`

stac-mjx relies on many prerequisites, therefore we suggest installing in a new conda environment, using the provided `environment.yaml`:
[Local installation before package is officially published]
1. Clone the repository `git clone https://github.com/talmolab/stac-mjx.git` and `cd` into it
2. Create and activate the `stac-mjx-env` environment:

```
conda env create -f environment.yaml
conda activate stac-mjx-env
```

3. Register the environment as a Jupyter kernel:
```bash
conda install ipykernel
python -m ipykernel install --user --name stac-mjx-env --display-name "Python (stac-mjx-env)"
```

### Headless Rendering Setup

Our rendering functions support multiple backends: `egl`, `glfw`, and `osmesa`. Below is the `osmesa` setup, which supports headless rendering common in remote/cluster environments (currently Linux only).

Install the required system and conda packages:
```bash
sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3
```

Set the following environment variables and reactivate your environment:
```bash
conda env config vars set MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
conda deactivate && conda activate base
```

## Usage

There are a couple of recommended ways to run stac-mjx: basic usage through a command line interface, and writing a custom script using the high-level API.

First, configure your body model and STAC parameters in `configs/`:

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Selects defaults for `model` and `stac` (copy/rename for custom presets) |
| `configs/model/*.yaml` | MuJoCo model path, keypoint names/order, `KEYPOINT_MODEL_PAIRS` mappings, initial offsets, scale factors |
| `configs/stac/*.yaml` | Data paths (`stac.data_path`), clip sizes, fit/IK output paths, solver settings |

Ensure `KEYPOINT_MODEL_PAIRS` covers every keypoint in your mocap data and that `stac.data_path` points to your `.nwb`, `.mat`, or `.h5` file. Use Hydra overrides to experiment without editing files, e.g., `stac-mjx stac.data_path=path/to/data.nwb model.MJCF_PATH=models/rodent.xml`.

### Command Line Interface
If no customization is needed, you can run the full pipeline from the CLI:

```bash
stac-mjx --config-path configs --config-name config
```

Common options:
- `--base-path`: base directory for data/models (defaults to current working directory)
- `--print-config`: show the composed Hydra config and exit
- `--skip-xla-flags`: skip setting XLA environment flags before running

Hydra overrides can be appended after the CLI flags. For example, to change the data path and number of fit frames:

```bash
stac-mjx --config-path configs --config-name config stac.data_path=path/to/data.nwb stac.n_fit_frames=100
```

### Custom Python Script/Jupyter Notebook

You can use a set of high-level functions instead of the end-to-end CLI script to decouple data preparation and configuration loading from execution of the STAC algorithm. This makes it easy to load custom data without adhering to the file structure or content assumptions imposed by the CLI.

> **Try the [demo notebook](https://github.com/talmolab/stac-mjx/blob/main/demos/rodent_demo.ipynb) to see it in action!**

The following is the full CLI script run as a sequence of function calls:

1. Run stac-mjx with its basic api: `load_configs` for loading configs and `run_stac` for the keypoint registration.

   ```python
   import stac_mjx
   from pathlib import Path


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

2. Render a video of the final result:
   ```python
   import mediapy as media

   data_path = base_path / "demo_fit_offsets.h5"
   n_frames = 10
   save_path = base_path / "videos/direct_render.mp4"

   # Call mujoco_viz
   cfg, frames = stac_mjx.viz_stac(
      data_path,
      n_frames,
      save_path,
      start_frame=0,
      camera="close_profile",
      base_path=Path.cwd().parent,
   )

   # Show the video in the notebook (it is also saved to the save_path)
   media.show_video(frames, fps=cfg.model.RENDER_FPS)
   ```

3. If the rendering is poor, it's likely that some hyperparameter tuning is necessary. (details WIP)


### Keypoint Correspondence UI
For establishing the correspondence between motion capture 3D landmarks and keypoints in the virtual body model, we provide a dedicated UI tool at [stac-keypoints-ui](https://github.com/talmolab/stac-keypoints-ui). This tool allows you to visually map your motion capture keypoints to the corresponding locations on the body model, which is essential for accurate inverse kinematics.
