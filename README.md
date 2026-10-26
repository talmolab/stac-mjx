# stac-mjx :rat:
`stac-mjx` is an implementation of the [STAC](https://ieeexplore.ieee.org/document/7030016) algorithm for inverse kinematics on markerless motion tracking data, using MuJoCo-compatible body models. It uses [MuJoCo XLA (MJX)](https://mujoco.readthedocs.io/en/stable/mjx.html) for GPU parallelization of the MuJoCo physics.

This is part of [MIMIC-MJX](https://mimic-mjx.talmolab.org/).

## Installation

### Using `uv`

Install STAC and its dependencies from `pyproject.toml`.

**Prerequisites:** Git, Python 3.11 or 3.12, and the [uv](https://docs.astral.sh/uv/) package manager. CUDA support requires Linux and a compatible NVIDIA GPU and driver; see [JAX's GPU installation requirements](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

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
cd stac-mjx
```
2. Create and activate a virtual environment:
```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Choose one installation command for your hardware. The CUDA extras install JAX's CUDA libraries through pip; the NVIDIA driver must already be installed.

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
For CPU development, use `uv pip install -e ".[dev]"`.

For Jupyter notebooks, install `uv pip install -e ".[notebooks]"`.
This extra includes the Jupyter kernel and NWB viewer widgets.

Check formatting and lint with the same Ruff rules used by TRACK-MJX and VNL-Playground:

```bash
ruff format --check stac_mjx/ tests/ run_stac.py
ruff check stac_mjx/ tests/ run_stac.py
```

4. Verify the installation:
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Available devices: {jax.devices()}')"
```
5. For notebooks, register the environment as a Jupyter kernel after installing the `[notebooks]` extra:
```bash
python -m ipykernel install --user --name=stac-mjx --display-name="Python (stac-mjx)"
```

### JAX configuration

STAC uses `~/.cache/stac-mjx/jax` for its persistent compilation cache by default. Set `JAX_COMPILATION_CACHE_DIR` to choose another directory, or `JAX_ENABLE_COMPILATION_CACHE=false` to disable it, before starting Python. STAC preserves cache settings configured through the environment or `jax.config.update(...)` before importing `stac_mjx`, including compilation-time and entry-size thresholds.

If you need optional `XLA_FLAGS`, set them before starting Python or Jupyter, as described in the [JAX guidance](https://docs.jax.dev/en/latest/xla_flags.html). STAC leaves these flags unchanged.

### Headless Rendering Setup

Rendering needs an OpenGL backend. Calibration and IK can run without creating a rendering context. The following settings apply to Linux.

Set the variables **before starting Python or Jupyter**. If MuJoCo is already imported in a notebook, restart its kernel after selecting a backend.

For hardware rendering on an NVIDIA GPU without a display, use EGL. The host needs an NVIDIA driver with EGL support:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

For CPU software rendering without a display, use OSMesa. On Debian or Ubuntu, install its system library:

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

Use the equivalent OSMesa package on other Linux distributions. For desktop rendering with a display, use MuJoCo's default backend; unset `MUJOCO_GL` and `PYOPENGL_PLATFORM` if you previously selected a headless backend.

Check that the selected backend can render a frame:

```bash
python - <<'PY'
import mujoco

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
print(f"Rendered RGB frame: {frame.shape}")
PY
```

The expected shape is `(64, 64, 3)`. If EGL fails to initialize, check the host's driver/EGL libraries or use OSMesa. These settings choose the rendering backend; they do not select the device used by JAX for IK.

## Usage

There are a couple of recommended ways to run stac-mjx: basic usage through a command line interface, and writing a custom script using the high-level API.

First, configure your body model and STAC parameters in `configs/`:

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Selects defaults for `model` and `stac` (copy/rename for custom presets) |
| `configs/model/*.yaml` | MuJoCo model path, keypoint names/order, `KEYPOINT_MODEL_PAIRS` mappings, initial offsets, scale factors |
| `configs/stac/*.yaml` | Data paths (`stac.data_path`), clip sizes, calibration/IK output paths, solver settings |

Ensure `KEYPOINT_MODEL_PAIRS` covers every keypoint in your mocap data and that `stac.data_path` points to your `.nwb`, `.mat`, or `.h5` file. Use Hydra overrides to experiment without editing files, e.g., `stac-mjx stac.data_path=path/to/data.nwb model.MJCF_PATH=models/rodent.xml`.

To infer velocities (`stac.infer_qvels=True`), set `stac.data_fps` to the sampling frequency in Hz of the keypoint frames passed to STAC. For example, use `stac.data_fps=50` for 50 Hz input. STAC produces one `qpos` frame per input frame and computes `qvel` with `dt = 1 / stac.data_fps`. If you keep every second frame from a 100 Hz recording, set `stac.data_fps=50`. Input frames must be evenly spaced in time. `model.RENDER_FPS` controls video playback, and the MuJoCo timestep controls simulation; neither sets the input data frequency.

### Command Line Interface
If no customization is needed, you can run the full pipeline from the CLI:

```bash
stac-mjx --config-path configs --config-name config
```

Common options:
- `--base-path`: base directory for data/models (defaults to current working directory)
- `--print-config`: show the composed Hydra config and exit

Hydra overrides can be appended after the CLI flags. For example, to change the data path and number of calibration frames:

```bash
stac-mjx --config-path configs --config-name config stac.data_path=path/to/data.nwb stac.n_calibration_frames=100
```

### Custom Python Script/Jupyter Notebook

You can use a set of high-level functions instead of the end-to-end CLI script to decouple data preparation and configuration loading from execution of the STAC algorithm. This makes it easy to load custom data without adhering to the file structure or content assumptions imposed by the CLI.

> **Try the [demo notebook](https://github.com/talmolab/stac-mjx/blob/main/demos/rodent_demo.ipynb) to see it in action!**

The following is the full CLI script run as a sequence of function calls:

1. Run stac-mjx with its basic api: `load_stac_config` for loading config, `load_keypoint_data` for loading keypoints, and `run_stac` for registration.

   ```python
   import stac_mjx
   from pathlib import Path


   # Choose parent directory as base path for data files
   base_path = Path.cwd().parent

   # Load config (uses `configs/config.yaml` defaults)
   cfg = stac_mjx.load_stac_config(base_path / "configs")

   # Load keypoint data
   kp_data, sorted_kp_names = stac_mjx.load_keypoint_data(cfg, base_path)

   # Run stac
   calibration_path, ik_path = stac_mjx.run_stac(
       cfg, kp_data, sorted_kp_names, base_path=base_path
   )
   ```

2. Render a video of the final result:
   ```python
   import mediapy as media

   data_path = base_path / "demo_calibration.h5"
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
