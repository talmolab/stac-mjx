"""Configuration loading utilities for stac-mjx."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Configuration for body model."""

    MJCF_PATH: str  # Path to model xml
    N_ITERS: int  # Number of iterations for STAC algorithm
    KP_NAMES: list[str]  # Ordered list of keypoint names
    KEYPOINT_MODEL_PAIRS: dict[str, str]  # Mapping from keypoint names to model bodies
    KEYPOINT_INITIAL_OFFSETS: dict[str, str]  # Initial offsets for keypoints
    ROOT_OPTIMIZATION_KEYPOINT: str  # Root optimization keypoint name
    TRUNK_OPTIMIZATION_KEYPOINTS: list[str]  # Trunk optimization keypoint names
    INDIVIDUAL_PART_OPTIMIZATION: dict[
        str, list[str]
    ]  # Part optimization keypoint groups
    KEYPOINT_COLOR_PAIRS: dict[str, str]  # RGBA color values for keypoints
    SCALE_FACTOR: float  # Scale factor for model
    MOCAP_SCALE_FACTOR: float  # Scale factor for mocap data (to convert to meters)
    SITES_TO_REGULARIZE: list[str]  # Sites to regularize during offset optimization
    RENDER_FPS: int  # FPS for rendering
    N_SAMPLE_FRAMES: int  # Number of frames to sample when computing offset residual
    M_REG_COEF: float  # Coefficient for regularization term in offset optimization
    MARKER_SIZE: float = 0.005  # Radius of site marker spheres for visualization


@dataclass
class MujocoConfig:
    """Configuration for Mujoco."""

    solver: str  # Solver to use ('cg' or 'newton')
    iterations: int  # Number of solver iterations
    ls_iterations: int  # Number of line search iterations


@dataclass
class StacConfig:
    """Configuration for STAC."""

    calibration_path: str  # Save path for calibrate() output
    ik_path: str  # Save path for run_ik() output
    data_path: str  # Path to mocap data
    n_calibration_frames: int  # Number of frames to use during calibration
    skip_calibration: bool  # Skip calibrate() step if True
    skip_ik: bool  # Skip run_ik() step if True
    infer_qvels: bool  # Infer qvels if True
    n_frames_per_clip: int  # Number of frames per IK chunk
    mujoco: MujocoConfig  # Configuration for Mujoco
    q_opt: "QOptConfig" = field(default_factory=lambda: QOptConfig())


@dataclass
class QOptConfig:
    """q optimization settings."""

    initial_step_damping: float = 1.0
    velocity_smoothness_weight: float = 0.05
    context_frames: int = 16
    coarse_init_stride: int = 12
    coarse_init_max_frames: int = 16
    calibration_max_iterations: int = 150
    ik_max_iterations: int = 150


@dataclass
class Config:
    """Combined configuration for the model and STAC."""

    model: ModelConfig  # Configuration for the model
    stac: StacConfig  # Configuration for STAC


def compose_config(
    config_path: Path | str,
    config_name: str = "config",
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """Load and validate the Hydra configuration."""
    overrides = list(overrides or [])
    overrides.extend(["hydra/job_logging=disabled", "hydra/hydra_logging=disabled"])

    config_dir = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    structured_config = OmegaConf.structured(Config)
    merged_cfg = OmegaConf.merge(structured_config, cfg)
    return merged_cfg
