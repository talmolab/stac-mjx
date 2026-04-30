"""Configuration loading utilities for stac-mjx."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Configuration for body model."""

    MJCF_PATH: str  # Path to model xml
    FTOL: float  # Tolerance for optimization
    ROOT_FTOL: float  # Tolerance for root optimization TODO: currently unused
    LIMB_FTOL: float  # Tolerance for limb optimization TODO: currently unused
    N_ITERS: int  # Number of iterations for STAC algorithm
    N_ITER_Q: int  # Number of iterations for q optimization
    KP_NAMES: list[str]  # Ordered list of keypoint names
    KEYPOINT_MODEL_PAIRS: dict[str, str]  # Mapping from keypoint names to model bodies
    KEYPOINT_INITIAL_OFFSETS: dict[str, str]  # Initial offsets for keypoints
    ROOT_OPTIMIZATION_KEYPOINT: str  # Root optimization keypoint name
    TRUNK_OPTIMIZATION_KEYPOINTS: list[str]  # Trunk optimization keypoint names
    INDIVIDUAL_PART_OPTIMIZATION: dict[str, list[str]]  # Part optimization keypoint groups
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

    fit_offsets_path: str  # Save path for fit_offsets() output
    ik_only_path: str  # Save path for ik_only() output
    data_path: str  # Path to mocap data
    num_clips: int  # Number of clips in mocap data
    n_fit_frames: int  # Number of frames to fit offsets to
    skip_fit_offsets: bool  # Skip fit_offsets() step if True
    skip_ik_only: bool  # Skip ik_only() step if True
    infer_qvels: bool  # Infer qvels if True
    n_frames_per_clip: int  # Number of frames per clip
    mujoco: MujocoConfig  # Configuration for Mujoco
    continuous: bool  # Whether the data is continuous (to allow for edge effects post-processing)


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
