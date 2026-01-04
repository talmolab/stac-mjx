"""Configuration loading utilities for stac-mjx."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from stac_mjx import io


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

    structured_config = OmegaConf.structured(io.Config)
    merged_cfg = OmegaConf.merge(structured_config, cfg)
    return merged_cfg
