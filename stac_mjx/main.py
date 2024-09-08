"""User-level API to run stac."""

from jax import numpy as jp

import pickle
import time
import logging
from omegaconf import DictConfig, OmegaConf

from stac_mjx import utils
from stac_mjx.controller import STAC
from pathlib import Path
from typing import List, Dict, Union
import hydra


def load_configs(config_dir: Union[Path, str]) -> DictConfig:
    """Initializes configs with hydra.

    Args:
        config_dir ([Path, str]): Absolute path to config directory.

    Returns:
        DictConfig: stac.yaml config to use in run_stac()
    """
    # Initialize Hydra and set the config path
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose the configuration by specifying the config name
        cfg = hydra.compose(config_name="config")
    return cfg


def run_stac(
    cfg: DictConfig,
    kp_data: jp.ndarray,
    kp_names: List[str],
    base_path=None,
) -> tuple[str, str]:
    """High level function for running skeletal registration.

    Args:
        cfg (DictConfig): Configs.
        kp_data (jp.ndarray): Mocap keypoints to fit to.
        kp_names (List[str]): Ordered list of keypoint names.
        base_path (Path, optional): Base path for reference files in configs. Defaults to Path.cwd().

    Returns:
        tuple[str, str]: Paths to saved outputs (fit and transform).
    """
    if base_path is None:
        base_path = Path.cwd()

    utils.enable_xla_flags()

    start_time = time.time()

    # Getting paths
    fit_path = base_path / cfg.stac.fit_path
    transform_path = base_path / cfg.stac.transform_path

    xml_path = base_path / cfg.model.MJCF_PATH

    stac = STAC(xml_path, cfg, kp_names)

    # Run fit if not skipping
    if cfg.stac.skip_fit != 1:
        fit_data = kp_data[: cfg.stac.n_fit_frames]
        logging.info(f"Running fit. Mocap data shape: {fit_data.shape}")
        fit_data = stac.fit(fit_data)

        logging.info(f"saving data to {fit_path}")
        utils.save(fit_data, fit_path)

    # Stop here if skipping transform
    if cfg.stac.skip_transform == 1:
        logging.info("skipping transform()")
        return fit_path, None

    logging.info("Running transform()")
    with open(fit_path, "rb") as file:
        fit_data = pickle.load(file)

    offsets = fit_data["offsets"]

    logging.info(f"kp_data shape: {kp_data.shape}")
    transform_data = stac.transform(kp_data, offsets)

    logging.info(
        f"Saving data to {transform_path}. Finished in {time.time() - start_time} seconds"
    )
    utils.save(transform_data, transform_path)

    return fit_path, transform_path
