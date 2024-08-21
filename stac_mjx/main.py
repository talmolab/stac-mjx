"""User-level API to run stac."""

from jax import numpy as jp

import pickle
import time
import logging
from omegaconf import DictConfig, OmegaConf

from stac_mjx import utils
from stac_mjx import controller as ctrl
from stac_mjx.controller import STAC
from pathlib import Path
from typing import List


def load_configs(stac_config_path: Path, model_config_path: Path) -> DictConfig:
    """Initializes configs.

    Args:
        stac_config_path (str): path to stac yaml file
        model_config_path (str): path to model yaml file

    Returns:
        DictConfig: stac.yaml config to use in run_stac()
    """

    return OmegaConf.load(stac_config_path), OmegaConf.to_container(
        OmegaConf.load(model_config_path), resolve=True
    )


def run_stac(
    stac_cfg: DictConfig,
    model_cfg,
    kp_data: jp.ndarray,
    kp_names: List[str],
    base_path: Path = Path.cwd(),
) -> tuple[str, str]:
    start_time = time.time()

    # Getting paths
    fit_path = base_path / stac_cfg.fit_path
    transform_path = base_path / stac_cfg.transform_path

    xml_path = base_path / model_cfg["MJCF_PATH"]

    stac = STAC(xml_path, stac_cfg, model_cfg, kp_names)

    # Run fit if not skipping
    if stac_cfg.skip_fit != 1:
        fit_data = kp_data[: stac_cfg.n_fit_frames]
        logging.info(f"Running fit. Mocap data shape: {fit_data.shape}")
        fit_data = stac.fit(fit_data)

        logging.info(f"saving data to {fit_path}")
        utils.save(fit_data, fit_path)

    # Stop here if skipping transform
    if stac_cfg.skip_transform == 1:
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
