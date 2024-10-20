"""User-level API to run stac."""

from jax import numpy as jp

import pickle
import time
import logging
from omegaconf import DictConfig, OmegaConf

from stac_mjx import io
from stac_mjx import op_utils
from stac_mjx.stac import Stac
from pathlib import Path
from typing import List, Union
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
        tuple[str, str]: Paths to saved outputs (fit_offsets and ik_only).
    """
    if base_path is None:
        base_path = Path.cwd()

    op_utils.enable_xla_flags()

    start_time = time.time()

    # Getting paths
    fit_offsets_path = base_path / cfg.stac.fit_offsets_path
    ik_only_path = base_path / cfg.stac.ik_only_path

    xml_path = base_path / cfg.model.MJCF_PATH

    stac = Stac(xml_path, cfg, kp_names)

    # Run fit_offsets if not skipping
    if cfg.stac.skip_fit_offsets != 1:
        fit_data = kp_data[: cfg.stac.n_fit_frames]
        logging.info(f"Running fit. Mocap data shape: {fit_data.shape}")
        fit_data = stac.fit_offsets(fit_data)

        logging.info(f"saving data to {fit_offsets_path}")
        io.save(fit_data, fit_offsets_path)

    # Stop here if not doing ik only phase
    if cfg.stac.skip_ik_only == 1:
        logging.info("skipping ik_only()")
        return fit_offsets_path, None
    # FLY_MODEL: The elif below must be commented out for fly_model.
    elif kp_data.shape[0] % cfg.model.N_FRAMES_PER_CLIP != 0:
        raise ValueError(
            f"N_FRAMES_PER_CLIP ({cfg.model.N_FRAMES_PER_CLIP}) must divide evenly with the total number of mocap frames({kp_data.shape[0]})"
        )

    logging.info("Running ik_only()")
    with open(fit_offsets_path, "rb") as file:
        fit_data = pickle.load(file)
    offsets = fit_data["offsets"]

    logging.info(f"kp_data shape: {kp_data.shape}")
    ik_only_data = stac.ik_only(kp_data, offsets)

    logging.info(
        f"Saving data to {ik_only_path}. Finished in {time.time() - start_time} seconds"
    )
    io.save(ik_only_data, ik_only_path)

    return fit_offsets_path, ik_only_path
