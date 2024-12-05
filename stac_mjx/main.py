"""User-level API to run stac."""

import jax
from jax import numpy as jp

import pickle
import time
import logging
from omegaconf import DictConfig, OmegaConf

from stac_mjx import io
from stac_mjx import utils
from stac_mjx.stac import Stac
from pathlib import Path
from typing import List, Union
import hydra
from functools import partial


def load_configs(
    config_dir: Union[Path, str], config_name: str = "config"
) -> DictConfig:
    """Initializes configs with hydra.

    Args:
        config_dir ([Path, str]): Absolute path to config directory.

    Returns:
        DictConfig: stac.yaml config to use in run_stac()
    """
    # Initialize Hydra and set the config path
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose the configuration by specifying the config name
        cfg = hydra.compose(config_name=config_name)
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

    utils.enable_xla_flags()

    start_time = time.time()

    # Getting paths
    fit_offsets_path = base_path / cfg.stac.fit_offsets_path
    ik_only_path = base_path / cfg.stac.ik_only_path

    xml_path = base_path / cfg.model.MJCF_PATH

    stac = Stac(xml_path, cfg, kp_names)

    # Initialize function to infer velocity from kinematics
    vmap_compute_velocity = jax.vmap(
        partial(
            utils.compute_velocity_from_kinematics,
            dt=stac._mj_model.opt.timestep,
            freejoint=stac._freejoint,
        )
    )

    # Run fit_offsets if not skipping
    if cfg.stac.skip_fit_offsets != 1:
        kps = kp_data[: cfg.stac.n_fit_frames]
        logging.info(f"Running fit. Mocap data shape: {kps.shape}")
        fit_offsets_data = stac.fit_offsets(kps)
        # Vmap this if multiple clips (only do this in ik_only?)
        # if cfg.stac.infer_qvels:
        #     qvels = vmap_compute_velocity(
        #         qpos_trajectory=fit_offsets_data["qpos"].reshape((1, -1))
        #     )
        #     fit_offsets_data["qvel"] = qvels
        logging.info(f"saving data to {fit_offsets_path}")
        io.save(fit_offsets_data, fit_offsets_path)

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
        fit_offsets_data = pickle.load(file)
    offsets = fit_offsets_data["offsets"]

    print(f"kp_data shape: {kp_data.shape}")
    ik_only_data = stac.ik_only(kp_data, offsets)
    batched_qpos = ik_only_data["qpos"].reshape(
        (cfg.stac.num_clips, kp_data.shape[0] // cfg.stac.num_clips, -1)
    )
    print(batched_qpos.shape, batched_qpos)
    # Vmap this if multiple clips
    if cfg.stac.infer_qvels:
        t_vel = time.time()
        qvels = vmap_compute_velocity(qpos_trajectory=batched_qpos)
        ik_only_data["qvel"] = qvels
        print(f"Finished compute velocity in {time.time() - t_vel}")

    logging.info(
        f"Saving data to {ik_only_path}. Finished in {time.time() - start_time} seconds"
    )
    io.save(ik_only_data, ik_only_path)

    return fit_offsets_path, ik_only_path
