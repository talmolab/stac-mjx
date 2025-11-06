"""User-level API to run stac."""

import jax
from jax import numpy as jp
import numpy as np
import pickle
import time
import logging
from omegaconf import DictConfig, OmegaConf
from stac_mjx import io, utils
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
        cfg = hydra.compose(
            config_name=config_name,
            overrides=["hydra/job_logging=disabled", "hydra/hydra_logging=disabled"],
        )
        # Convert to structured config
        structured_config = OmegaConf.structured(io.Config)
        OmegaConf.merge(structured_config, cfg)
        print("Config loaded and validated.")
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

    compute_velocity_fn = partial(
        utils.compute_velocity_from_kinematics,
        dt=stac._mj_model.opt.timestep,
        freejoint=stac._freejoint,
    )
    # Initialize function to infer velocity from kinematics
    vmap_compute_velocity_fn = jax.vmap(compute_velocity_fn)

    # Run fit_offsets if not skipping
    if cfg.stac.skip_fit_offsets != 1:
        kps = kp_data[: cfg.stac.n_fit_frames]
        print(f"Running fit. Mocap data shape: {kps.shape}")
        fit_offsets_data = stac.fit_offsets(kps)
        print(f"saving data to {fit_offsets_path}")
        io.save_data_to_h5(
            config=cfg, file_path=fit_offsets_path, **fit_offsets_data.as_dict()
        )
        (fit_offsets_data, fit_offsets_path)
    else:
        print(
            "Skipping fit_offsets. To change this behavior, set cfg.stac.skip_fit_offsets to 0."
        )

    # Stop here if not doing ik only phase
    if cfg.stac.skip_ik_only == 1:
        print(
            "Skipping IK-only phase. To change this behavior, set cfg.stac.skip_ik_only to 0."
        )
        return fit_offsets_path, None
    elif kp_data.shape[0] % cfg.stac.n_frames_per_clip != 0:
        raise ValueError(
            f"n_frames_per_clip ({cfg.stac.n_frames_per_clip}) must divide evenly with the total number of mocap frames({kp_data.shape[0]})"
        )

    print("Running ik_only()")
    cfg, fit_offsets_data = io.load_stac_data(fit_offsets_path)

    offsets = fit_offsets_data.offsets

    print(f"kp_data shape: {kp_data.shape}")
    ik_only_data = stac.ik_only(kp_data, offsets)

    # TODO: if continuous, reshape to remove overlapping frames
    batched_qpos = ik_only_data.qpos.reshape(
        (-1, cfg.stac.n_frames_per_clip, ik_only_data.qpos.shape[-1])
    )
    if cfg.stac.infer_qvels:
        t_vel = time.time()
        qvels = vmap_compute_velocity_fn(qpos_trajectory=batched_qpos)
        # set dict key after reshaping and casting to numpy
        ik_only_data.qvel = np.array(qvels).reshape(-1, *qvels.shape[2:])
        print(f"Finished compute velocity in {time.time() - t_vel}")

    print(
        f"Saving data to {ik_only_path}. Finished in {time.time() - start_time} seconds"
    )
    io.save_data_to_h5(config=cfg, file_path=ik_only_path, **ik_only_data.as_dict())
    return fit_offsets_path, ik_only_path
