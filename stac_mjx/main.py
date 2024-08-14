"""User-level API to run stac."""

import mujoco
import jax
from jax import numpy as jp
from dm_control import mjcf

import pickle
import time
import logging
from omegaconf import DictConfig, OmegaConf

from stac_mjx import utils
from stac_mjx import controller as ctrl
from pathlib import Path


def load_configs(stac_config_path: str, model_config_path: str) -> DictConfig:
    """Initializes configs.

    Args:
        stac_config_path (str): path to stac yaml file
        model_config_path (str): path to model yaml file

    Returns:
        DictConfig: stac.yaml config to use in run_stac()
    """
    utils.init_params(
        OmegaConf.to_container(OmegaConf.load(model_config_path), resolve=True)
    )
    return OmegaConf.load(stac_config_path)


def run_stac(
    cfg: DictConfig, kp_data: jp.ndarray, base_path: Path = Path.cwd()
) -> tuple[str, str]:
    """Runs stac through fit and transform stages (optionally).

    Args:
        cfg (DictConfig): stac config file (standard being stac.yaml)
        kp_data (jp.Array): Keypoint data of shape (frame, X),
                            where X is a flattened array of keypoint coordinates
                            in the order specified by the model config file (KEYPOINT_MODEL_PAIRS)


    Returns:
        tuple[str, str]: (fit_path, transform_path)
    """
    start_time = time.time()

    # Gettings paths
    fit_path = base_path / cfg.paths.fit_path
    transform_path = base_path / cfg.paths.transform_path

    xml_path = base_path / cfg.paths.xml

    # Set up mjcf
    root = mjcf.from_path(xml_path)
    physics, mj_model = ctrl.create_body_sites(root)
    ctrl.part_opt_setup(physics)

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }[cfg.mujoco.solver.lower()]

    mj_model.opt.iterations = cfg.mujoco.iterations
    mj_model.opt.ls_iterations = cfg.mujoco.ls_iterations

    # Runs faster on GPU with this
    mj_model.opt.jacobian = 0  # dense

    # Run fit if not skipping
    if cfg.skip_fit != 1:
        logging.info(f"kp_data shape: {kp_data.shape}")
        logging.info(f"Sampling {cfg.n_fit_frames} random frames for fit")
        fit_data = jax.random.choice(
            jax.random.PRNGKey(0), kp_data, (cfg.n_fit_frames,), replace=False
        )

        logging.info(f"fit_data shape: {fit_data.shape}")
        mjx_model, q, x, walker_body_sites, clip_data = ctrl.fit(mj_model, fit_data)

        fit_data = ctrl.package_data(
            mjx_model, physics, q, x, walker_body_sites, clip_data
        )

        logging.info(f"saving data to {fit_path}")
        utils.save(fit_data, fit_path)

    # Stop here if skipping transform
    if cfg.skip_transform == 1:
        logging.info("skipping transform()")
        return fit_path, None

    logging.info("Running transform()")
    with open(fit_path, "rb") as file:
        fit_data = pickle.load(file)

    offsets = fit_data["offsets"]
    kp_data = ctrl.chunk_kp_data(kp_data)
    logging.info(f"kp_data shape: {kp_data.shape}")
    mjx_model, q, x, walker_body_sites, kp_data = ctrl.transform(
        mj_model, kp_data, offsets
    )

    transform_data = ctrl.package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data, batched=True
    )

    logging.info(
        f"Saving data to {transform_path}. Finished in {time.time() - start_time} seconds"
    )
    utils.save(transform_data, transform_path)

    return fit_path, transform_path
