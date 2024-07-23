"""This module is the entry point for the stac-mjx algorithm."""

import mujoco
import jax
from jax import numpy as jnp
from jax.lib import xla_bridge
from dm_control import mjcf
import numpy as np

import os
import pickle
import time
import argparse
import random
import logging
import sys
import hydra
from hydra import initialize
from omegaconf import DictConfig, OmegaConf

import utils
import controller as ctrl


def run_stac(cfg: DictConfig):
    """Run the core of the stac-mjx algorithm."""
    # setting paths
    fit_path = cfg.paths.fit_path
    transform_path = cfg.paths.transform_path

    ratpath = cfg.paths.xml
    kp_names = utils.params["KP_NAMES"]
    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)
    data_path = cfg.paths.data_path

    # Load kp_data, /1000 to scale data (from mm to meters)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000

    # Preparing data by reordering and reshaping (TODO: will this stay the same?)
    # Resulting kp_data is of shape (n_frames, n_keypoints)
    kp_data = jnp.array(kp_data[:, :, stac_keypoint_order])
    kp_data = jnp.transpose(kp_data, (0, 2, 1))
    kp_data = jnp.reshape(kp_data, (kp_data.shape[0], -1))

    # Set up mjcf
    root = mjcf.from_path(ratpath)
    physics, mj_model = ctrl.create_body_sites(root)
    ctrl.part_opt_setup(physics)

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }[cfg.mujoco.solver.lower()]

    mj_model.opt.iterations = cfg.mujoco.iterations
    mj_model.opt.ls_iterations = cfg.mujoco.ls_iterations
    mj_model.opt.jacobian = 0  # dense

    # Run fit if not skipping
    if cfg.test.skip_fit != 1:
        logging.info(f"kp_data shape: {kp_data.shape}")
        if cfg.sampler == "first":
            logging.info("Sampling the first n frames")
            fit_data = kp_data[: cfg.n_fit_frames]
        elif cfg.sampler == "every":
            logging.info("Sampling every x frames")
            every = kp_data.shape[0] // cfg.n_fit_frames
            fit_data = kp_data[::every]
        elif cfg.sampler == "random":
            logging.info("Sampling n random frames")
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
    if cfg.test.skip_transform == 1:
        logging.info("skipping transform()")
        return fit_path, "none"

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

    logging.info(f"saving data to {transform_path}")
    utils.save(transform_data, transform_path)

    return fit_path, transform_path


@hydra.main(config_path="../configs", config_name="stac", version_base=None)
def hydra_entry(cfg: DictConfig):
    """Prepare and run the stac-mjx algorith."""
    # Initialize configs and convert to dictionaries
    global_cfg = hydra.compose(config_name="rodent")
    logging.info(f"cfg: {OmegaConf.to_yaml(cfg)}")
    logging.info(f"global_cfg: {OmegaConf.to_yaml(cfg)}")
    utils.init_params(OmegaConf.to_container(global_cfg, resolve=True))

    # Don't preallocate RAM?
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # XLA flags for Nvidia GPU
    if xla_bridge.get_backend().platform == "gpu":
        # Set num. gpus. Enable when support for multiple GPUs is implemented
        # utils.params["N_GPUS"] = jax.local_device_count("gpu")
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
            # These may provide additional speed ups, but are currently disabled
            # due to errors.
            # "--xla_gpu_enable_highest_priority_async_stream=true "
            # "--xla_gpu_enable_async_collectives=true "
            # "--xla_gpu_enable_latency_hiding_scheduler=true "
        )

    return run_stac(cfg)


if __name__ == "__main__":
    hydra_entry()
