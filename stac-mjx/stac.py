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
from colorama import Fore, Style

@hydra.main(config_path="../config", config_name="stac", version_base=None)
def run_stac(cfg: DictConfig):
    global_cfg = hydra.compose(config_name="rodent")
    print(f"cfg: {OmegaConf.to_yaml(cfg)}")
    print(f"global_cfg: {OmegaConf.to_yaml(cfg)}")
    utils.init_params(OmegaConf.to_container(global_cfg, resolve=True))
    
    # setting paths
    fit_path = cfg.paths.fit_path
    transform_path = cfg.paths.transform_path
    
    ratpath = cfg.paths.xml 
    kp_names = utils.params['KP_NAMES']
    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)   

    data_path = cfg.paths.data_path

    # Load kp_data, /1000 to scale data (from mm to meters)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000
    
    # First half of the data
    # kp_data = jnp.split(kp_data, 2)[0]
    
    kp_data = ctrl.prep_kp_data(kp_data, stac_keypoint_order)

    # Set up mjcf
    root = mjcf.from_path(ratpath)
    physics, mj_model = ctrl.create_body_sites(root)
    ctrl.part_opt_setup(physics)
    
    mj_model.opt.solver = {
      'cg': mujoco.mjtSolver.mjSOL_CG,
      'newton': mujoco.mjtSolver.mjSOL_NEWTON,
    }[cfg.mujoco.solver.lower()]
    
    mj_model.opt.iterations = cfg.mujoco.iterations
    mj_model.opt.ls_iterations = cfg.mujoco.ls_iterations  
    mj_model.opt.jacobian = 0 # dense

    # Run fit if not skipping
    if cfg.test.skip_fit != 1:
        print(f"kp_data shape: {kp_data.shape}")
        
        if cfg.sampler == "first":
            print("sample the first n frames")
            fit_data = kp_data[cfg.stac.first_start:cfg.first_start + cfg.n_fit_frames]
        elif cfg.sampler == "every":
            print("sample every x frames")
            every = kp_data.shape[0] // cfg.n_fit_frames
            fit_data = kp_data[::every]
        elif cfg.sampler == "random":  
            print("sample n random frames")
            fit_data = jax.random.choice(jax.random.PRNGKey(0), kp_data, (cfg.n_fit_frames,), replace=False)
        
        print(f"fit_data shape: {fit_data.shape}")
        mjx_model, q, x, walker_body_sites, clip_data = ctrl.fit(mj_model, fit_data)

        fit_data = ctrl.package_data(mjx_model, physics, q, x, walker_body_sites, clip_data)

        logging.info(f"saving data to {fit_path}")
        utils.save(fit_data, fit_path)

    # Stop here if skipping transform
    if cfg.test.skip_transform==1:
        print("skipping transform()")
        end(start_time)
    
    print("Running transform()")
    with open(fit_path, "rb") as file:
        fit_data = pickle.load(file)

    offsets = fit_data["offsets"] 
    kp_data = ctrl.chunk_kp_data(kp_data)
    print(f"kp_data shape: {kp_data.shape}")
    mjx_model, q, x, walker_body_sites, kp_data = ctrl.transform(mj_model, kp_data, offsets)

    transform_data = ctrl.package_data(mjx_model, physics, q, x, walker_body_sites, kp_data, batched=True)
    
    logging.info(f"saving data to {transform_path}")
    utils.save(transform_data, transform_path)
    
    return fit_path, transform_path


if __name__ == "__main__":
    # Don't preallocate RAM?
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false" 
    
    # When using nvidia gpu do this thing
    if xla_bridge.get_backend().platform == 'gpu':
        os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        )
        # Set N_GPUS
        utils.params["N_GPUS"] = jax.local_device_count("gpu")
    run_stac()
    print(f"{Fore.CYAN}{Style.BRIGHT}STAC completed{Style.RESET_ALL}")
        