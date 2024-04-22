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
from omegaconf import DictConfig, OmegaConf

import utils
# Gotta do this before importing controller
utils.init_params("././params/params.yaml")

import controller as ctrl

def get_clip(kp_data, n_frames):
    max_index = kp_data.shape[0] - n_frames + 1
    rand_start = random.randint(0, max_index)
    print(f"fit clip starts at index {rand_start} and runs for {n_frames}, ending at {rand_start+n_frames}")
    return kp_data[rand_start:rand_start+n_frames,:]


def end(start_time):
    print(f"Job complete in {time.time()-start_time}")
    exit()
    
@hydra.main(config_path="", config_name="stac_test_config", version_base=None)
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    start_time = time.time()
    
    # Allocate 90% instead of 75% of GPU vram
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9' 
    
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


    if not cfg.paths.fit_path:
        raise Exception("arg fit_path required")
    if not cfg.paths.transform_path:
        raise Exception("arg transform_path required")
    if cfg.stac.n_fit_frames:
        print(f"setting fit frames to {cfg.stac.n_fit_frames}")
        utils.params['n_fit_frames'] = cfg.stac.n_fit_frames
    
    # setting paths
    fit_path = cfg.paths.fit_path
    transform_path = cfg.paths.transform_path
    
    ratpath = cfg.paths.xml 
    rat23path = "././models/rat23.mat"
    kp_names = utils.loadmat(rat23path)["joint_names"]
    utils.params["kp_names"] = kp_names
    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)   
    
    # model = mujoco.MjModel.from_xml_path(ratpath)
    
    # Need to download this data file and provide the path
    # data_path = "save_data_AVG.mat"
    data_path = cfg.paths.data_path

    # Load kp_data, /1000 to scale data (from mm to meters)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000

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
        print(f"Running fit() on {utils.params['n_fit_frames']}")
        # clip = get_clip(kp_data, utils.params['n_fit_frames'])
        # print(f"clip shape: {clip.shape}")
        mjx_model, q, x, walker_body_sites, clip_data = ctrl.fit(mj_model, kp_data[:utils.params['n_fit_frames']])

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
    
    end(start_time)
    
if __name__ == "__main__":
    main()
