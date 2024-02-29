import mujoco
from jax.lib import xla_bridge
from dm_control import mjcf
import numpy as np

import os
import pickle
import time
import argparse

import controller as ctrl
import utils


def get_clip(kp_data, n_frames):
    import random
    max_index = kp_data.shape[0] - n_frames + 1
    rand_start = random.randint(0, max_index)
    return kp_data[rand_start:rand_start+n_frames,:]


def main():
    # If your machine is low on ram:
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.6' 
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"   
    # When using nvidia gpu do this thing
    if xla_bridge.get_backend().platform == 'gpu':
       os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        )

    
    """Processes command-line arguments and prints a message based on tolerance."""
    parser = argparse.ArgumentParser(description=
                                    'calls fit and transform, using the given optimizer tolerance')
    parser.add_argument('-fp', '--fit_path', type=str, help='fit path')
    parser.add_argument('-tp', '--transform_path', type=str, help='transform path')
    parser.add_argument('-qt', '--qtol', type=float, help='q optimizer tolerance')
    parser.add_argument('-mt', '--mtol', type=float, help='m optimizer tolerance')
    parser.add_argument('-n', '--n_fit_frames', type=int, help='number of frames to fit')
    parser.add_argument('-s', '--skip_transform', type=int, help='True if skip transform')

    args = parser.parse_args()

    utils.init_params("././params/params.yaml")

    if not args.fit_path:
        raise Exception("arg fit_path required")
    if not args.transform_path:
        raise Exception("arg transform_path required")
    if args.qtol:
        print(f"setting tolerance to {args.qtol}")
        utils.params['Q_TOL'] = args.qtol
    if args.n_fit_frames:
        print(f"setting fit frames to {args.n_fit_frames}")
        utils.params['n_fit_frames'] = args.n_fit_frames

    # setting paths
    fit_path = args.fit_path
    transform_path = args.transform_path
    
    ratpath = "././models/rodent_stac.xml"
    rat23path = "././models/rat23.mat"
    model = mujoco.MjModel.from_xml_path(ratpath)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    model.opt.iterations = 1
    model.opt.ls_iterations = 4

    # Need to download this data file and provide the path
    # data_path = "save_data_AVG.mat"
    data_path = "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/DANNCE/predict03/save_data_AVG.mat" 

    root = mjcf.from_path(ratpath)

    # Default ordering of mj sites is alphabetical, so we reorder to match
    kp_names = utils.loadmat(rat23path)["joint_names"]
    utils.params["kp_names"] = kp_names

    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)
    # Load kp_data, /1000 to scale data (from mm to meters i think?)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000

    kp_data = ctrl.prep_kp_data(kp_data, stac_keypoint_order)

    # setup for fit
    physics, mj_model = ctrl.create_body_sites(root)
    ctrl.part_opt_setup(physics)
    
    # Running fit then transform
    print(f"kp_data shape: {kp_data.shape}")
    print(f"Running fit() on {utils.params['n_fit_frames']}")
    clip = get_clip(kp_data, utils.params['n_fit_frames'])
    print(f"clip shape: {clip.shape}")
    mjx_model, q, x, walker_body_sites, kp_data = ctrl.fit(mj_model, clip)

    fit_data = ctrl.package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data
    )

    print(f"saving data to {fit_path}")
    ctrl.save(fit_data, fit_path)

    if args.skip_transform==1:
        print("skipping transform()")
        return
    
    print("Running transform()")
    with open(fit_path, "rb") as file:
        fit_data = pickle.load(file)

    offsets = fit_data["offsets"] 
    kp_data = ctrl.chunk_kp_data(kp_data)
    mjx_model, q, x, walker_body_sites, kp_data = ctrl.transform(mj_model, kp_data, offsets)

    transform_data = ctrl.package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data, batched=True
    )
    
    print(f"saving data to {transform_path}")
    ctrl.save(transform_data, transform_path)

if __name__ == "__main__":
    start_time = time.time()
    main()

    print(f"Job complete in {time.time()-start_time}")
