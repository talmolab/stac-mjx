import utils
import mujoco
import os
import pickle
from scipy.io import savemat 
from dm_control import mjcf
import numpy as np
import jax
from jax import numpy as jnp
import time
from controller import *


# If your machine is low on ram:
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.6'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

# print(f"total envs: {n_envs}")
# fit_data = test_opt(root, kp_data)
# Single clip optimization for first 500 frames
def test_single_clip_fit(root, kp_data):
    # returns fit_data
    print(f"kp_data shape: {kp_data.shape}")
    fit_data = single_clip_opt(root, kp_data[:2000])
    offset_path = "offset_2000_high_maxiter1.p"
    print(f"saving data to {offset_path}")
    save(fit_data, offset_path)

def test_transform(offset_path, root, kp_data):
    print("Running transform()")
    with open(offset_path, "rb") as file:
        data = pickle.load(file)
    offsets = data["offsets"] 
    kp_data, n_envs = chunk_kp_data(kp_data)
    transform_data = transform(root, kp_data, offsets)
    transform_path = "transform1.p"
    print(f"saving data to {transform_path}")
    save(transform_data, transform_path)


import argparse

def main():
    """Processes command-line arguments and prints a message based on tolerance."""
    parser = argparse.ArgumentParser(description=
                                    'calls fit and transform, using the given optimizer tolerance')
    parser.add_argument('--fit_path', type=str, help='fit path')
    parser.add_argument('--transform_path', type=str, help='transform path')
    parser.add_argument('--tol', type=float, help='optimizer tolerance')
    parser.add_argument('--n_fit_frames', type=int, help='number of frames to fit')
    args = parser.parse_args()

    utils.init_params("././params/params.yaml")

    if not args.fit_path:
        raise Exception("arg fit_path required")
    if not args.transform_path:
        raise Exception("arg transform_path required")
    if args.tol:
        print(f"setting tolerance to {args.tol}")
        utils.params['FTOL'] = args.tol
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
    # data_path = "/home/charles/Desktop/save_data_AVG.mat"
    data_path = "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/DANNCE/predict03/save_data_AVG.mat" 

    root = mjcf.from_path(ratpath)

    # Default ordering of mj sites is alphabetical, so we reorder to match
    kp_names = utils.loadmat(rat23path)["joint_names"]
    utils.params["kp_names"] = kp_names

    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)
    # Load kp_data, /1000 to scale data (from mm to meters i think?)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000

    kp_data = prep_kp_data(kp_data, stac_keypoint_order)

    # setup for fit
    physics, mj_model = set_body_sites(root)
    part_opt_setup(physics)
    
    # Running fit then transform

    print(f"kp_data shape: {kp_data.shape}")
    print(f"Running fit() on {utils.params['n_fit_frames']}")
    mjx_model, q, x, walker_body_sites, kp_data = fit(mj_model, kp_data[:utils.params['n_fit_frames']])

    fit_data = package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data
    )

    print(f"saving data to {fit_path}")
    save(fit_data, fit_path)

    print("Running transform()")
    with open(fit_path, "rb") as file:
        fit_data = pickle.load(file)

    offsets = fit_data["offsets"] 
    kp_data, n_envs = chunk_kp_data(kp_data)
    mjx_model, q, x, walker_body_sites, kp_data = transform(mj_model, kp_data, offsets)

    transform_data = package_data(
        mjx_model, physics, q, x, walker_body_sites, kp_data, batched=True
    )
    
    print(f"saving data to {transform_path}")
    save(transform_data, transform_path)

if __name__ == "__main__":
    start_time = time.time()
    main()

    print(f"Job complete in {time.time()-start_time}")
