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
import stac_base
import argparse


# jax.disable_jit(disable=True)

# If your machine is low on ram:
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.6'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                    'runs and saves root opt')
    parser.add_argument('-p', '--save_path', type=str, help='save path')
    parser.add_argument('-t', '--tol', type=float, help='optimizer tolerance')

    args = parser.parse_args()

    offset_path = args.save_path

    start_time = time.time()
    
    param_path = "../params/params.yaml"
    utils.init_params(param_path)

    print(f"setting tolerance to {args.tol}")
    utils.params['Q_TOL'] = args.tol
    
    rat_xml = "../models/rodent_stac.xml"
    rat23 = "../models/rat23.mat"
    data_path = "../save_data_AVG.mat"
    # data_path = "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1/DANNCE/predict03/save_data_AVG.mat" 

    fit_path = "floating_fit.p"
    transform_path = "floating_transform.p"
    utils.params['FTOL'] = 1e-05
    utils.params['n_fit_frames'] = 500
    utils.params['N_ITERS'] = 1
    skip_transform = True

    model = mujoco.MjModel.from_xml_path(rat_xml)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    model.opt.iterations = 1
    model.opt.ls_iterations = 4

    start_time = time.time()

    root = mjcf.from_path(rat_xml)

    # Default ordering of mj sites is alphabetical, so we reorder to match
    kp_names = utils.loadmat(rat23)["joint_names"]
    utils.params["kp_names"] = kp_names

    # argsort returns the indices that would sort the array
    stac_keypoint_order = np.argsort(kp_names)
    # Load kp_data, /1000 to scale data (from mm to meters i think?)
    kp_data = utils.loadmat(data_path)["pred"][:] / 1000

    kp_data = prep_kp_data(kp_data, stac_keypoint_order)

    # setup for fit
    physics, mj_model = set_body_sites(root)
    part_opt_setup(physics)

    # Run root optimization
    # Create mjx model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    # Get and set the offsets of the markers
    offsets = jnp.copy(stac_base.get_site_pos(mjx_model))
    offsets *= utils.params['SCALE_FACTOR']

    # print(mjx_model.site_pos, mjx_model.site_pos.shape)
    mjx_model = stac_base.set_site_pos(mjx_model, offsets)

    # forward is used to calculate xpos and such
    mjx_data = mjx.kinematics(mjx_model, mjx_data)
    mjx_data = mjx.com_pos(mjx_model, mjx_data)
    mjx_data = root_optimization(mjx_model, mjx_data, kp_data)

    data = {
            "kp_data": kp_data,
            "qpos": [mjx_data.qpos[:]],
            "offsets": offsets,
            # "walker_body_sites": [stac_base.get_site_xpos(mjx_data)],
            "xpos": [mjx_data.xpos[:]],
            "names_qpos": initialize_part_names(physics) # utils.params["part_names"],
            # "names_xpos": physics.named.data.xpos.axes.row.names,
        }
    if os.path.dirname(offset_path) != "":
        os.makedirs(os.path.dirname(offset_path), exist_ok=True)
    with open(offset_path, "wb") as output_file:
        pickle.dump(data, output_file, protocol=2)

    print(f"Job complete in {time.time()-start_time}")
