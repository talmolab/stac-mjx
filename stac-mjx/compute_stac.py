"""Compute stac optimization on data."""
from scipy.io import savemat
from dm_control.locomotion.walkers import rescale
import mujoco
from mujoco import mjx
import jax
from jax import jit, vmap
import jax.numpy as jnp
import stac_base
import utils
import pickle
import os
from typing import List, Dict, Tuple, Text

def root_opt_setup(mjx_model, mjx_data, kp_data):
    q0 = jnp.copy(mjx_data.qpos[:])
    q_copy = jnp.copy(q0)
    # Set the center to help with finding the optima (does not need to be exact)
    q0 = q0.at[:3].set(kp_data[frame, :][12:15])

    qs_to_opt = jnp.zeros_like(q0, dtype=bool)
    qs_to_opt = qs_to_opt.at[:7].set(True)
    
    # Limit the optimizer to a subset of qpos
    bounds = stac_base.get_q_bounds(mjx_model)
    bounds = (b[qs_to_opt] for b in bounds)

    q0 = q0[qs_to_opt]
    
    return q0, q_copy, bounds
    
@jax.vmap
def root_optimization(mjx_model, mjx_data, kp_data, frame: int = 0):
    """Optimize only the root.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary
        frame (int, optional): Frame to optimize
    """
    
    print("Root Optimization:")
    ftol = utils.params["ROOT_FTOL"]
    
    q0, q_copy, bounds = root_opt_setup(mjx_model, mjx_data, kp_data)
    
    mjx_data = stac_base.q_opt(
        mjx_model, 
        mjx_data,
        kp_data[frame, :],
        q0,
        q_copy,
        bounds=bounds,
        ftol=ftol,
    )

    # First optimize over the trunk
    trunk_kps = [
        any([n in kp_name for n in utils.params["TRUNK_OPTIMIZATION_KEYPOINTS"]])
        for kp_name in utils.params["kp_names"]
    ]
    trunk_kps = jnp.repeat(jnp.array(trunk_kps), 3)
    
    q0, q_copy, bounds = root_opt_setup(mjx_model, mjx_data, kp_data)
    
    mjx_data = stac_base.q_opt(
        mjx_model, 
        mjx_data,
        kp_data[frame, :],
        q0,
        q_copy,
        bounds=bounds,
        ftol=ftol,
        kps_to_opt=trunk_kps,
    )
    
    return mjx_data

@jax.vmap
def offset_optimization(mjx_model, mjx_data, kp_data, offsets, q, maxiter: int = 100):
    key = jax.random.PRNGKey(0)
    time_indices = jax.random.randint(
        key, shape=[utils.params["N_SAMPLE_FRAMES"]], minval=0, maxval=utils.params["n_frames"], 
    )
    
    print("Begining offset optimization:")

    mjx_model, mjx_data = stac_base.m_phase(
        mjx_model, 
        mjx_data,
        kp_data,
        time_indices,
        q,
        offsets,
        reg_coef=utils.params["M_REG_COEF"],
        maxiter=maxiter,
    )


@jax.vmap
def pose_optimization(mjx_model, mjx_data, kp_data) -> Tuple:
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary.

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    q = jnp.array([])
    x = jnp.array([])
    walker_body_sites = jnp.array([])
    
    parts = utils.params["indiv_parts"]

    # Iterate through all of the frames in the clip
    frames = jnp.arange(utils.params["n_frames"])
    
    ftol = utils.params["FTOL"]
    part_ftol = utils.params["LIMB_FTOL"]
    
    bounds = stac_base.get_q_bounds(mjx_model)
    
    # Get bounds and loss functions for part optimization
    parts_bounds = [(b[part] for b in bounds) for part in parts]
    part_opt_loss_fs = [jit(partial(stac_base.q_loss, qs_to_opt=part)) for part in parts]
    
    print("Pose Optimization:")
    
    @jit
    def f(mjx_data, kp_data, n_frame, bounds, parts_bounds, part_opt_loss_fs, parts):
        q0 = jnp.copy(mjx_data.qpos[:])
        q_copy = jnp.copy(q0)
        
        # Get initial parameters for part optimization
        part_q0s = [q0[part] for part in parts] 
        
        mjx_data = stac_base.q_opt(
            mjx_model, 
            mjx_data,
            kp_data[n_frame, :],
            q0,
            q_copy,
            bounds=bounds,
            ftol=ftol,
        )

        for loss_fn in part_opt_loss_fs:
            mjx_data = stac_base.q_opt(
                mjx_model, 
                mjx_data,
                kp_data[frame, :],
                q0,
                q_copy,
                loss_fn=loss_fn,
                bounds=bounds,
                ftol=part_ftol,
            )
        
        return mjx_data
    
    for n_frame in frames:
        mjx_data = f(mjx_data, 
                         kp_data, 
                         n_frame, 
                         bounds, 
                         parts_bounds, 
                         part_opt_loss_fs, 
                         parts)
        q = jnp.append(q, mjx_data.qpos)
        x = jnp.append(x, mjx_data.xpos)
        walker_body_sites = jnp.append(walker_body_sites,
            stac_base.get_site_xpos(mjx_data)
        )
        print(f"Frame {n_frame} done")
    
    print("Pose Optimization done")
    return mjx_data, q, walker_body_sites, x

def initialize_part_names(physics):
    # Get the ids of the limbs, accounting for quaternion and pos
    part_names = physics.named.data.qpos.axes.row.names
    for _ in range(6):
        part_names.insert(0, part_names[0])
    return part_names

# TODO: Do we need to package the data like this? DOes it need to be jax compatible?
def package_data(mjx_model, physics, q, x, walker_body_sites, kp_data):
    # Extract pose, offsets, data, and all parameters
    offsets = stac_base.get_site_pos(mjx_model).copy()
    
    names_xpos = physics.named.data.xpos.axes.row.names
    x = x.reshape(-1, x.shape[-1])
    q = q.reshape(-1, q.shape[-1])
    kp_data = kp_data.reshape(-1, kp_data.shape[-1])
    data = {
        "qpos": q,
        "xpos": x,
        "walker_body_sites": walker_body_sites,
        "offsets": offsets,
        "names_qpos": utils.params["part_names"],
        "names_xpos": names_xpos,
        "kp_data": jnp.copy(kp_data),
    }
    
    for k, v in utils.params.items():
        data[k] = v
    
    return data


# class STAC:
#     def __init__(
#         self,
#         param_path: Text,
#     ):
#         """Initialize STAC

#         Args:
#             param_path (Text): Path to parameters .yaml file.
#         """
#         self._properties = util.load_params(param_path)
#         self._properties["data"] = None
#         self._properties["n_frames"] = None

#         # Default ordering of mj sites is alphabetical, so we reorder to match
#         self._properties["kp_names"] = util.loadmat(self._properties["SKELETON_PATH"])["joint_names"]
#         # argsort returns the indices that would sort the array
#         self._properties["stac_keypoint_order"] = jnp.argsort(
#             self._properties["kp_names"]
#         )
#         for property_name in self._properties.keys():

#             def getter(self, name=property_name):
#                 return self._properties[name]

#             def setter(self, value, name=property_name):
#                 self._properties[name] = value

#             setattr(STAC, property_name, property(fget=getter, fset=setter))

#     def _prepare_data(self, kp_data: jnp.ndarray) -> jnp.ndarray:
#         """Prepare the data for STAC.

#         Args:
#             kp_data (jnp.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).

#         Returns:
#             jnp.ndarray: Keypoint data in meters (n_frames, n_keypoints * 3).
#         """
#         kp_data = kp_data[:, :, self.stac_keypoint_order]
#         kp_data = jnp.transpose(kp_data, (0, 2, 1))
#         kp_data = jnp.reshape(kp_data, (kp_data.shape[0], -1))
#         return kp_data

#     def fit(self, kp_data: jnp.ndarray) -> "STAC":
#         """Calibrate and fit the model to keypoints.

#         Performs three rounds of alternating marker and quaternion optimization. Optimal
#         results with greater than 200 frames of data in which the subject is moving.

#         Args:
#             keypoints (jnp.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).
#                 Keypoint order must match the order in the skeleton file.

#         Example:
#             st = st.fit(keypoints)

#         Returns: STAC object with fitted model.
#         """
#         kp_data = self._prepare_data(kp_data)
#         self.n_frames = kp_data.shape[0]
#         mjx_model, mjx_data = build_env(kp_data, self._properties)
#         part_names = initialize_part_names(mjx_model, mjx_data)

#         # Get and set the offsets of the markers
#         offsets = jnp.copy(env.physics.bind(env.task._walker.body_sites).pos[:])
#         offsets *= self.SCALE_FACTOR
#         env.physics.bind(env.task._walker.body_sites).pos[:] = offsets
        
#         mjx_data = stac_base.jit_forward(mjx_model, mjx_data)

#         for n_site, p in enumerate(env.physics.bind(env.task._walker.body_sites).pos):
#             env.task._walker.body_sites[n_site].pos = p

#         # Optimize the pose and offsets for the first frame
#         print("Initial optimization")
#         root_optimization(mjx_model, mjx_data, self._properties)

#         for n_iter in range(self.N_ITERS):
#             print(f"Calibration iteration: {n_iter + 1}/{self.N_ITERS}")
#             q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)
#             offset_optimization(mjx_model, mjx_data, offsets, q, self._properties)

#         # Optimize the pose for the whole sequence
#         print("Final pose optimization")
#         q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)
#         self.data = package_data(
#             mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, self._properties
#         )
#         return self

#     def transform(self, kp_data: jnp.ndarray, offset_path: Text) -> Dict:
#         """Register skeleton to keypoint data

#         Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

#         Example:
#             data = stac.transform(keypoints, offset_path)

#         Args:
#             keypoints (jnp.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).
#                 Keypoint order must match the order in the skeleton file.
#             offset_path (Text): Path to offset file saved after .fit()

#         Returns:
#             Dict: Registered data dictionary
#         """
#         kp_data = self._prepare_data(kp_data)
#         self.n_frames = kp_data.shape[0]
#         mjx_model, mjx_data = build_env(kp_data, self._properties)
#         part_names = initialize_part_names(mjx_model, mjx_data)

#         # Set the offsets.
#         self.offset_path = offset_path
#         with open(self.offset_path, "rb") as f:
#             in_dict = pickle.load(f)
#         sites = env.task._walker.body_sites
#         env.physics.bind(sites).pos[:] = in_dict["offsets"]
#         for n_site, p in enumerate(env.physics.bind(sites).pos):
#             sites[n_site].pos = p

#         # TODO: these three function calls need to be vmapped and jitted somehow. 
#         # the batch size will be some factor of the total clips (clips=short chunks), or if possible, the entire set
#         # will need to vectorize kp_data

#         # Optimize the root position
#         root_optimization(mjx_model, mjx_data, self._properties)

#         # Optimize the pose for the whole sequence
#         q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)

#         # Extract pose, offsets, data, and all parameters
#         self.data = package_data(
#             mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, self._properties
#         )
#         return self.data

#     def save(self, save_path: Text):
#         """Save data.

#         Args:
#             save_path (Text): Path to save data. Defaults to None.
#         """
#         if os.path.dirname(save_path) != "":
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         _, file_extension = os.path.splitext(save_path)
#         if file_extension == ".p":
#             with open(save_path, "wb") as output_file:
#                 pickle.dump(self.data, output_file, protocol=2)
#         elif file_extension == ".mat":
#             savemat(save_path, self.data)