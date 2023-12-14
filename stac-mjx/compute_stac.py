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
import numpy as jnp
import pickle
import os
from typing import List, Dict, Tuple, Text
from tqdm import tqdm
import state
import numpy as np
import functools

def root_optimization(mjx_model, mjx_data, kp_data, site_index_map, params: Dict, frame: int = 0):
    """Optimize only the root.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary
        frame (int, optional): Frame to optimize
    """
    
    print("Root Optimization:")
    
    mjx_data = stac_base.q_phase(
        mjx_model, 
        mjx_data,
        kp_data[frame, :],
        site_index_map,
        params,
        root_only=True,
    )

    # First optimize over the trunk
    trunk_kps = [
        any([n in kp_name for n in params["TRUNK_OPTIMIZATION_KEYPOINTS"]])
        for kp_name in params["kp_names"]
    ]
    trunk_kps = jnp.repeat(jnp.array(trunk_kps), 3)
    mjx_data = stac_base.q_phase(
        mjx_model, 
        mjx_data,
        kp_data[frame, :],
        site_index_map,
        params,
        root_only=True,
        kps_to_opt=trunk_kps,
    )
    
    return mjx_data

def offset_optimization(mjx_model, mjx_data, kp_data, offsets, q, site_index_map: Dict, params: Dict, maxiter: int = 100):
    key = jax.random.PRNGKey(0)
    time_indices = jax.random.randint(
        key, shape=[params["N_SAMPLE_FRAMES"]], minval=0, maxval=params["n_frames"], 
    )
    
    print("Offset Optimization:")

    mjx_model, mjx_data = stac_base.m_phase(
        mjx_model, 
        mjx_data,
        kp_data,
        site_index_map,
        time_indices,
        q,
        offsets,
        params,
        reg_coef=params["M_REG_COEF"],
        maxiter=maxiter,
    )

# TODO: 
def get_part_ids(physics, parts: List) -> jnp.ndarray:
    """Get the part ids given a list of parts.
This code creates a JAX NumPy-like Boolean array where each element 
represents whether any of the strings in the parts list is found as a substring in 
the corresponding name from the part_names list.
    Args:
        env (TYPE): Environment
        parts (List): List of part names

    Returns:
        jnp.ndarray: Array of part identifiers
    """
    part_names = physics.named.data.qpos.axes.row.names
    return np.array([any(part in name for part in parts) for name in part_names])

def pose_optimization(mjx_model, mjx_data, kp_data, site_index_map, params: Dict) -> Tuple:
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary.

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    # q = jnp.array([])
    # x = jnp.array([])
    # walker_body_sites = jnp.array([])

    # TODO: move out
    # Use global indiv parts to scan over range and index in if can't scan ragged arrays
    if params["INDIVIDUAL_PART_OPTIMIZATION"] is None:
        indiv_parts = []
    else:
        indiv_parts = [
            get_part_ids(mjx_model, mjx_data, parts)
            for parts in params["INDIVIDUAL_PART_OPTIMIZATION"].values()
        ]
    
    # Iterate through all of the frames in the clip
    frames = jnp.arange(params["n_frames"])
    
    print("Pose Optimization:")

    def f(carry, n_frame):
        # unpack carry
        mjx_data, q, x, walker_body_sites = carry
        
        # Optimize over all points
        mjx_data = stac_base.q_phase(
            mjx_model, 
            mjx_data,
            kp_data[n_frame, :],
            site_index_map,
            params,
        )

        # TODO: scan this too
        # Next optimize over parts individually to improve time and accur.
        for part in indiv_parts:
            mjx_data = stac_base.q_phase(
                mjx_model, 
                mjx_data,
                kp_data[n_frame, :],
                site_index_map,
                params,
                qs_to_opt=part,
            )
        q = q.append(jnp.copy(mjx_data.qpos[:]))
        x = x.append(jnp.copy(mjx_data.xpos[:]))
        walker_body_sites = walker_body_sites.append(
            jnp.copy(stac_base.get_site_xpos(mjx_data, site_index_map))
        )
    
    carry = (mjx_data, jnp.empty(0), jnp.empty(0), jnp.empty(0))
    (mjx_data, q, x, walker_body_sites), _ = jax.lax.scan(f, carry, frames)
    
    # for n_frame in frames:
    #     # Optimize over all points
    #     mjx_data = stac_base.q_phase(
    #         mjx_model, 
    #         mjx_data,
    #         kp_data[n_frame, :],
    #         site_index_map,
    #         params,
    #     )

    #     # Next optimize over parts individually to improve time and accur.
    #     for part in indiv_parts:
    #         mjx_data = stac_base.q_phase(
    #             mjx_model, 
    #             mjx_data,
    #             kp_data[n_frame, :],
    #             site_index_map,
    #             params,
    #             qs_to_opt=part,
    #         )
    #     q = q.append(jnp.copy(mjx_data.qpos[:]))
    #     x = x.append(jnp.copy(mjx_data.xpos[:]))
    #     walker_body_sites = walker_body_sites.append(
    #         jnp.copy(stac_base.get_site_xpos(mjx_data, site_index_map))
    #     )

    return q, walker_body_sites, x

# TODO: delete?
def build_env(kp_data: jnp.ndarray, params: Dict):
    """loads mjmodel and makes mjdata, also does rescaling.

    Args:
        kp_data (jnp.ndarray): Key point data.
        params (Dict): Parameters for the environment.

    Returns:
        : The environment
    """
    model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
    mjx_model = mjx.device_put(model)
    mjx_data = mjx.make_data(mjx_model)

    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["SCALE_FACTOR"],
        params["SCALE_FACTOR"],
    )
    
    stac_base.jit_forward(mjx_model, mjx_data)

    return mjx_model, mjx_data

def initialize_part_names(physics):
    # Get the ids of the limbs, accounting for quaternion and pos
    part_names = physics.named.data.qpos.axes.row.names
    for _ in range(6):
        part_names.insert(0, part_names[0])
    return part_names

# TODO: Do we need to package the data like this? DOes it need to be jax compatible?
def package_data(mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, site_index_map, params):
    # Extract pose, offsets, data, and all parameters
    offsets = stac_base.get_site_pos(mjx_model, site_index_map).copy()
    
    names_xpos = mjx_data.xpos.axes.row.names
    x = x.reshape(-1, x.shape[-1])
    q = q.reshape(-1, q.shape[-1])
    kp_data = kp_data.reshape(-1, kp_data.shape[-1])
    data = {
        "qpos": q,
        "xpos": x,
        "walker_body_sites": walker_body_sites,
        "offsets": offsets,
        "names_qpos": part_names,
        "names_xpos": names_xpos,
        "kp_data": jnp.copy(kp_data),
    }
    
    for k, v in params.items():
        data[k] = v
    
    return data


class STAC:
    def __init__(
        self,
        param_path: Text,
    ):
        """Initialize STAC

        Args:
            param_path (Text): Path to parameters .yaml file.
        """
        self._properties = util.load_params(param_path)
        self._properties["data"] = None
        self._properties["n_frames"] = None

        # Default ordering of mj sites is alphabetical, so we reorder to match
        self._properties["kp_names"] = util.loadmat(self._properties["SKELETON_PATH"])["joint_names"]
        # argsort returns the indices that would sort the array
        self._properties["stac_keypoint_order"] = jnp.argsort(
            self._properties["kp_names"]
        )
        for property_name in self._properties.keys():

            def getter(self, name=property_name):
                return self._properties[name]

            def setter(self, value, name=property_name):
                self._properties[name] = value

            setattr(STAC, property_name, property(fget=getter, fset=setter))

    def _prepare_data(self, kp_data: jnp.ndarray) -> jnp.ndarray:
        """Prepare the data for STAC.

        Args:
            kp_data (jnp.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).

        Returns:
            jnp.ndarray: Keypoint data in meters (n_frames, n_keypoints * 3).
        """
        kp_data = kp_data[:, :, self.stac_keypoint_order]
        kp_data = jnp.transpose(kp_data, (0, 2, 1))
        kp_data = jnp.reshape(kp_data, (kp_data.shape[0], -1))
        return kp_data

    def fit(self, kp_data: jnp.ndarray) -> "STAC":
        """Calibrate and fit the model to keypoints.

        Performs three rounds of alternating marker and quaternion optimization. Optimal
        results with greater than 200 frames of data in which the subject is moving.

        Args:
            keypoints (jnp.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.

        Example:
            st = st.fit(keypoints)

        Returns: STAC object with fitted model.
        """
        kp_data = self._prepare_data(kp_data)
        self.n_frames = kp_data.shape[0]
        mjx_model, mjx_data = build_env(kp_data, self._properties)
        part_names = initialize_part_names(mjx_model, mjx_data)

        # Get and set the offsets of the markers
        offsets = jnp.copy(env.physics.bind(env.task._walker.body_sites).pos[:])
        offsets *= self.SCALE_FACTOR
        env.physics.bind(env.task._walker.body_sites).pos[:] = offsets
        
        mjx_data = stac_base.jit_forward(mjx_model, mjx_data)

        for n_site, p in enumerate(env.physics.bind(env.task._walker.body_sites).pos):
            env.task._walker.body_sites[n_site].pos = p

        # Optimize the pose and offsets for the first frame
        print("Initial optimization")
        root_optimization(mjx_model, mjx_data, self._properties)

        for n_iter in range(self.N_ITERS):
            print(f"Calibration iteration: {n_iter + 1}/{self.N_ITERS}")
            q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)
            offset_optimization(mjx_model, mjx_data, offsets, q, self._properties)

        # Optimize the pose for the whole sequence
        print("Final pose optimization")
        q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)
        self.data = package_data(
            mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, self._properties
        )
        return self

    def transform(self, kp_data: jnp.ndarray, offset_path: Text) -> Dict:
        """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Example:
            data = stac.transform(keypoints, offset_path)

        Args:
            keypoints (jnp.ndarray): Keypoint data in meters (n_frames, 3, n_keypoints).
                Keypoint order must match the order in the skeleton file.
            offset_path (Text): Path to offset file saved after .fit()

        Returns:
            Dict: Registered data dictionary
        """
        kp_data = self._prepare_data(kp_data)
        self.n_frames = kp_data.shape[0]
        mjx_model, mjx_data = build_env(kp_data, self._properties)
        part_names = initialize_part_names(mjx_model, mjx_data)

        # Set the offsets.
        self.offset_path = offset_path
        with open(self.offset_path, "rb") as f:
            in_dict = pickle.load(f)
        sites = env.task._walker.body_sites
        env.physics.bind(sites).pos[:] = in_dict["offsets"]
        for n_site, p in enumerate(env.physics.bind(sites).pos):
            sites[n_site].pos = p

        # TODO: these three function calls need to be vmapped and jitted somehow. 
        # the batch size will be some factor of the total clips (clips=short chunks), or if possible, the entire set
        # will need to vectorize kp_data

        # Optimize the root position
        root_optimization(mjx_model, mjx_data, self._properties)

        # Optimize the pose for the whole sequence
        q, walker_body_sites, x = pose_optimization(mjx_model, mjx_data, self._properties)

        # Extract pose, offsets, data, and all parameters
        self.data = package_data(
            mjx_model, mjx_data, q, x, walker_body_sites, part_names, kp_data, self._properties
        )
        return self.data

    def save(self, save_path: Text):
        """Save data.

        Args:
            save_path (Text): Path to save data. Defaults to None.
        """
        if os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _, file_extension = os.path.splitext(save_path)
        if file_extension == ".p":
            with open(save_path, "wb") as output_file:
                pickle.dump(self.data, output_file, protocol=2)
        elif file_extension == ".mat":
            savemat(save_path, self.data)