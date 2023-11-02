"""Compute stac optimization on data.
"""
from dm_control import viewer
from scipy.io import savemat
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import mjlib
import scipy.ndimage
import clize
import stac.stac_base as stac_base
import stac.rodent_environments as rodent_environments
import numpy as np
import stac.util as util
import pickle
import os
import stac.tasks as tasks
from typing import List, Dict, Tuple, Text, Union

_MM_TO_METERS = 1000


def _downsample(
    kp_data: np.ndarray, params: Dict, orig_freq: float = 300.0
) -> np.ndarray:
    """Helpfer function to downsample data.

    Args:
        kp_data (np.ndarray): n_frames x n_markers x ndimensions keypoint data
        params (Dict): Parameters dictionary.
        orig_freq (float, optional): Original frequency of data.

    No Longer Returned:
        np.ndarray: Resampled keypoint data
    """
    n_samples = kp_data.shape[0]
    n_upsamples = int(np.round(n_samples / orig_freq * (1 / params["_TIME_BINS"])))
    interp_x_vals = np.linspace(0, n_samples, n_upsamples)
    kp_data_downsampled = np.zeros((interp_x_vals.size, kp_data.shape[1]))

    # Linearly interpolate to downsample for each marker
    for marker in range(kp_data.shape[1]):
        kp_data_downsampled[:, marker] = np.interp(
            interp_x_vals, range(n_samples), kp_data[:, marker]
        )
    return kp_data_downsampled


def _smooth(kp_data: np.ndarray, kp_names: List, sigma: float = 1.0) -> np.ndarray:
    """Helper function to smooth data.

    Applies a Gaussian and median filter.

    Args:
        kp_data (np.ndarray): n_frames x n_markers x ndimensions keypoint data
        kp_names (List): Keypoint names
        sigma (float, optional): Sigma of Gaussian smoothing.

    Returns:
        np.ndarray: Smoothed keypoint data
    """
    parts_to_smooth = ["Arm", "Elbow"]
    ids = np.argwhere(
        [any(part in name for name in kp_names) for part in parts_to_smooth]
    )
    for n_marker in ids:
        kp_data[:, n_marker] = scipy.ndimage.gaussian_filter1d(
            kp_data[:, n_marker], sigma, axis=0
        )
        kp_data[:, n_marker] = scipy.signal.medfilt(
            kp_data[:, n_marker], [5, 1], axis=0
        )
    return kp_data


def preprocess_snippet(kp_data: np.ndarray, kp_names: List, params: Dict) -> np.ndarray:
    """Preprocess snippet data.

    Args:
        kp_data (np.ndarray): n_frames x n_markers x ndimensions keypoint data
        kp_names (List): Keypoint names
        params (Dict): Patameters dictionary

    Returns:
        np.ndarray: Preprocessed keypoint data
    """
    kp_data = kp_data / _MM_TO_METERS

    # Handle z-offset conditions
    kp_data[:, 2::3] -= params["z_offset"]
    return kp_data


def preprocess_data(
    data_path: Text,
    start_frame: int,
    end_frame: int,
    skip: int,
    params: Dict,
    struct_name: Text = "markers_preproc",
) -> Tuple[np.ndarray, List]:
    """Preprocess mocap data for stac fitting.

    Args:
        data_path (Text): Path to .mat mocap file
        start_frame (int): Frame to start stac tracking
        end_frame (int): Frame to end stac tracking
        skip (int): Subsampling rate for the frames
        params (Dict): Parameters dictionary
        struct_name (Text, optional): Field name of .mat file to load. DEPRECATED

    Returns:
        Tuple: kp_data (np.ndarray): Keypoint data
               kp_names (List): List of keypoint names
    """
    kp_data, kp_names = util.load_dannce_data(
        data_path,
        params["skeleton_path"],
        start_frame=start_frame,
        end_frame=end_frame,
    )
    kp_data = kp_data[::skip, :]
    kp_data = preprocess_snippet(kp_data, kp_names, params)
    return kp_data, kp_names


def initial_optimization(
    env, initial_offsets: np.ndarray, params: Dict, maxiter: int = 100
):
    """Optimize the first frame with alternating q and m phase.

    Args:
        env (TYPE): Environment
        initial_offsets (np.ndarray): Vector of starting offsets for initial q_phase
        params (Dict): parameter dictionary
        maxiter (int, optional): Maximum number of iterations for m-phase optimization
    """
    if params["verbose"]:
        print("Root Optimization", flush=True)
    root_optimization(env, params)

    # Initial q-phase optimization to get joints into approximate position.
    q, _, _ = q_clip_iso(env, params)

    # Initial m-phase optimization to calibrate offsets
    stac_base.m_phase(
        env.physics,
        env.task.kp_data,
        env.task._walker.body_sites,
        np.arange(params["n_frames"]),
        q,
        initial_offsets,
        params,
        reg_coef=params["m_reg_coef"],
        maxiter=maxiter,
    )


def root_optimization(env, params: Dict, frame: int = 0):
    """Optimize only the root.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary
        frame (int, optional): Frame to optimize
    """
    stac_base.q_phase(
        env.physics,
        env.task.kp_data[frame, :],
        env.task._walker.body_sites,
        params,
        root_only=True,
    )
    # TODO(Generalize): Trunk keypoints should be general for any skeleton. Define in config.
    # First optimize over the trunk
    trunk_kps = [
        any([n in kp_name for n in ["Spine", "Hip", "Shoulder", "Offset"]])
        for kp_name in params["kp_names"]
    ]
    trunk_kps = np.repeat(np.array(trunk_kps), 3)
    stac_base.q_phase(
        env.physics,
        env.task.kp_data[frame, :],
        env.task._walker.body_sites,
        params,
        root_only=True,
        kps_to_opt=trunk_kps,
    )


def q_clip(env, qs_to_opt: np.ndarray, params: Dict) -> Tuple:
    """Q-phase across the clip: optimize joint angles.

    Args:
        env (TYPE): Environment
        qs_to_opt (np.ndarray): boolean array denoting joints to optimize.
        params (Dict): Parameters dictionary

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    q, x, walker_body_sites = [], [], []
    for i in range(params["n_frames"]):
        stac_base.q_phase(
            env.physics,
            env.task.kp_data[i, :],
            env.task._walker.body_sites,
            params,
            reg_coef=params["q_reg_coef"],
        )
        # Make sure to only use forward temporal regularization on frames 1...n
        if i == 0:
            temp_reg = False
            q_prev = 0
        else:
            temp_reg = True
            q_prev = q[i - 1]

        if params["temporal_reg_coef"] > 0.0:
            stac_base.q_phase(
                env.physics,
                env.task.kp_data[i, :],
                env.task._walker.body_sites,
                params,
                reg_coef=params["q_reg_coef"],
                qs_to_opt=qs_to_opt,
                q_prev=q_prev,
                temporal_regularization=temp_reg,
            )
        q.append(np.copy(env.physics.named.data.qpos[:]))
        x.append(np.copy(env.physics.named.data.xpos[:]))

        walker_body_sites.append(
            np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
        )
    return q, walker_body_sites, x


def _get_part_ids(env, parts: List) -> np.ndarray:
    """Get the part ids given a list of parts.

    Args:
        env (TYPE): Environment
        parts (List): List of part names

    Returns:
        np.ndarray: Array of part identifiers
    """
    part_names = env.physics.named.data.qpos.axes.row.names
    return np.array([any(part in name for part in parts) for name in part_names])


def q_clip_iso(env, params: Dict) -> Tuple:
    """Perform q_phase over the entire clip.

    Optimizes limbs and head independently.
    Perform bidirectional temporal regularization.

    Args:
        env (TYPE): Environment
        params (Dict): Parameters dictionary.

    Returns:
        Tuple: qpos, walker body sites, xpos
    """
    q = []
    x = []
    walker_body_sites = []

    r_leg = _get_part_ids(
        env,
        [
            "vertebra_1",
            "vertebra_2",
            "vertebra_3",
            "vertebra_4",
            "vertebra_5",
            "vertebra_6",
            "hip_R",
            "knee_R",
            "ankle_R",
            "foot_R",
        ],
    )
    l_leg = _get_part_ids(
        env,
        [
            "vertebra_1",
            "vertebra_2",
            "vertebra_3",
            "vertebra_4",
            "vertebra_5",
            "vertebra_6",
            "hip_L",
            "knee_L",
            "ankle_L",
            "foot_L",
        ],
    )
    r_arm = _get_part_ids(
        env,
        [
            "scapula_R",
            "shoulder_R",
            "shoulder_s",
            "elbow_R",
            "hand_R",
            "finger_R",
        ],
    )
    l_arm = _get_part_ids(
        env,
        [
            "scapula_L",
            "shoulder_L",
            "shoulder_s",
            "elbow_L",
            "hand_L",
            "finger_L",
        ],
    )
    head = _get_part_ids(env, ["atlas", "cervical", "atlant_extend"])
    if params["LIMBS_TO_TEMPORALLY_REGULARIZE"] == "arms":
        temp_reg_indiv_parts = [r_arm, l_arm]
        indiv_parts = [r_leg, l_leg, head]
    elif params["LIMBS_TO_TEMPORALLY_REGULARIZE"] == "arms and legs":
        temp_reg_indiv_parts = [r_leg, l_leg, r_arm, l_arm]
        indiv_parts = [head]
    else:
        indiv_parts = [head, r_leg, l_leg, r_arm, l_arm]
        temp_reg_indiv_parts = []

    # Iterate through all of the frames in the clip
    for i in range(params["n_frames"]):
        # Optimize over all points
        stac_base.q_phase(
            env.physics,
            env.task.kp_data[i, :],
            env.task._walker.body_sites,
            params,
            reg_coef=params["q_reg_coef"],
        )

        # Make sure to only use forward temporal regularization on frames 1...n
        if i == 0:
            temp_reg = False
            q_prev = 0
        else:
            temp_reg = True
            q_prev = q[i - 1]

        # Next optimize over the limbs individually to improve time and accur.
        for part in indiv_parts:
            stac_base.q_phase(
                env.physics,
                env.task.kp_data[i, :],
                env.task._walker.body_sites,
                params,
                reg_coef=params["q_reg_coef"],
                qs_to_opt=part,
            )
        for part in temp_reg_indiv_parts:
            stac_base.q_phase(
                env.physics,
                env.task.kp_data[i, :],
                env.task._walker.body_sites,
                params,
                reg_coef=params["q_reg_coef"],
                qs_to_opt=part,
                q_prev=q_prev,
                temporal_regularization=temp_reg,
            )
        q.append(np.copy(env.physics.named.data.qpos[:]))
        x.append(np.copy(env.physics.named.data.xpos[:]))
        walker_body_sites.append(
            np.copy(env.physics.bind(env.task._walker.body_sites).xpos[:])
        )

    # Bidirectional temporal regularization
    if params["temporal_reg_coef"] > 0.0:
        for i in range(1, params["n_frames"] - 1):
            # Set model state to current frame
            env.physics.named.data.qpos[:] = q[i]

            # Recompute position of select parts with bidirectional
            # temporal regularizer.
            for part in [r_arm, l_arm, r_leg, l_leg]:
                stac_base.q_phase(
                    env.physics,
                    env.task.kp_data[i, :],
                    env.task._walker.body_sites,
                    params,
                    reg_coef=params["q_reg_coef"],
                    qs_to_opt=part,
                    temporal_regularization=True,
                    q_prev=q[i - 1],
                    q_next=q[i + 1],
                )

                # Update the parts for the current frame
                q[i][part] = np.copy(env.physics.named.data.qpos[:][part])
                x[i] = np.copy(env.physics.named.data.xpos[:])
            walker_body_sites[i] = np.copy(
                env.physics.bind(env.task._walker.body_sites).xpos[:]
            )
    return q, walker_body_sites, x


def qpos_z_offset(env, q: List, x: List) -> Tuple[List, np.ndarray]:
    """Add a z offset to the qpos root equal to the minimum of the hands/ankles.

    Args:
        env (TYPE): Environment
        q (List): List of qposes over a clip.
        x (List): List of xposes over a clip.

    Returns:
        Tuple[List, np.ndarray]: qpos (List): List of altered qposes
            ground_pos (np.ndarray): ground position.
    """
    # Find the indices of hands and feet xpositions
    ground_parts = ["foot", "hand"]
    part_names = env.physics.named.data.xpos.axes.row.names
    ground_ids = [any(part in name for part in ground_parts) for name in part_names]

    # Estimate the ground position by the 2nd percentile of the hands/feet
    ground_part_pos = np.zeros((len(x), np.sum(ground_ids)))

    for i, xpos in enumerate(x):
        ground_part_pos[i, :] = xpos[ground_ids, 2]

    # Set the minimum position over the clip to be the ground_pos
    ground_pos = np.min(np.nanpercentile(ground_part_pos, 0.05, axis=0))
    # ground_pos = ground_pos - .013
    for i, qpos in enumerate(q):
        qpos[2] -= ground_pos
        q[i] = qpos
    return q, ground_pos


def compute_stac(kp_data: np.ndarray, params: Dict):
    """Perform stac on rat mocap data.

    Args:
        kp_data (np.ndarray): n_frames x n_markers x ndimensions keypoint data
        params (Dict): Parameters dictionary
    """
    if params["n_frames"] is None:
        params["n_frames"] = kp_data.shape[0]
    params["n_frames"] = int(params["n_frames"])

    # Build the environment
    env = rodent_environments.rodent_mocap(kp_data, params)
    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["scale_factor"],
        params["scale_factor"],
    )
    mjlib.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
    # Center of mass position
    mjlib.mj_comPos(env.physics.model.ptr, env.physics.data.ptr)
    env.reset()

    # Get the ids of the limbs
    # TODO(partnames): This currently changes the list everywhere.
    # Technically this is what we always want, but consider changing.
    part_names = env.physics.named.data.qpos.axes.row.names
    for i in range(6):
        part_names.insert(0, part_names[0])

    # If preloading offsets, set them now.
    if params["offset_path"] is not None:
        with open(params["offset_path"], "rb") as f:
            in_dict = pickle.load(f)

        sites = env.task._walker.body_sites
        env.physics.bind(sites).pos[:] = in_dict["offsets"]

        for n_site, p in enumerate(env.physics.bind(sites).pos):
            sites[n_site].pos = p

        if params["verbose"]:
            print("Root Optimization", flush=True)
        root_optimization(env, params)
    else:
        # Get the initial offsets of the markers
        initial_offsets = np.copy(env.physics.bind(env.task._walker.body_sites).pos[:])
        initial_offsets *= params["scale_factor"]
        # Set pose to the optimized m and step forward.
        env.physics.bind(env.task._walker.body_sites).pos[:] = initial_offsets
        # Forward kinematics, and save the results to the walker sites as well
        mjlib.mj_kinematics(env.physics.model.ptr, env.physics.data.ptr)
        # Center of mass position
        mjlib.mj_comPos(env.physics.model.ptr, env.physics.data.ptr)
        for n_site, p in enumerate(env.physics.bind(env.task._walker.body_sites).pos):
            env.task._walker.body_sites[n_site].pos = p

        # First optimize the first frame to get an approximation
        # of the m and q phases
        if params["verbose"]:
            print("Initial Optimization", flush=True)
        initial_optimization(env, initial_offsets, params)

        # Find the frames to use in the m-phase optimization.
        time_indices = np.random.randint(
            0, params["n_frames"], params["n_sample_frames"]
        )

    # Q_phase optimization
    if params["verbose"]:
        print("q-phase", flush=True)
    # q, walker_body_sites = q_clip(env, limbs, params)
    q, walker_body_sites, x = q_clip_iso(env, params)

    # If you've precomputed the offsets, stop here.
    # Otherwise do another m and q phase.
    if params["offset_path"] is None:
        # M-phase across the subsampling: optimize offsets
        if params["verbose"]:
            print("m-phase", flush=True)
        stac_base.m_phase(
            env.physics,
            env.task.kp_data,
            env.task._walker.body_sites,
            time_indices,
            q,
            initial_offsets,
            params,
            reg_coef=params["m_reg_coef"],
        )
        if params["verbose"]:
            print("q-phase", flush=True)
        q, walker_body_sites, x = q_clip_iso(env, params)

    # Save the pose, offsets, data, and all parameters
    offsets = env.physics.bind(env.task._walker.body_sites).pos[:].copy()
    names_xpos = env.physics.named.data.xpos.axes.row.names
    out_dict = {
        "qpos": q,
        "xpos": x,
        "walker_body_sites": walker_body_sites,
        "offsets": offsets,
        "names_qpos": part_names,
        "names_xpos": names_xpos,
        "kp_data": np.copy(kp_data[: params["n_frames"], :]),
    }
    for k, v in params.items():
        out_dict[k] = v
    return out_dict


class STAC:
    def __init__(
        self,
        data_path: Text,
        param_path: Text,
        save_path: Text = None,
        offset_path: Text = None,
        start_frame: int = 0,
        end_frame: int = 0,
        n_frames: int = None,
        n_sample_frames: int = 50,
        skip: int = 1,
        verbose: bool = False,
        skeleton_path: Text = "/n/holylfs02/LABS/olveczky_lab/Diego/code/Label3D/skeletons/rat23.mat",
    ):
        """Initialize STAC

        Args:
            data_path (Text): Path to dannce .mat file
            param_path (Text): Path to parameters .yaml file.
            save_path (Text, optional): Path to save data. Defaults to None.
            offset_path (Text, optional): Path to offset .p file. Defaults to None.
            start_frame (int, optional): Starting frame. Defaults to 0.
            end_frame (int, optional): Ending frame. Defaults to 0.
            n_frames (int, optional): Number of frames to evaluate. Defaults to None.
            n_sample_frames (int, optional): Number of frames to evaluate for m-phase estimation. Defaults to 50.
            skip (int, optional): Frame skip. Defaults to 1.
            verbose (bool, optional): If True, print status messages. Defaults to False.
            skeleton_path (Text, optional): Path to skeleton file. Defaults to "/n/holylfs02/LABS/olveczky_lab/Diego/code/Label3D/skeletons/rat23.mat".
        """
        # Aggregate optional cl arguments into params dict
        kw = {
            "data_path": data_path,
            "save_path": save_path,
            "offset_path": offset_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_frames,
            "n_sample_frames": n_sample_frames,
            "verbose": verbose,
            "skip": skip,
            "skeleton_path": skeleton_path,
        }
        self.params = util.load_params(param_path)
        for key, v in kw.items():
            self.params[key] = v

    def fit(self) -> Dict:
        """Calibrate and fit the model to keypoints.

        Performs three rounds of alternating marker and quaternion optimization. Optimal
        results with greater than 200 frames of data in which the subject is moving.

        Example:
            stac = STAC(data_path, param_path, **kwargs)
            offset_data = stac.fit()
            stac.save(offset_data, offset_save_path)

        Returns:
            Dict: Data dictionary
        """
        kp_data = self._prepare_data()
        data = compute_stac(kp_data, self.params)
        return data

    def _prepare_data(self) -> np.ndarray:
        """Preprocess data and keypoint names.

        Returns:
            np.ndarray: Keypoint data (nSamples, nKeypoints*3)
        """
        kp_data, kp_names = preprocess_data(
            self.params["data_path"],
            self.params["start_frame"],
            self.params["end_frame"],
            self.params["skip"],
            self.params,
        )
        self.params["kp_names"] = kp_names
        return kp_data

    def transform(self, offset_path: Text) -> Dict:
        """Register skeleton to keypoint data

        Transform should be used after a skeletal model has been fit to keypoints using the fit() method.

        Example:
            stac = STAC(data_path, param_path, **kwargs)
            offset_data = stac.fit()
            stac.save(offset_data, offset_save_path)
            data = stac.transform(offset_save_path)
            stac.save(data, save_path)

        Args:
            offset_path (Text): Path to offset file saved after .fit()

        Returns:
            Dict: Registered data dictionary
        """
        kp_data = self._prepare_data()
        self.params["offset_path"] = offset_path
        data = compute_stac(kp_data, self.params)
        return data

    def save(self, data: Dict, save_path: Text = None):
        """Save data.

        Args:
            data (Dict): Data dictionary (output of fit() or transform())
            save_path (Text, optional): Path to save data. Defaults to None.
        """
        if save_path is None:
            save_path = self.params["save_path"]
        _, file_extension = os.path.splitext(self.params["save_path"])
        if not os.path.exists(os.path.dirname(self.params["save_path"])):
            os.makedirs(os.path.dirname(self.params["save_path"]), exist_ok=True)
        if file_extension == ".p":
            with open(self.params["save_path"], "wb") as output_file:
                pickle.dump(data, output_file, protocol=2)
        elif file_extension == ".mat":
            savemat(self.params["save_path"], data)


def handle_args(
    data_path: Text,
    param_path: Text,
    *,
    save_path: Text = None,
    offset_path: Text = None,
    start_frame: int = 0,
    end_frame: int = 0,
    n_frames: int = None,
    n_sample_frames: int = 50,
    skip: int = 1,
    adaptive_z_offset: bool = False,
    verbose: bool = False,
    visualize: bool = False,
    render_video: bool = False,
    process_snippet: bool = True,
    skeleton_path: Text = "/n/holylfs02/LABS/olveczky_lab/Diego/code/Label3D/skeletons/rat23.mat",
):
    """Wrap compute_stac to perform appropriate processing.

    :param data_path: List of paths to .mat mocap data files
    :param save_path: File to save optimized qposes
    :param offset_path: Path to precomputed marker offsets
    :param start_frame: Frame within mocap file to start optimization
    :param n_frames: Number of frames to optimize
    :param n_sample_frames: Number of frames to use in m-phase optimization
    :param skip: Subsampling rate for the frames
    :param verbose: If True, display messages during optimization.
    :param visualize: If True, launch viewer
    :param render_video: If True, render_video
    :param process_snippet: If True, process snippet,
                            otherwise assume mocap struct.
    """
    # Aggregate optional cl arguments into params dict
    kw = {
        "data_path": data_path,
        "offset_path": offset_path,
        "start_frame": start_frame,
        "n_frames": n_frames,
        "n_sample_frames": n_sample_frames,
        "verbose": verbose,
        "skip": skip,
        "adaptive_z_offset": adaptive_z_offset,
        "visualize": visualize,
        "render_video": render_video,
        "skeleton_path": skeleton_path,
    }
    params = util.load_params(param_path)
    for key, v in kw.items():
        params[key] = v

    if process_snippet:
        data, kp_names, behavior, com_vel = util.load_snippets_from_file(data_path)
        # Put useful statistics into params dict for now. Consider stats dict
        params["behavior"] = behavior
        params["com_vel"] = com_vel

        kp_data = preprocess_snippet(data, kp_names, params)

        # Save the file as a pickle with the same name as the input file in the
        # folder specified by save_path
        file, ext = os.path.splitext(os.path.basename(data_path))
        save_path = os.path.join(save_path, file + ".p")
        print(save_path, flush=True)
        compute_stac(kp_data, params)
        print("Finished %s" % (save_path))

    # Support file-based processing
    else:
        if end_frame == 0:
            end_frame = start_frame + 3600
        # M = scipy.io.loadmat('test.mat')
        # kp_data = M['kp_data'][:]
        # kp_names = M['kp_names'][:]
        kp_data, kp_names = preprocess_data(
            data_path, start_frame, end_frame, skip, params
        )

        if save_path is None:
            save_path = os.path.join(
                os.getcwd(), "results", "snippet" + data_path[:-4] + ".p"
            )
        params["kp_names"] = kp_names
        # savemat('test.mat', {'kp_data': kp_data, 'kp_names':kp_names})
        compute_stac(kp_data, params)


if __name__ == "__main__":
    clize.run(handle_args)
