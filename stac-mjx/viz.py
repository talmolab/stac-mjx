import cv2
import imageio
import numpy as np
import pickle
import re
import scipy.io as spio
import stac.rodent_environments as rodent_environments
import stac.util as util
import yaml
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
from typing import Text, List, Dict
import os

ALPHA_BASE_VALUE = 0.5
FPS = 50

# Standard image shape for dannce rig data
HEIGHT = 1200
WIDTH = 1920
MODELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def fix_tail(q: np.ndarray, q_names: List):
    """Fix the tail so that it is in the base position.

    Args:
        q (np.ndarray): qpos
        q_names (List): names of each dimension of qpos

    Returns:
        [type]: [description]
    """
    for i, name in enumerate(q_names):
        if re.match("walker/vertebra_C.*bend", name):
            q[:, i] = 0.0
        if re.match("walker/vertebra_C.*extend", name):
            q[:, i] = 0.0
    return q


def setup_visualization(
    param_path: str,
    q,
    offsets,
    kp_data,
    n_frames: int,
    render_video: bool = False,
    segmented=False,
    camera_kwargs: List = None,
    registration_xml: bool = False,
):
    """Sets up stac visualization.

    Args:
        param_path (str): Path to stac parameters file.
        q (TYPE): Array of qpos for each frames.
        offsets (TYPE): Site offsets
        kp_data (TYPE): Description
        n_frames (TYPE): Number of frames
        render_video (bool, optional): If True, make a video and put it in clips.
    """
    params = util.load_params(param_path)
    params["n_frames"] = n_frames - 1

    if segmented:
        alpha = 0.0
        params["ARENA_DIAMETER"] = None
    else:
        alpha = 1.0
        params["ARENA_DIAMETER"] = None
    if registration_xml:
        params["XML_PATH"] = os.path.join(MODELS_PATH, "rodent_stac.xml")

    env = setup_arena(kp_data, params, alpha=alpha)
    rescale.rescale_subtree(
        env.task._walker._mjcf_root,
        params["SCALE_FACTOR"],
        params["SCALE_FACTOR"],
    )

    if camera_kwargs is not None:
        for kwargs in camera_kwargs:
            env.task._arena._mjcf_root.worldbody.add("camera", **kwargs)

    env.reset()
    setup_sites(q, offsets, env)
    env.task.render_video = render_video
    env.task.initialize_episode(env.physics, 0)
    scene_option = setup_scene()
    return params, env, scene_option


def setup_scene():
    """Return scene_option for hires scene.

    Returns:
        [type]: MjvOption
    """
    scene_option = wrapper.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 0
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_FOG] = False
    return scene_option


def setup_sites(q: np.ndarray, offsets: np.ndarray, env):
    """Setup the walker sites.

    Args:
        q (np.ndarray): qpos
        offsets (np.ndarray): model stac offsets
        env ([type]): environment
    """
    sites = env.task._walker.body_sites
    env.physics.bind(sites).pos[:] = offsets
    for n_offset, site in enumerate(sites):
        site.pos = offsets[n_offset, :]
    env.task.qpos = q


def overlay_frame(
    rgb_frame: np.ndarray,
    params: List,
    recon_frame: np.ndarray,
    seg_frame: np.ndarray,
    camera: int,
) -> np.ndarray:
    """Overlay the reconstructed frame on top of the rgb frame.

    Args:
        rgb_frame (np.ndarray): Frame from the rgb video.
        params (List): Camera parameters.
        recon_frame (np.ndarray): Reconstructed frame.
        seg_frame (np.ndarray): Segmented frame.
        camera (int): Camera name.

    Returns:
        np.ndarray: Overlayed frame.
    """
    cam_id = int(camera[-1]) - 1
    # Load and undistort the rgb frame
    rgb_frame = cv2.undistort(
        rgb_frame,
        params[cam_id].K.T,
        np.concatenate(
            [params[cam_id].RDistort, params[cam_id].TDistort], axis=0
        ).T.squeeze(),
        params[cam_id].K.T,
    )

    # Calculate the alpha mask using the segmented video
    alpha = (seg_frame[:, :, 0] >= 0.0) * ALPHA_BASE_VALUE
    alpha = gaussian_filter(alpha, 2)
    alpha = gaussian_filter(alpha, 2)
    alpha = gaussian_filter(alpha, 2)
    frame = np.zeros_like(recon_frame)

    # Correct the segmented frame by cropping such that the optical center is at the center of the image
    recon_frame = correct_optical_center(params, recon_frame, cam_id)
    seg_frame = correct_optical_center(params, seg_frame, cam_id, pad_val=-1)

    # Calculate the alpha mask using the segmented video
    alpha = (seg_frame[:, :, 0] >= 0.0) * ALPHA_BASE_VALUE
    alpha = gaussian_filter(alpha, 2)
    alpha = gaussian_filter(alpha, 2)
    alpha = gaussian_filter(alpha, 2)
    frame = np.zeros_like(recon_frame)

    # Blend the two videos
    for n_chan in range(recon_frame.shape[2]):
        frame[:, :, n_chan] = (
            alpha * recon_frame[:, :, n_chan] + (1 - alpha) * rgb_frame[:, :, n_chan]
        )
    return frame


def correct_optical_center(
    params, frame: np.ndarray, cam_id: int, pad_val=0
) -> np.ndarray:
    """Correct the optical center of the frame.

    Args:
        params (_type_): Matlab camera parameters
        frame (np.ndarray): frame to correct
        cam_id (int): camera id
        pad_val (int, optional): Pad value. Defaults to 0.

    Returns:
        np.ndarray: Corrected frame
    """
    # Get the optical center
    cx = params[cam_id].K[2, 0]
    cy = params[cam_id].K[2, 1]

    # Compute the offset and pad the frame
    crop_offset_x = int(-cx + (frame.shape[1] / 2))
    crop_offset_y = int(-cy + (frame.shape[0] / 2))
    padding = np.max(np.abs([crop_offset_x, crop_offset_y])) + 10
    padded_frame = np.pad(
        frame,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="constant",
        constant_values=pad_val,
    )
    crop_offset_x += padding
    crop_offset_y += padding

    # Crop the frame
    frame = padded_frame[
        crop_offset_y : crop_offset_y + frame.shape[0],
        crop_offset_x : crop_offset_x + frame.shape[1],
    ]
    return frame


def mujoco_loop(
    save_path: Text,
    params: Dict,
    env,
    scene_option,
    camera: Text = "walker/close_profile",
    height: int = 1200,
    width: int = 1920,
):
    """Rendering loop for generating mujoco videos.

    Args:
        save_path (Text): Path to save overlay video
        params (Dict): stac parameters dictionary
        env ([type]): rodent environment
        scene_option ([type]): MjvOption rendering options
        camera (Text, optional): Name of the camera to use for rendering. Defaults to "walker/close_profile".
        height (int, optional): Camera height in pixels. Defaults to 1200.
        width (int, optional): Camera width in pixels. Defaults to 1920.
    """

    def render_frame(env, scene_option, height, width, camera):
        env.task.after_step(env.physics, None)
        return env.physics.render(
            height,
            width,
            camera_id=camera,
            scene_option=scene_option,
        )

    prev_time = env.physics.time()
    with imageio.get_writer(save_path, fps=FPS) as video:
        # Render the first frame before stepping in the physics
        reconArr = render_frame(env, scene_option, height, width, camera)
        video.append_data(reconArr)
        while prev_time < env._time_limit:
            while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
                "TIME_BINS"
            ]:
                env.physics.step()
            reconArr = render_frame(env, scene_option, height, width, camera)
            video.append_data(reconArr)
            prev_time = np.round(env.physics.time(), decimals=2)


def overlay_loop(
    save_path: Text,
    video_path: Text,
    calibration_path: Text,
    frames: np.ndarray,
    params: Dict,
    env,
    scene_option,
    camera: Text = "walker/close_profile",
    height: int = 1200,
    width: int = 1920,
):
    """Rendering loop for generating overlay videos.

    Args:
        save_path (Text): Path to save overlay video
        video_path (Text): Path to undistorted rgb video
        calibration_path (Text): Path to calibration dannce.mat file.
        frames (np.ndarray): Frames to render
        params (Dict): stac parameters dictionary
        env ([type]): rodent environment
        scene_option ([type]): MjvOption rendering options
        camera (Text, optional): Name of the camera to use for rendering. Defaults to "walker/close_profile".
        height (int, optional): Camera height in pixels. Defaults to 1200.
        width (int, optional): Camera width in pixels. Defaults to 1920.
    """

    def render_frame(
        frames, env, scene_option, camera, height, width, reader, n_frame, cam_params
    ):
        env.task.after_step(env.physics, None)
        reconArr = env.physics.render(
            height,
            width,
            camera_id=camera,
            scene_option=scene_option,
        )
        segArr = env.physics.render(
            height,
            width,
            camera_id=camera,
            scene_option=scene_option,
            segmentation=True,
        )
        rgbArr = reader.get_data(frames[n_frame])
        frame = overlay_frame(rgbArr, cam_params, reconArr, segArr, camera)
        return frame

    prev_time = env.physics.time()
    reader = imageio.get_reader(video_path)
    n_frame = 0
    cam_params = util.loadmat(calibration_path)["params"]
    env.task.after_step(env.physics, None)
    with imageio.get_writer(save_path, fps=FPS) as video:
        for n_frame in range(len(frames)):
            if n_frame > 0:
                while (np.round(env.physics.time() - prev_time, decimals=5)) < params[
                    "TIME_BINS"
                ]:
                    env.physics.step()
            frame = render_frame(
                frames,
                env,
                scene_option,
                camera,
                height,
                width,
                reader,
                n_frame,
                cam_params,
            )
            video.append_data(frame)
            prev_time = np.round(env.physics.time(), decimals=2)


def setup_arena(kp_data, params, alpha=1.0):
    # Build the environment, and set the offsets, and params
    if params["ARENA_TYPE"] == "Standard":
        env = rodent_environments.rodent_mocap(kp_data, params)
    elif params["ARENA_TYPE"] == "DannceArena":
        if "ARENA_DIAMETER" in params:
            diameter = params["ARENA_DIAMETER"]
        else:
            diameter = None

        if "ARENA_CENTER" in params:
            center = params["ARENA_CENTER"]
        else:
            center = None
        env = rodent_environments.rodent_mocap(
            kp_data,
            params,
            arena_diameter=diameter,
            arena_center=center,
            alpha=alpha,
        )
    return env


def convert_camera(cam, idx):
    """Convert a camera from Matlab convention to Mujoco convention."""
    # Matlab camera X faces the opposite direction of Mujoco X
    rot = R.from_matrix(cam.r.T)
    eul = rot.as_euler("zyx")
    eul[2] += np.pi
    modified_rot = R.from_euler("zyx", eul)
    quat = modified_rot.as_quat()

    # Convert the quaternion convention from scipy.spatial.transform.Rotation to Mujoco.
    quat = quat[np.array([3, 0, 1, 2])]
    quat[0] *= -1
    # The y field of fiew is a function of the focal y and the image height.
    fovy = 2 * np.arctan(HEIGHT / (2 * cam.K[1, 1])) / (2 * np.pi) * 360
    return {
        "name": f"Camera{idx + 1}",
        "pos": -cam.t @ cam.r.T / 1000,
        "fovy": fovy,
        "quat": quat,
    }


def convert_cameras(params) -> List[Dict]:
    """Convert cameras from Matlab convention to Mujoco convention.

    Args:
        params: Camera parameters structure

    Returns:
        List[Dict]: List of dicts containing kwargs for Mujoco camera addition through worldbody.
    """
    camera_kwargs = [convert_camera(cam, idx) for idx, cam in enumerate(params)]
    return camera_kwargs


def render_mujoco(
    param_path: Text,
    data_path: Text,
    save_path: Text,
    frames: np.ndarray = None,
    camera: Text = "walker/close_profile",
    calibration_path: Text = None,
    segmented: bool = False,
    height: int = HEIGHT,
    width: int = WIDTH,
    xml_path: Text = None,
):
    """Render a video of a STAC dataset using mujoco.

    Args:
        param_path (Text): Path to the stac parameters file.
        data_path (Text): Path to the stac data file.
        save_path (Text): Path to save the video.
        frames (np.ndarray, optional): Frames to render. Defaults to None.
        camera (Text, optional): Camera to render. Defaults to "walker/close_profile".
        calibration_path (Text, optional): Path to the calibration file. Defaults to None.
        segmented (bool, optional): Whether to segment the rendered video. Defaults to False.
        height (int, optional): Image height. Defaults to 1200.
        width (int, optional): Image width. Defaults to 1920.
        xml_path (Text, optional): Path to the xml file. Defaults to None.
    """
    xml_path, qpos, kp_data, n_frames, offsets, camera_kwargs = load_data(
        param_path, data_path, frames, calibration_path, xml_path
    )

    # Prepare the environment
    params, env, scene_option = setup_visualization(
        param_path,
        qpos,
        offsets,
        kp_data,
        n_frames,
        render_video=True,
        segmented=segmented,
        camera_kwargs=camera_kwargs,
        registration_xml=xml_path,
    )
    mujoco_loop(
        save_path,
        params,
        env,
        scene_option,
        camera=camera,
        height=height,
        width=width,
    )


def load_data(
    param_path: str,
    data_path: str,
    frames: np.ndarray,
    calibration_path: str,
    xml_path: str,
):
    # Load XML path from parameters file if not provided
    if xml_path is None:
        with open(param_path, "rb") as file:
            params = yaml.safe_load(file)
        xml_path = params["XML_PATH"]

    # Load data from the data file
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    # Process qpos data
    qpos = np.stack(data["qpos"][:], axis=0)
    q_names = data["names_qpos"][:]
    qpos = fix_tail(qpos, q_names)

    # Filter qpos and kp_data based on provided frames
    if frames is not None:
        qpos = qpos[frames, ...].copy()
        kp_data = data["kp_data"][frames, ...].copy()
    n_frames = qpos.shape[0]
    offsets = data["offsets"]

    # Load camera parameters and convert them if a calibration path is provided
    if calibration_path is not None:
        params = util.loadmat(calibration_path)["params"]
        camera_kwargs = convert_cameras(params)
    else:
        camera_kwargs = None
    return xml_path, qpos, kp_data, n_frames, offsets, camera_kwargs


def render_overlay(
    param_path: str,
    video_path: str,
    data_path: str,
    save_path: str,
    frames=None,
    camera="Camera1",
    calibration_path=None,
    segmented=False,
    height=HEIGHT,
    width=WIDTH,
    xml_path=None,
):
    xml_path, qpos, kp_data, n_frames, offsets, camera_kwargs = load_data(
        param_path, data_path, frames, calibration_path, xml_path
    )

    # Prepare the environment
    params, env, scene_option = setup_visualization(
        param_path,
        qpos,
        offsets,
        kp_data,
        n_frames,
        render_video=True,
        segmented=segmented,
        camera_kwargs=camera_kwargs,
        registration_xml=xml_path,
    )
    overlay_loop(
        save_path=save_path,
        video_path=video_path,
        calibration_path=calibration_path,
        frames=frames,
        params=params,
        env=env,
        scene_option=scene_option,
        camera=camera,
        height=height,
        width=width,
    )