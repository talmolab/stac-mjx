"""A collection mujoco-mjx vizualization utilities."""

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import enums
import mujoco
from jax import numpy as jnp
import pickle
import imageio
import numpy as np
from typing import List, Dict, Text
import os
from dm_control.mujoco import wrapper
import cv2
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

import utils

# Gotta do this before importing controller
utils.init_params("../params/params.yaml")
import controller as ctrl
import stac_base

# Standard image shape for dannce rig data
# TODO: make this a param
HEIGHT = 1200
WIDTH = 1920

# Param for overlay_frame()
ALPHA_BASE_VALUE = 0.5


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


def overlay_frame(
    rgb_frame: np.ndarray,
    params: List,
    recon_frame: np.ndarray,
    seg_frame: np.ndarray,
    camera: Text,
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
    # TODO: id 1 for camera 2; change to param later
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
    # (No longer needed for mujoco > 3.0.0)
    # recon_frame = correct_optical_center(params, recon_frame, cam_id)
    # seg_frame = correct_optical_center(params, seg_frame, cam_id, pad_val=-1)

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


def convert_camera_indiv(cam, id):
    """Convert a camera from Matlab convention to Mujoco convention."""
    # Matlab camera X faces the opposite direction of Mujoco X
    rot = R.from_matrix(cam["TDistort"])
    eul = rot.as_euler("zyx")
    eul[2] += np.pi
    modified_rot = R.from_euler("zyx", eul)
    quat = modified_rot.as_quat()

    # Convert the quaternion convention from scipy.spatial.transform.Rotation to Mujoco.
    quat = quat[np.array([3, 0, 1, 2])]
    quat[0] *= -1
    # The y field of fiew is a function of the focal y and the image height.
    fovy = 2 * np.arctan(HEIGHT / (2 * cam["K"][1, 1])) / (2 * np.pi) * 360
    return {
        "name": f"Camera{id}",
        "pos": -cam["t"] @ cam["TDistort"] / 1000,
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


def overlay_viz(
    data_path,
    calibration_path,
    video_path,
    model_xml,
    n_frames,
    save_path,
    camera: Text = "close_profile",
):
    """Overlay 3D mocap forward kinematics for the model on top the original video.

    Uses camera parameters from dannce mocap recording setup and aligns the video
    of the recording with the rendering of the mujoco forward kinematics and
    overlays them.
    """
    scene_option = wrapper.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 1

    scene_option.sitegroup[3] = 1
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_FOG] = False

    # Load mjx_model and mjx_data and set marker sites
    root = mjcf.from_path(model_xml)
    rescale.rescale_subtree(
        root,
        utils.params["SCALE_FACTOR"],
        utils.params["SCALE_FACTOR"],
    )
    # Add cameras
    cam_params = utils.loadmat(calibration_path)["params"]
    # print(cam_params.keys())
    # print(cam_params)
    # root.worldbody.add("camera", **convert_camera_indiv(cam_params, 2)) # Camera 2
    camera_kwargs = convert_cameras(cam_params)
    for kwargs in camera_kwargs:
        root.worldbody.add("camera", **kwargs)

    physics, mj_model = ctrl.create_body_sites(root)
    physics, mj_model, keypoint_sites = ctrl.create_keypoint_sites(root)
    physics.forward()
    # Load data
    with open(data_path, "rb") as file:
        d = pickle.load(file)
        qposes = np.array(d["qpos"])
        kp_data = np.array(d["kp_data"])

    renderer = mujoco.Renderer(mj_model, height=1200, width=1920)

    kp_data = kp_data[: qposes.shape[0]]

    prev_time = physics.time()
    reader = imageio.get_reader(video_path)

    frames = []
    with imageio.get_writer(save_path, fps=utils.params["RENDER_FPS"]) as video:
        for i, (qpos, kps) in enumerate(zip(qposes, kp_data)):
            if i % 100 == 0:
                print(f"rendering frame {i}")

            # TODO: cut qposes and kp_data into the right shape beforehand
            if i == n_frames:
                break
            # This only happens with the simulation fps and render fps don't match up
            # commenting out for now since I'll need to implement time tracking outside of dmcontrol
            if i > 0:
                while (np.round(physics.time() - prev_time, decimals=5)) < utils.params[
                    "TIME_BINS"
                ]:
                    # mujoco.mj_forward(mj_model, mj_data)
                    physics.step()
            # Set keypoints
            physics, mj_model = ctrl.set_keypoint_sites(physics, keypoint_sites, kps)
            # mj_data.qpos = qpos
            # mujoco.mj_forward(mj_model, mj_data)
            physics.data.qpos = qpos
            physics.step()

            # SEGMENT or IDCOLOR?
            # https://github.com/google-deepmind/mujoco/blob/725630c95ddebceb32b89c45cfc14a5eae7f8a8a/include/mujoco/mjvisualize.h#L149
            # scene_option.flags[enums.mjtRndFlag.mjRND_SEGMENT] = False
            # renderer.update_scene(physics.data, camera=camera, scene_option=scene_option)
            # reconArr = renderer.render()

            # # Get segmentation rendering
            # scene_option.flags[enums.mjtRndFlag.mjRND_SEGMENT] = True
            # renderer.update_scene(physics.data, camera=camera, scene_option=scene_option)
            # segArr = renderer.render()

            reconArr = physics.render(
                HEIGHT,
                WIDTH,
                camera_id=camera,
                scene_option=scene_option,
            )

            segArr = physics.render(
                HEIGHT,
                WIDTH,
                camera_id=camera,
                scene_option=scene_option,
                segmentation=True,
            )

            rgbArr = reader.get_data(i)
            frame = overlay_frame(rgbArr, cam_params, reconArr, segArr, camera)

            video.append_data(frame)
            frames.append(frame)
            prev_time = np.round(physics.time(), decimals=2)

    return frames


def mujoco_viz(data_path, model_xml, n_frames, save_path, start_frame: int = 0):
    """Render forward kinematics from keypoint positions."""
    scene_option = mujoco.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 1

    scene_option.sitegroup[3] = 1
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_FOG] = False

    # Load mjx_model and mjx_data and set marker sites
    root = mjcf.from_path(model_xml)
    physics, mj_model = ctrl.create_body_sites(root)
    physics, mj_model, keypoint_sites = ctrl.create_keypoint_sites(root)

    rescale.rescale_subtree(
        root,
        utils.params["SCALE_FACTOR"],
        utils.params["SCALE_FACTOR"],
    )

    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_kinematics(mj_model, mj_data)

    # Load data
    with open(data_path, "rb") as file:
        d = pickle.load(file)
        qposes = np.array(d["qpos"])
        kp_data = np.array(d["kp_data"])

    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)

    # Make sure there are enough frames to render
    if qposes.shape[0] < n_frames - 1:
        raise Exception(
            f"Trying to render {n_frames} frames when data['qpos'] only has {qposes.shape[0]}"
        )

    # slice kp_data to match qposes length
    kp_data = kp_data[: qposes.shape[0]]

    # Slice arrays to be the range that is being rendered
    kp_data = kp_data[start_frame : start_frame + n_frames]
    qposes = qposes[start_frame : start_frame + n_frames]

    frames = []
    # render while stepping using mujoco
    with imageio.get_writer(save_path, fps=utils.params["RENDER_FPS"]) as video:
        for i, (qpos, kps) in enumerate(zip(qposes, kp_data)):
            if i % 100 == 0:
                print(f"rendering frame {i}")

            # Set keypoints
            physics, mj_model = ctrl.set_keypoint_sites(physics, keypoint_sites, kps)
            mj_data.qpos = qpos
            mujoco.mj_forward(mj_model, mj_data)

            renderer.update_scene(
                mj_data, camera="close_profile", scene_option=scene_option
            )
            pixels = renderer.render()
            video.append_data(pixels)
            frames.append(pixels)

    return frames


# Render two rats in the same sim! This can be programmatically extended to any number of rats
# No keypoints or body site rendering unfortunately since they are explicitly named
def mujoco_pair_viz(
    data_path1,
    data_path2,
    model_xml,
    n_frames,
    save_path,
    start_frame1: int = 0,
    start_frame2: int = 0,
):
    """Render two models in the same simulation."""
    scene_option = mujoco.MjvOption()
    # scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    # scene_option.geomgroup[3] = 0
    # scene_option.sitegroup[0] = 0
    # scene_option.sitegroup[1] = 0
    scene_option.sitegroup[2] = 1

    scene_option.sitegroup[3] = 1
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = False
    scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = False
    scene_option.flags[enums.mjtRndFlag.mjRND_FOG] = False

    # Load mjx_model and mjx_data and set marker sites
    # root = mjcf.from_path(model_xml)
    # physics = mjcf.Physics.from_mjcf_model(root)
    mj_model = mujoco.MjModel.from_xml_path(model_xml)
    # physics, mj_model = ctrl.create_body_sites(root)
    # physics, mj_model, keypoint_sites = ctrl.create_keypoint_sites(root)

    # rescale.rescale_subtree(
    #     root,
    #     utils.params["SCALE_FACTOR"],
    #     utils.params["SCALE_FACTOR"],
    # )

    # Starting xpos and xquat for mjdata
    _UPRIGHT_POS = (0.0, 0.0, 0.94)
    _UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

    mj_data = mujoco.MjData(mj_model)
    # mj_data.xpos = _UPRIGHT_POS
    # mj_data.xquat = _UPRIGHT_QUAT
    mujoco.mj_kinematics(mj_model, mj_data)

    # Load data
    with open(data_path1, "rb") as file:
        d1 = pickle.load(file)
        qposes1 = np.array(d1["qpos"])
        kp_data1 = np.array(d1["kp_data"])

    with open(data_path2, "rb") as file:
        d2 = pickle.load(file)
        qposes2 = np.array(d2["qpos"])
        kp_data2 = np.array(d2["kp_data"])

    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)

    # Make sure there are enough frames to render
    if qposes1.shape[0] < n_frames - 1:
        raise Exception(
            f"Trying to render {n_frames} frames when data['qpos'] only has {qposes1.shape[0]}"
        )

    # slice kp_data to match qposes length
    kp_data1 = kp_data1[: qposes1.shape[0]]
    kp_data2 = kp_data2[: qposes2.shape[0]]
    # Slice arrays to be the range that is being rendered
    kp_data1 = kp_data1[start_frame1 : start_frame1 + n_frames]
    qposes1 = qposes1[start_frame1 : start_frame1 + n_frames]

    kp_data2 = kp_data2[start_frame2 : start_frame2 + n_frames]
    qposes2 = qposes2[start_frame2 : start_frame2 + n_frames]

    frames = []
    # render while stepping using mujoco
    with imageio.get_writer(save_path, fps=utils.params["RENDER_FPS"]) as video:
        for i, (qpos1, qpos2) in enumerate(zip(qposes1, qposes2)):
            if i % 100 == 0:
                print(f"rendering frame {i}")

            # Set keypoints
            # physics, mj_model = ctrl.set_keypoint_sites(physics, keypoint_sites, kps)
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)

            renderer.update_scene(
                mj_data, camera="close_profile", scene_option=scene_option
            )
            pixels = renderer.render()
            video.append_data(pixels)
            frames.append(pixels)

    return frames
