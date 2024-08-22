"""A collection mujoco-mjx vizualization utilities."""

from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
from dm_control.mujoco.wrapper.mjbindings import enums
import mujoco
import pickle
import imageio
import numpy as np

from stac_mjx import utils
from stac_mjx import controller as ctrl

# Standard image shape for dannce rig data
# TODO: make this a param
HEIGHT = 1200
WIDTH = 1920

# Param for overlay_frame()
ALPHA_BASE_VALUE = 0.5

def export_viz(model_xml, params):
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    scene_option.geomgroup[3] = 1
    scene_option.sitegroup[0] = 0
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
    #physics, mj_model = ctrl.create_body_sites(root)
    kps = utils.load_data("../tests/data/points3d_00_scaleSmooth_1_rpt_15.h5", params)[1, :]
    physics, mj_model = ctrl.create_body_sites(root)
    physics, mj_model, keypoint_sites = ctrl.create_keypoint_sites_centroid(root, kps)
    physics, mj_model = ctrl.create_tendons(root)
    

    #physics, mj_model, keypoint_sites = ctrl.create_keypoint_sites(root)
    #physics, mj_model = ctrl.set_keypoint_sites_centroid(physics, keypoint_sites, kps)

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    return mj_data, mj_model


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
    print("kp_data shape = ", kp_data.shape)
    print("qp shape = ", qposes.shape)
    kp_data = kp_data[: qposes.shape[0]]
    print("kp_data reshape = ", kp_data.shape)

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
                mj_data, camera="com", scene_option=scene_option
            )
            pixels = renderer.render()
            video.append_data(pixels)
            frames.append(pixels)

    return frames
