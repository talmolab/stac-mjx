import mediapy as media
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums
import mujoco
from jax import numpy as jnp
from mujoco import mjx
import pickle
import imageio
import numpy as np
import os
import controller as ctrl
import utils
import stac_base

# let user set this 
os.environ["MUJOCO_GL"] = "osmesa"


rat_xml = "../models/rodent_stac.xml"
offset_path = "../standard_fit_6loops.p"
params_path = "../params/params.yaml"

def viz()
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

utils.init_params(params_path)

# load mjx_model and mjx_data and set marker sites

root = mjcf.from_path(rat_xml)
physics, mj_model = ctrl.create_body_sites(root)
physics, mj_model, keypoint_sites = ctrl.create_keypoint_sites(root)

#starting xpos anda xquat for mjdata
_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# mjx_data = stac_base.kinematics(mjx_model, mjx_data)
mj_data = mujoco.MjData(mj_model)
mj_data.xpos = _UPRIGHT_POS
mj_data.xquat = _UPRIGHT_QUAT
mujoco.mj_kinematics(mj_model, mj_data)

# open data file, loop through qpos blah blah
# offset_path="../LM_fit.p"
with open(offset_path, "rb") as file:
    d = pickle.load(file)
    qposes = np.array(d["qpos"])
    kp_data = np.array(d["kp_data"])

FPS=50
n_frames = 2000
frames=[]
save_path="../videos/direct_render.mp4"
renderer = mujoco.Renderer(mj_model, height=1200, width=1920)
# Make sure there are enough frames to render
if qposes.shape[0] < n_frames-1:
    raise Exception(f"Trying to render {n_frames} frames when data['qpos'] only has {qposes.shape[0]}")

# slice kp_data to match qposes length
kp_data = kp_data[:qposes.shape[0]]

# render while stepping using mujoco, not mjx
with imageio.get_writer(save_path, fps=FPS) as video:
    for i, (qpos, kps) in enumerate(zip(qposes, kp_data)):
        if i%100 == 0:
            print(f"rendering frame {i}")
        if i == n_frames:
            break
        
        #set keypoints
        physics, mj_model = ctrl.set_keypoint_sites(physics, keypoint_sites, kps)
        mj_data.qpos = qpos
        mujoco.mj_forward(mj_model, mj_data)

        renderer.update_scene(mj_data, camera="close_profile", scene_option=scene_option)
        pixels = renderer.render()
        video.append_data(pixels)
        frames.append(pixels)

media.show_video(frames, fps=FPS)