"""Render all h5 files from refined_STACed_data_janelia_longer using a custom right-side camera."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stac_mjx.io import load_stac_data
from stac_mjx.stac import Stac
import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from copy import deepcopy
from dm_control import mjcf
import imageio
from tqdm import tqdm

INPUT_DIR = Path("refined_STACed_data_janelia_longer")
OUTPUT_DIR = Path("refined_STACed_data_janelia_longer_right_side")
BASE_PATH = Path(__file__).parent
HEIGHT = 480
WIDTH = 640

OUTPUT_DIR.mkdir(exist_ok=True)


CAMERA_NAME = "right_side_shifted_v3"


def render_h5(h5_path, save_path):
    cfg, d = load_stac_data(h5_path)
    qposes = d.qpos
    kp_data = d.kp_data
    offsets = d.offsets
    kp_names = d.kp_names
    n_frames = qposes.shape[0]

    xml_path = BASE_PATH / cfg.model.MJCF_PATH
    stac = Stac(xml_path, cfg, kp_names)

    # Create render sites (keypoints + body sites)
    render_mj_model, body_site_idxs, keypoint_site_idxs = stac._create_render_sites()

    # Add body sites for learned offsets
    for (key, v), pos in zip(
        cfg.model.KEYPOINT_MODEL_PAIRS.items(), offsets.reshape((-1, 3))
    ):
        parent = stac._root.find("body", v)
        parent.add(
            "site", name=key + "_new", type="sphere",
            size=[0.001], rgba="0 0 0 1", pos=pos, group=2,
        )

    # Add the custom camera to the mjcf model before compiling
    stac._root.worldbody.add(
        "camera",
        name=CAMERA_NAME,
        pos=[0.025, -0.05, 0.065],
        xyaxes=[1, 0, 0, 0, 0, 1],
        fovy=35,
    )

    physics = mjcf.Physics.from_mjcf_model(stac._root)
    render_mj_model = deepcopy(physics.model.ptr)

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    scene_option.sitegroup[2] = 1
    scene_option.sitegroup[3] = 0
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = True
    scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = True
    scene_option.flags[enums.mjtRndFlag.mjRND_FOG] = True

    mj_data = mujoco.MjData(render_mj_model)
    mujoco.mj_kinematics(render_mj_model, mj_data)
    renderer = mujoco.Renderer(render_mj_model, height=HEIGHT, width=WIDTH)

    kp_data = kp_data[: qposes.shape[0]]

    frames = []
    with imageio.get_writer(save_path, fps=cfg.model.RENDER_FPS) as video:
        for qpos, kps in tqdm(zip(qposes, kp_data)):
            render_mj_model.site_pos[keypoint_site_idxs] = np.reshape(kps, (-1, 3))
            mj_data.qpos = qpos
            mujoco.mj_fwdPosition(render_mj_model, mj_data)
            renderer.update_scene(mj_data, camera=CAMERA_NAME, scene_option=scene_option)
            pixels = renderer.render()
            video.append_data(pixels)
            frames.append(pixels)

    return frames


h5_files = sorted(INPUT_DIR.glob("*.h5"))
print(f"Found {len(h5_files)} h5 files")

for h5_path in h5_files:
    mp4_name = h5_path.stem + ".mp4"
    save_path = OUTPUT_DIR / mp4_name
    print(f"Rendering {h5_path.name} -> {save_path}")
    try:
        frames = render_h5(h5_path, save_path)
        print(f"  Done: {len(frames)} frames")
    except Exception as e:
        print(f"  ERROR: {e}")
