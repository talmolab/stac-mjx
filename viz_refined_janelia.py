#!/usr/bin/env python
"""Generate videos from existing STAC IK results in refined_STACed_data_janelia.

Uses custom camera view: transparent mesh, keypoint spheres, centered on skeleton.
"""
import os
from glob import glob
import logging
import traceback

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

import numpy as np
import imageio
import mujoco
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from stac_mjx import io
from stac_mjx.stac import Stac

output_dir = "/mnt/vast/niyo/stac-mjx/refined_STACed_data_janelia"
ik_files = sorted(glob(os.path.join(output_dir, "*_ik.h5")))
logging.info(f"Found {len(ik_files)} IK files to visualize")

for ik_path in ik_files:
    basename = os.path.basename(ik_path).replace("_ik.h5", "")
    mp4_path = os.path.join(output_dir, f"{basename}_stac.mp4")

    try:
        logging.info(f"Rendering {basename}...")

        cfg, d = io.load_stac_data(ik_path)
        qposes = d.qpos
        kp_data = d.kp_data
        kp_names = d.kp_names
        offsets = d.offsets

        xml_path = Path(cfg.model.MJCF_PATH)
        stac = Stac(xml_path, cfg, kp_names)

        # Create render sites (keypoint spheres)
        keypoint_sites = []
        keypoint_site_names = []
        for name in cfg.model.KEYPOINT_MODEL_PAIRS:
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = cfg.model.KEYPOINT_COLOR_PAIRS[name]
            site_name = name + "_kp"
            keypoint_site_names.append(site_name)
            stac._root.worldbody.add(
                "site", name=site_name, type="sphere",
                size=[0.0015],
                rgba=rgba, pos=start, group=2,
            )

        # Add learned offset sites
        for (key, v), pos in zip(
            cfg.model.KEYPOINT_MODEL_PAIRS.items(), offsets.reshape((-1, 3))
        ):
            parent = stac._root.find("body", v)
            parent.add(
                "site", name=key + "_new", type="sphere",
                size=[0.001], rgba="0 0 0 1", pos=pos, group=2,
            )

        physics = mjcf.Physics.from_mjcf_model(stac._root)

        # Build site index map
        axis = physics.named.model.site_pos._axes[0]
        site_index_map = {
            key: int(axis.convert_key_item(key))
            for key in list(cfg.model.KEYPOINT_MODEL_PAIRS.keys()) + keypoint_site_names
        }
        keypoint_site_idxs = [site_index_map[n] for n in keypoint_site_names]

        render_mj_model = deepcopy(physics.model.ptr)

        # Fix sites: dm_control converts spheres to elongated ellipsoids
        for i in range(render_mj_model.nsite):
            render_mj_model.site_type[i] = mujoco.mjtGeom.mjGEOM_SPHERE
            r = render_mj_model.site_size[i, 0]
            render_mj_model.site_size[i] = [r, r, r]

        # Zero out joint visualization scale so they never render
        render_mj_model.vis.scale.jointlength = 0
        render_mj_model.vis.scale.jointwidth = 0

        # Scene options: transparent mesh, keypoint spheres, no joints/actuators
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[1] = 0
        scene_option.geomgroup[2] = 0
        scene_option.sitegroup[0] = 0
        scene_option.sitegroup[1] = 0
        scene_option.sitegroup[2] = 1
        scene_option.sitegroup[3] = 0
        scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = False
        scene_option.flags[enums.mjtVisFlag.mjVIS_ACTUATOR] = False
        scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
        scene_option.flags[enums.mjtVisFlag.mjVIS_LIGHT] = True
        scene_option.flags[enums.mjtVisFlag.mjVIS_CONVEXHULL] = True
        scene_option.flags[enums.mjtRndFlag.mjRND_SHADOW] = True
        scene_option.flags[enums.mjtRndFlag.mjRND_REFLECTION] = True
        scene_option.flags[enums.mjtRndFlag.mjRND_SKYBOX] = True

        mj_data = mujoco.MjData(render_mj_model)
        mujoco.mj_kinematics(render_mj_model, mj_data)

        height, width = 480, 640
        renderer = mujoco.Renderer(render_mj_model, height=height, width=width)

        # Compute centroid for camera
        mj_data.qpos = qposes[0]
        mujoco.mj_fwdPosition(render_mj_model, mj_data)
        body_positions = mj_data.xpos[1:]
        centroid = body_positions.mean(axis=0)

        # Free camera centered on skeleton
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = centroid
        cam.distance = 0.055
        cam.azimuth = 165
        cam.elevation = -25

        n_frames = min(50, qposes.shape[0])
        kp_data_sliced = kp_data[:n_frames]
        qposes_sliced = qposes[:n_frames]

        with imageio.get_writer(mp4_path, fps=30) as video:
            for qpos, kps in tqdm(zip(qposes_sliced, kp_data_sliced), total=n_frames, desc=basename):
                render_mj_model.site_pos[keypoint_site_idxs] = np.reshape(kps, (-1, 3))
                mj_data.qpos = qpos
                mujoco.mj_fwdPosition(render_mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)

        renderer.close()
        logging.info(f"Saved {mp4_path}")

    except Exception as e:
        logging.error(f"Error rendering {basename}: {e}")
        logging.error(traceback.format_exc())

logging.info("Done")
