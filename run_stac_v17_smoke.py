"""v17 STAC smoke test: run on 1 reach per animal × 5 sessions, render reproject videos.

Output:
  refined_STACed_data_v17/<animal>/<session>_trial<NNN>_fit.h5
  refined_STACed_data_v17/<animal>/<session>_trial<NNN>_ik.h5
  refined_STACed_data_v17/<animal>/<session>_trial<NNN>_reproj_v17.mp4
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import logging, traceback, gc, h5py
import numpy as np
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import stac_mjx
import jax

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"JAX devices: {jax.devices()}")

DATA_ROOT = Path("/mnt/vast/eric/azim-emg-data")
SMOKE_PICKS = {
    # animal: (session_dir_name, clip_start_frame, n_frames)
    # A36-1 clip 25781: BEST d2+d5 validity for v17 supination test
    #   (67% both valid, 86% d2_mid, 81% d5_mid; sidx=25739, apex_in_session=25831)
    "A36-1": ("A36-1_2023-07-18_16-54-01_lightOff_video", 25781, 100),
}


def extract_clip(reg_file, start_frame, n_frames):
    with h5py.File(reg_file, "r") as f:
        T = f["tracks"].shape[0]
        if start_frame < 0 or start_frame + n_frames > T:
            raise ValueError(f"clip [{start_frame}, {start_frame+n_frames}) out of bounds for T={T}")
        clip = f["tracks"][start_frame:start_frame + n_frames]   # (n, 1, 5, 3)
    assert clip.shape == (n_frames, 1, 5, 3), f"unexpected clip shape {clip.shape}"
    return clip


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm_hand_v17", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()
    base_path = Path("/mnt/vast/niyo/stac-mjx").resolve()
    output_root = base_path / "refined_STACed_data_v17"
    output_root.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "model") or isinstance(cfg.model, str):
        model_config = OmegaConf.load(str(base_path / "configs/model/mouse_arm_hand_v17.yaml"))
        cfg.model = model_config
    OmegaConf.set_struct(cfg.model, False)

    for animal, (session, clip_start, n_frames) in SMOKE_PICKS.items():
        out_dir = output_root / animal
        out_dir.mkdir(parents=True, exist_ok=True)
        sd = DATA_ROOT / session
        reg_file = sd / f"{session}_mujoco_registered_forelimb_right_hand_v1.h5"
        if not reg_file.exists():
            logging.error(f"[{animal}] missing splice file: {reg_file}")
            continue

        suffix = f"{session.replace('_video','')}_trial{clip_start}"
        try:
            logging.info(f"\n{'='*70}\n[{animal}] clip [{clip_start},{clip_start + n_frames})  {reg_file.name}\n{'='*70}")
            clip = extract_clip(reg_file, clip_start, n_frames)
            clip_path = out_dir / f"{suffix}_clip.h5"
            with h5py.File(clip_path, "w") as h:
                h.create_dataset("tracks", data=clip)

            OmegaConf.set_struct(cfg.stac, False)
            cfg.stac.data_path = str(clip_path)
            cfg.stac.fit_offsets_path = f"{suffix}_fit.h5"
            cfg.stac.ik_only_path = f"{suffix}_ik.h5"

            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
            logging.info(f"[{animal}] kp_data shape={kp_data.shape}, names={sorted_kp_names}")
            OmegaConf.update(cfg.stac, "n_frames_per_clip", kp_data.shape[0], merge=True)

            fit_path, ik_path = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=out_dir)
            logging.info(f"[{animal}] STAC done: fit={fit_path}, ik={ik_path}")

            # Render reprojection video
            from stac_mjx.reproject_stac_model import render_reprojection_video
            calib = sd / "calibration.toml"
            video_dir = sd
            sid_short = session.replace("_video", "")
            env_pts = base_path / "smoke_test_v16_reproj" / f"{sid_short}_env_points3d.h5"
            raw_pts = sd / f"{session}_points3d.h5"
            reproj_out = out_dir / f"{suffix}_reproj_v17.mp4"
            try:
                render_reprojection_video(
                    stac_h5_path=str(ik_path),
                    calibration_path=str(calib),
                    video_dir=str(video_dir),
                    env_points3d_path=str(env_pts),
                    registered_h5_path=str(reg_file),
                    raw_points3d_path=str(raw_pts),
                    clip_csv_path=None,
                    trial_number=None,
                    output_path=str(reproj_out),
                    fps=30,
                )
                logging.info(f"[{animal}] reproj saved: {reproj_out}")
            except Exception as e:
                logging.warning(f"[{animal}] reproj failed: {e}")
                logging.warning(traceback.format_exc())
        except Exception as e:
            logging.error(f"[{animal}] STAC error: {e}")
            logging.error(traceback.format_exc())
        gc.collect()

    logging.info("v17 smoke test done.")


if __name__ == "__main__":
    main()
