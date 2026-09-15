#!/usr/bin/env python
"""v21 STAC smoke test: run on 1 trial per CFL_35/36/37 session (matching the
Phase D alignment-check picks), render reprojection videos.

Unlike the older run_stac*.py scripts, the js_luc3d_hand CFL trials are
already fixed-length (126-frame) standalone clips (register_v21_mocap.py's
output), so there's no clip-extraction-from-a-longer-session step needed --
each registered h5 is fed to STAC directly.

Output:
  refined_STACed_data_v21/<trial>/<trial>_fit.h5
  refined_STACed_data_v21/<trial>/<trial>_ik.h5
  refined_STACed_data_v21/<trial>/<trial>_reproj_v21.mp4
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# osmesa (software rendering), not egl -- this sandbox has no working GPU/EGL
# driver (confirmed earlier this session); libosmesa6/libosmesa6-dev were
# installed via apt to make osmesa work.
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import logging
import traceback
import gc
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import jax
import mujoco
import numpy as np
import stac_mjx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"JAX devices: {jax.devices()}")

TRAINING_ROOT = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData")
BASE_PATH = Path("/mnt/vast/niyo/stac-mjx").resolve()
OUTPUT_ROOT = BASE_PATH / "refined_STACed_data_v21"

SMOKE_PICKS = [
    "CFL_35_20240128_trial_0001",
    "CFL_36_20231220_trial_0001",
    "CFL_37_20240423_trial_0001",
]


def check_qpos_ranges(model_path, ik_h5_path):
    """Per STAC_EMG_DIAGNOSTIC_REPORT.md's heuristic: flag any fitted DOF
    whose empirical range sits suspiciously at its jnt_range limit (possible
    saturation) rather than silently accepting the fit."""
    import h5py
    m = mujoco.MjModel.from_xml_path(str(model_path))
    with h5py.File(ik_h5_path, "r") as f:
        qpos = f["qpos"][:]
        names_qpos = [n.decode() if isinstance(n, bytes) else n for n in f["names_qpos"][:]]

    flags = []
    for jname in names_qpos:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            flags.append(f"  {jname}: NOT FOUND in model (name mismatch!)")
            continue
        qadr = m.jnt_qposadr[jid]
        lo, hi = m.jnt_range[jid]  # MuJoCo always stores jnt_range in RADIANS
        # internally, regardless of the XML's compiler angle="degree" setting
        # (that attribute only affects how XML text attributes are PARSED, not
        # the compiled model's internal units) -- convert both sides to
        # degrees consistently for hinge joints so the comparison is apples-
        # to-apples (comparing raw qpos radians against a printed range in
        # degrees was the bug in an earlier version of this check).
        is_hinge = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        if lo == 0 and hi == 0:
            continue  # unlimited joint, no range to check against
        vals = qpos[:, qadr]
        if is_hinge:
            vals = np.degrees(vals)
            lo, hi = np.degrees(lo), np.degrees(hi)
        vmin, vmax = vals.min(), vals.max()
        span = hi - lo
        near_lo = (vmin - lo) < 0.02 * span
        near_hi = (hi - vmax) < 0.02 * span
        marker = " <-- SATURATED" if (near_lo or near_hi) else ""
        print(f"  {jname:32s} fitted=[{vmin:8.2f}, {vmax:8.2f}]  jnt_range=[{lo:8.2f}, {hi:8.2f}]{marker}")
        if near_lo or near_hi:
            flags.append(jname)
    return flags


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm_hand_joystick_v21", version_base=None)
def main(cfg):
    stac_mjx.enable_xla_flags()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "model") or isinstance(cfg.model, str):
        model_config = OmegaConf.load(str(BASE_PATH / "configs/model/mouse_arm_hand_joystick_v21.yaml"))
        cfg.model = model_config
    OmegaConf.set_struct(cfg.model, False)

    for trial_name in SMOKE_PICKS:
        trial_dir = TRAINING_ROOT / trial_name
        reg_h5 = trial_dir / f"{trial_name}_registered_v21.h5"
        if not reg_h5.exists():
            logging.error(f"[{trial_name}] missing {reg_h5}, run register_v21_mocap.py first")
            continue

        out_dir = OUTPUT_ROOT / trial_name
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info(f"\n{'='*70}\n[{trial_name}]\n{'='*70}")

            OmegaConf.set_struct(cfg.stac, False)
            cfg.stac.data_path = str(reg_h5)
            cfg.stac.fit_offsets_path = f"{trial_name}_fit.h5"
            cfg.stac.ik_only_path = f"{trial_name}_ik.h5"

            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, BASE_PATH)
            logging.info(f"[{trial_name}] kp_data shape={kp_data.shape}, names={sorted_kp_names}")
            OmegaConf.update(cfg.stac, "n_frames_per_clip", kp_data.shape[0], merge=True)

            fit_path, ik_path = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=out_dir)
            logging.info(f"[{trial_name}] STAC done: fit={fit_path}, ik={ik_path}")

            print(f"\n[{trial_name}] qpos range check (jnt_range saturation heuristic):")
            flags = check_qpos_ranges(cfg.model.MJCF_PATH, ik_path)
            if flags:
                logging.warning(f"[{trial_name}] {len(flags)} DOF(s) saturated at joint limits: {flags}")

            from stac_mjx.reproject_stac_model import render_reprojection_video
            reproj_out = out_dir / f"{trial_name}_reproj_v21.mp4"
            try:
                render_reprojection_video(
                    stac_h5_path=str(ik_path),
                    calibration_path=str(trial_dir / "calibration.toml"),
                    video_dir=str(trial_dir),
                    env_points3d_path=None,   # no rig-landmark tracking in this dataset
                    registered_h5_path=str(reg_h5),
                    raw_points3d_path=str(trial_dir / f"{trial_name}_points3d.h5"),
                    clip_csv_path=None,        # each trial IS the clip, no CSV lookup needed
                    trial_number=None,
                    output_path=str(reproj_out),
                    fps=25,                    # matches the CFL videos' actual fps
                    render_size=(540, 720),    # matches the CFL videos' actual resolution
                )
                logging.info(f"[{trial_name}] reproj saved: {reproj_out}")
            except Exception as e:
                logging.warning(f"[{trial_name}] reproj failed: {e}")
                logging.warning(traceback.format_exc())

        except Exception as e:
            logging.error(f"[{trial_name}] STAC error: {e}")
            logging.error(traceback.format_exc())
        gc.collect()

    logging.info("v21 STAC smoke test done.")


if __name__ == "__main__":
    main()
