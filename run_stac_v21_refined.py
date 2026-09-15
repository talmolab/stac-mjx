#!/usr/bin/env python
"""Rerun of the v21 STAC smoke test using calibration-refined data:
  - registered h5 built from aniposelib bundle-adjustment-refined
    triangulation (*_points3d_refined.h5 -> *_registered_v21_refined.h5)
  - reprojection rendered using the original calibration.toml (NOT
    calibration_refined.toml -- see the render call's comment below for why)
  - v21 model rebuilt with the calibration-refined joystick_base position
  - q_solver uses stepsize=0.0 (line search) instead of a fixed step, and
    sh_tx/ty/tz are back at their original +-10mm range (see stac_core.py
    and build_v21_xml.py for why)

Same 3 trials as the original Phase E smoke test (run_stac_v21_smoke.py):
CFL_35/36/37, trial_0001 each. Output overwrites refined_STACed_data_v21/
(the same directory Eric already reviewed) since this is the rerun he asked
for ("rerun this refined_STACed_data_v21 with the improved calibration").
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ.setdefault('MUJOCO_GL', 'egl')

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

SMOKE_PICKS = sorted(d.name for d in TRAINING_ROOT.iterdir()
                      if d.is_dir() and d.name.startswith(("CFL_35", "CFL_36", "CFL_37")))
# All 15 trials (not just 1 per session): checking consistency WITHIN a
# session (same animal, multiple trials) matters just as much as across
# sessions -- a per-trial scale/rotation fit could in principle be
# consistent cross-session by luck while still varying trial-to-trial
# within one animal's own recording session.

# For fast iteration: set N_FRAMES_LIMIT env var to e.g. 15 (~0.6s at 25fps)
# to only fit/render a short clip instead of the full 126-frame trial. Unset
# (None) runs the full trial as before.
N_FRAMES_LIMIT = int(os.environ["N_FRAMES_LIMIT"]) if "N_FRAMES_LIMIT" in os.environ else None


def check_qpos_ranges(model_path, ik_h5_path):
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
        lo, hi = m.jnt_range[jid]
        is_hinge = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        if lo == 0 and hi == 0:
            continue
        vals = qpos[:, qadr]
        if is_hinge:
            vals = np.degrees(vals)
            lo, hi = np.degrees(lo), np.degrees(hi)
        vmin, vmax = vals.min(), vals.max()
        span = hi - lo
        near_lo = (vmin - lo) < 0.02 * span
        near_hi = (hi - vmax) < 0.02 * span
        marker = " <-- SATURATED" if (near_lo or near_hi) else ""
        print(f"  {jname:32s} fitted=[{vmin:8.4f}, {vmax:8.4f}]  jnt_range=[{lo:8.4f}, {hi:8.4f}]{marker}")
        if near_lo or near_hi:
            flags.append(jname)
    return flags


def marker_error_stats(ik_h5_path):
    """Mean per-keypoint marker error (mm) between mocap input and FK-computed
    site positions, for a quick before/after fit-quality comparison."""
    import h5py
    with h5py.File(ik_h5_path, "r") as f:
        kp_data = f["kp_data"][:] if "kp_data" in f else None
        marker_sites = f["marker_sites"][:] if "marker_sites" in f else None
    if kp_data is None or marker_sites is None:
        return None
    n_frames = marker_sites.shape[0]
    kp = kp_data.reshape(n_frames, -1, 3)
    err = np.linalg.norm(kp - marker_sites, axis=-1)  # (n_frames, n_kp)
    return {"mean_mm": float(np.mean(err) * 1000), "median_mm": float(np.median(err) * 1000),
            "max_mm": float(np.max(err) * 1000)}


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm_hand_joystick_v21", version_base=None)
def main(cfg):
    stac_mjx.enable_xla_flags()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "model") or isinstance(cfg.model, str):
        model_config = OmegaConf.load(str(BASE_PATH / "configs/model/mouse_arm_hand_joystick_v21.yaml"))
        cfg.model = model_config
    OmegaConf.set_struct(cfg.model, False)

    fit_summaries = {}

    for trial_name in SMOKE_PICKS:
        trial_dir = TRAINING_ROOT / trial_name
        reg_h5 = trial_dir / f"{trial_name}_registered_v21_refined.h5"
        if not reg_h5.exists():
            logging.error(f"[{trial_name}] missing {reg_h5}, run register_v21_mocap.py --points3d-suffix _points3d_refined.h5 first")
            continue

        out_dir = OUTPUT_ROOT / trial_name
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info(f"\n{'='*70}\n[{trial_name}] (calibration-refined rerun)\n{'='*70}")

            OmegaConf.set_struct(cfg.stac, False)
            cfg.stac.data_path = str(reg_h5)
            cfg.stac.fit_offsets_path = f"{trial_name}_fit.h5"
            cfg.stac.ik_only_path = f"{trial_name}_ik.h5"

            kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, BASE_PATH)
            if N_FRAMES_LIMIT is not None:
                kp_data = kp_data[:N_FRAMES_LIMIT]
                logging.info(f"[{trial_name}] N_FRAMES_LIMIT={N_FRAMES_LIMIT}: truncated kp_data to shape={kp_data.shape}")
            logging.info(f"[{trial_name}] kp_data shape={kp_data.shape}, names={sorted_kp_names}")
            OmegaConf.update(cfg.stac, "n_frames_per_clip", kp_data.shape[0], merge=True)

            fit_path, ik_path = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=out_dir)
            logging.info(f"[{trial_name}] STAC done: fit={fit_path}, ik={ik_path}")

            print(f"\n[{trial_name}] qpos range check (jnt_range saturation heuristic):")
            flags = check_qpos_ranges(cfg.model.MJCF_PATH, ik_path)
            if flags:
                logging.warning(f"[{trial_name}] {len(flags)} DOF(s) saturated at joint limits: {flags}")

            fit_stats = marker_error_stats(ik_path)
            if fit_stats:
                print(f"[{trial_name}] marker fit error: mean={fit_stats['mean_mm']:.3f}mm "
                      f"median={fit_stats['median_mm']:.3f}mm max={fit_stats['max_mm']:.3f}mm")
            fit_summaries[trial_name] = {"flags": flags, "fit_stats": fit_stats}

            from stac_mjx.reproject_stac_model import render_reprojection_video
            reproj_out = out_dir / f"{trial_name}_reproj_v21_refined.mp4"
            try:
                render_reprojection_video(
                    stac_h5_path=str(ik_path),
                    # calibration_refined.toml, NOT the original calibration.toml:
                    # registered_v21_refined.h5's registration_rotation/translation
                    # (multicam.py's RegistrationTransform) map model-frame points
                    # back into whatever frame points3d_refined.h5 was triangulated
                    # in when it was fed to register_v21_mocap.py -- currently
                    # calibration_refined.toml's own (self-consistent, uncorrected)
                    # frame, per the revert documented in CALIBRATION_REFINEMENT_REPORT.md
                    # section 3 addendum. An earlier version of this script used the
                    # original calibration.toml here on the theory that a Umeyama
                    # gauge-correction had re-expressed the points in the original
                    # frame -- that correction was tried and reverted (made results
                    # visibly worse), so this must revert to match.
                    calibration_path=str(trial_dir / "calibration_refined.toml"),
                    video_dir=str(trial_dir),
                    env_points3d_path=None,
                    registered_h5_path=str(reg_h5),
                    raw_points3d_path=str(trial_dir / f"{trial_name}_points3d_refined.h5"),
                    clip_csv_path=None,
                    trial_number=None,
                    output_path=str(reproj_out),
                    fps=25,
                    render_size=(540, 720),
                )
                logging.info(f"[{trial_name}] reproj saved: {reproj_out}")
            except Exception as e:
                logging.warning(f"[{trial_name}] reproj failed: {e}")
                logging.warning(traceback.format_exc())

        except Exception as e:
            logging.error(f"[{trial_name}] STAC error: {e}")
            logging.error(traceback.format_exc())
        gc.collect()

    logging.info("v21 STAC calibration-refined rerun done.")
    import json
    with open(OUTPUT_ROOT / "refined_rerun_fit_summary.json", "w") as f:
        json.dump(fit_summaries, f, indent=2)


if __name__ == "__main__":
    main()
