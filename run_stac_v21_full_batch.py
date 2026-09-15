#!/usr/bin/env python
"""Full-batch v21 STAC run (all 15 CFL_35/36/37 trials, not just the 3-trial
smoke test) using the calibration-refined + gauge-corrected registration,
the fixed q_solver (stepsize=0.0 line search, stac_core.py), and the
original +-10mm sh_tx/ty/tz range (build_v21_xml.py) -- see
run_stac_v21_refined.py (the 3-trial smoke-test version this generalizes)
and CALIBRATION_REFINEMENT_REPORT.md for the full history/justification.

Output: refined_STACed_data_v21/<trial>/<trial>_{fit,ik}.h5 and
<trial>_reproj_v21_refined.mp4 for all 15 trials (adds to what the smoke
test already produced for the 3 SMOKE_PICKS trials -- harmless recompute,
kept for a uniform naming/output scheme across all 15).
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import logging
import traceback
import gc
import json
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

ALL_TRIALS = sorted(
    d.name for d in TRAINING_ROOT.iterdir()
    if d.is_dir() and d.name.startswith(("CFL_35", "CFL_36", "CFL_37"))
)


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
        if near_lo or near_hi:
            flags.append(jname)
    return flags


def marker_error_stats(ik_h5_path):
    import h5py
    with h5py.File(ik_h5_path, "r") as f:
        kp_data = f["kp_data"][:] if "kp_data" in f else None
        marker_sites = f["marker_sites"][:] if "marker_sites" in f else None
    if kp_data is None or marker_sites is None:
        return None
    n_frames = marker_sites.shape[0]
    kp = kp_data.reshape(n_frames, -1, 3)
    err = np.linalg.norm(kp - marker_sites, axis=-1)
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
    logging.info(f"Processing {len(ALL_TRIALS)} trials: {ALL_TRIALS}")

    for trial_name in ALL_TRIALS:
        trial_dir = TRAINING_ROOT / trial_name
        reg_h5 = trial_dir / f"{trial_name}_registered_v21_refined.h5"
        if not reg_h5.exists():
            logging.error(f"[{trial_name}] missing {reg_h5}, skipping")
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
            OmegaConf.update(cfg.stac, "n_frames_per_clip", kp_data.shape[0], merge=True)

            fit_path, ik_path = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=out_dir)
            logging.info(f"[{trial_name}] STAC done: fit={fit_path}, ik={ik_path}")

            flags = check_qpos_ranges(cfg.model.MJCF_PATH, ik_path)
            if flags:
                logging.warning(f"[{trial_name}] {len(flags)} DOF(s) saturated at joint limits: {flags}")

            fit_stats = marker_error_stats(ik_path)
            if fit_stats:
                logging.info(f"[{trial_name}] marker fit error: mean={fit_stats['mean_mm']:.3f}mm "
                             f"median={fit_stats['median_mm']:.3f}mm max={fit_stats['max_mm']:.3f}mm")
            fit_summaries[trial_name] = {"flags": flags, "fit_stats": fit_stats}

            from stac_mjx.reproject_stac_model import render_reprojection_video
            reproj_out = out_dir / f"{trial_name}_reproj_v21_refined.mp4"
            try:
                render_reprojection_video(
                    stac_h5_path=str(ik_path),
                    calibration_path=str(trial_dir / "calibration_refined.toml"),  # see run_stac_v21_refined.py's comment
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

    logging.info("v21 STAC full-batch run done.")
    with open(OUTPUT_ROOT / "full_batch_fit_summary.json", "w") as f:
        json.dump(fit_summaries, f, indent=2)

    ok = {k: v for k, v in fit_summaries.items() if v.get("fit_stats")}
    if ok:
        means = [v["fit_stats"]["mean_mm"] for v in ok.values()]
        logging.info(f"Across {len(ok)} trials: mean marker error {np.mean(means):.3f}mm "
                     f"(range {np.min(means):.3f}-{np.max(means):.3f}mm)")


if __name__ == "__main__":
    main()
