"""v17 full batch: run STAC on every valid reach across all 5 sessions.

Uses the canonical refined CSV (_reach_csv_refined/<animal>_reach_metrics.csv)
to pick reach windows. Each clip is 100 frames centered on the reach apex.

Output:
  refined_STACed_data_v17_full/<animal>/<session>_trial<NNN>_fit.h5
  refined_STACed_data_v17_full/<animal>/<session>_trial<NNN>_ik.h5
  refined_STACed_data_v17_full/<animal>/<session>_trial<NNN>_reproj_v17.mp4
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

DATA_ROOT = Path("/mnt/vast/eric/azim-emg-data")
CSV_DIR = DATA_ROOT / "_reach_csv_refined"

SESSIONS = {
    "A36-1": "A36-1_2023-07-18_16-54-01_lightOff_video",
    "AT006": "AT006_2024-02-28_15-39-40_LightOff_video",
    "AT009": "AT009_2024-04-25_13-50-34_LightOff_video",
    "AT012": "AT012_2024-06-21_12-32-30_LightOff_video",
    "AT013": "AT013_2024-06-20_13-41-20_LightOff_video",
}

N_FRAMES = 100  # clip length, frames @ 200fps
PAD_BEFORE_APEX = 50  # frames before reach_apex


def load_reaches(animal):
    csv = CSV_DIR / f"{animal}_reach_metrics.csv"
    df = pd.read_csv(csv).dropna(subset=["reach_apexPos"]).reset_index(drop=True)
    df["clip_start"] = (df["start_idx"].astype(int) + df["reach_apexPos"].astype(int) - PAD_BEFORE_APEX)
    df["clip_end"] = df["clip_start"] + N_FRAMES
    return df


def extract_clip(reg_file, start_frame, n_frames):
    with h5py.File(reg_file, "r") as f:
        T = f["tracks"].shape[0]
        if start_frame < 0 or start_frame + n_frames > T:
            raise ValueError(f"clip out of bounds: [{start_frame}, {start_frame+n_frames}) vs T={T}")
        clip = f["tracks"][start_frame:start_frame + n_frames]
    return clip


@hydra.main(config_path="configs", config_name="stac/stac_mouse_arm_hand_v17", version_base=None)
def main(cfg: DictConfig):
    stac_mjx.enable_xla_flags()
    base_path = Path("/mnt/vast/niyo/stac-mjx").resolve()
    output_root = base_path / "refined_STACed_data_v17_full"
    output_root.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "model") or isinstance(cfg.model, str):
        model_cfg = OmegaConf.load(str(base_path / "configs/model/mouse_arm_hand_v17.yaml"))
        cfg.model = model_cfg
    OmegaConf.set_struct(cfg.model, False)

    total_done, total_failed = 0, 0
    for animal, session in SESSIONS.items():
        out_dir = output_root / animal
        out_dir.mkdir(parents=True, exist_ok=True)
        sd = DATA_ROOT / session
        reg_file = sd / f"{session}_mujoco_registered_forelimb_right_hand_v1.h5"
        sid_short = session.replace("_video", "")
        env_pts = base_path / "smoke_test_v16_reproj" / f"{sid_short}_env_points3d.h5"
        raw_pts = sd / f"{session}_points3d.h5"
        calib = sd / "calibration.toml"
        if not all(p.exists() for p in [reg_file, env_pts, raw_pts, calib]):
            logging.error(f"[{animal}] missing input file(s); skipping session")
            continue

        df = load_reaches(animal)
        logging.info(f"\n===== [{animal}] {len(df)} reaches in CSV =====")

        # Build a tiny per-session trial CSV for the reproject (trial,trial_start,start,end format)
        trial_csv_path = out_dir / "_trials_v17.csv"
        trial_rows = []
        for i, row in df.iterrows():
            trial_rows.append({
                "trial": int(i + 1),
                "trial_start": int(row["clip_start"]),
                "start": 0,
                "end": int(N_FRAMES),
            })
        pd.DataFrame(trial_rows).to_csv(trial_csv_path, index=False)

        for i, row in df.iterrows():
            clip_start = int(row["clip_start"])
            trial_id = i + 1
            suffix = f"{sid_short}_trial{trial_id:03d}"
            ik_path = out_dir / f"{suffix}_ik.h5"
            reproj_out = out_dir / f"{suffix}_reproj_v17.mp4"

            if ik_path.exists() and reproj_out.exists():
                logging.info(f"[{animal} t{trial_id:03d}] already done, skip")
                continue

            try:
                clip = extract_clip(reg_file, clip_start, N_FRAMES)
                clip_path = out_dir / f"{suffix}_clip.h5"
                with h5py.File(clip_path, "w") as h:
                    h.create_dataset("tracks", data=clip)

                OmegaConf.set_struct(cfg.stac, False)
                cfg.stac.data_path = str(clip_path)
                cfg.stac.fit_offsets_path = f"{suffix}_fit.h5"
                cfg.stac.ik_only_path = f"{suffix}_ik.h5"

                if not ik_path.exists():
                    kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)
                    OmegaConf.update(cfg.stac, "n_frames_per_clip", kp_data.shape[0], merge=True)
                    fit_path, ik_p = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=out_dir)
                    logging.info(f"[{animal} t{trial_id:03d}] STAC done")

                if not reproj_out.exists():
                    from stac_mjx.reproject_stac_model import render_reprojection_video
                    render_reprojection_video(
                        stac_h5_path=str(ik_path),
                        calibration_path=str(calib),
                        video_dir=str(sd),
                        env_points3d_path=str(env_pts),
                        registered_h5_path=str(reg_file),
                        raw_points3d_path=str(raw_pts),
                        clip_csv_path=str(trial_csv_path),
                        trial_number=trial_id,
                        output_path=str(reproj_out),
                        fps=30,
                    )
                    logging.info(f"[{animal} t{trial_id:03d}] reproj saved")
                total_done += 1
            except Exception as e:
                logging.error(f"[{animal} t{trial_id:03d}] FAILED: {e}")
                logging.error(traceback.format_exc())
                total_failed += 1
            gc.collect()

    logging.info(f"\n=========\nv17 batch done. Successful: {total_done}, Failed: {total_failed}\n=========")


if __name__ == "__main__":
    main()
