# Mocap → STAC → v16 Reprojection Pipeline

End-to-end recipe for going from raw 2D pose predictions to multi-camera reprojection videos overlaying the STAC'd MuJoCo arm onto the original camera frames. Captures the working `test_reprojection_trial01_v16_fixed_ref.mp4` flow and the gotchas we hit while extending it from A36 to AT006/9/12/13.

## TL;DR — what to run

For an existing session that already has `points3d.h5`, a manual reach CSV, calibration, and camera videos:

1. **Triangulate env labels** for the session into `smoke_test_v16_reproj/<sid>_env_points3d.h5` (or symlink A36's env file there as a fallback if no env labels exist for that session — produces ~2-3° rotation drift).

2. **Regenerate `v7.h5`** with the env-anchored recipe (preferred over OLD/mocap-Procrustes since it handles sessions where the mouse moves at session start):
   ```bash
   /root/vast/eric/stac-mjx/.venv/bin/python /root/vast/eric/stac-mjx/regen_v7_old_recipe.py \
       --root /root/vast/eric/azim-emg-data \
       --events <session_dir_name>
   ```
   Looks up `smoke_test_v16_reproj/<sid_short>_env_points3d.h5`, Procrustes-aligns it to A36's env, chains with A36's known-good rotation. Writes `<session>_mujoco_registered_forelimb_right_v7_oldrecipe.h5`.

3. **Run STAC + reprojection** via `smoke_test_v16_reproj.py`. The `SESSIONS` list points at the regenerated v7s, the manual reach CSVs, and (for sessions without env labels) A36's env points as a fallback.

**Real video framerate is ~200fps** (the mp4 metadata says 25fps because the videos were re-encoded with the wrong rate tag, but the per-frame timestamps in `*_tone_on.mat:videoFrames_timestamps` are 5ms apart). Reach offsets in the reach CSVs are in real-time frames, so 250ms = 50 frames. Smoke-test clips are **50 frames long with a 5-frame pre-roll** (clip starts at `reach_start - 5` so the first ~25ms of video shows the arm at rest before the labeled reach onset).

## Pipeline stages

```
points3d.h5 (raw 3D mocap)        env_points3d.h5 (3D env markers)
        │                                    │
        ▼ regen_v7_old_recipe.py             │
   v7.h5 (registered to MuJoCo)              │
        │                                    │
        ▼ run_stac (v16 recipe)              │
   ik_only.h5 (per-frame qpos)               │
        │                                    │
        └─────► render_reprojection_video ◄──┘
                           │
                           ▼
                    *_reproj_v16.mp4
```

### Stage 1 — Triangulation (already done for all 5 sessions)
- Reads predicted 2D labels (`<session>/CamA/<session>_CamA_resample.predictions.slp` etc.)
- Triangulates with anipose-format `calibration.toml` per session
- Driver: `/root/vast/eric/azim-emg-data/triangulate_3d_all.py` (uses `sleap_io` + `sleap_3d`, env: `/root/miniforge3/envs/sleap_nn`)
- Output: `<session>/<session>_points3d.h5` with `tracks` shape `(T, 1, 3, 3)` — frames × tracks × [shoulder, elbow, wrist] × [x,y,z] in **mm**.

### Stage 2 — Mocap registration to MuJoCo (`v7`)
Aligns triangulated mocap to the MuJoCo `mouse_forelimb_right` model frame so STAC can fit. Writes `<session>_mujoco_registered_forelimb_right_v7*.h5` with `tracks (T, 1, 3, 3)` in **meters** plus attrs `registration_rotation`, `registration_translation`, `shoulder_reference_mm`, `target_humerus_len`, `target_forearm_len`.

**Three rotation strategies — use env-anchored.**

| Strategy | Rotation source | Where it fails |
|---|---|---|
| **env-anchored (preferred)** | Procrustes-align this session's env points to A36's env points → chain with A36's working R | requires env points triangulated for the session AND for A36 |
| mocap-resting Procrustes | median of low-velocity early frames in mocap, Procrustes to MODEL_CENTERED | unreliable when mouse is moving at session start (e.g. AT006 first 150 frames have 5× higher velocity than later) |
| `azim-emg-data/register_to_mujoco.py` | mocap-resting with median over 1000-frame velocity-thresholded window | model mirror-flips in CamB/CamC for sessions where the natural resting orientation differs from the model's gravity-settled pose |

**Why env-anchored works:** the bars and water spout are physically fixed in the rig across sessions, so triangulating them in each session gives a stable world-frame anchor. The rotation between sessions' calibration frames is then ~2° (just camera-mount jitter), so chaining with A36's known-good R produces a registration rotation within ~2-3° of A36's — small enough to look correct in reprojection.

`regen_v7_old_recipe.py` does this automatically when it finds a `<sid>_env_points3d.h5` next to the smoke-test outputs (or accepts `--anchor-ref-env` and `--anchor-ref-v7`). Pass `--no-anchor` to fall back to mocap-resting Procrustes.

**Common steps regardless of rotation strategy:**

**Common steps in both recipes:**
1. Savitzky–Golay smooth raw mocap (window=15, polyorder=3).
2. Center on `shoulder_reference_mm`, convert mm → m.
3. Per-frame bone rescale to model lengths (`humerus≈10.37 mm`, `forearm≈10.92 mm`).
4. Procrustes-rotate resting median to `MODEL_CENTERED` (model's gravity-settled resting pose).
5. Rotate, then translate so shoulder lands on `MODEL_SHOULDER` (≈`[0.01404, 0.00953, 0.07198]` m).

### Stage 3 — STAC fit (v16 settings)
- Run via `stac_mjx.run_stac` with `SCALE_FACTOR=1.0`, `MOCAP_SCALE_FACTOR=1.0`, `N_ITERS=6`, `N_SAMPLE_FRAMES=50`, `M_REG_COEF=0`.
- MuJoCo model: `/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml`.
- Keypoint pairs: `Shoulder→humerus, Elbow→ulna, Wrist→wrist`.
- Input: 100-frame clip extracted from `v7.h5` starting at the reach frame.
- Output: `<clip>_fit.h5` (offsets) and `<clip>_ik.h5` (per-frame qpos + FK marker sites).

**JIT cost:** ~25–30 s for the first frame (compilation), then ~0.4 s/frame. With `N_ITERS=6` × 100 frames ≈ 5–6 minutes total per session.

### Stage 4 — Reprojection video
`stac_mjx.reproject_stac_model.render_reprojection_video()`:
- Reads STAC `ik.h5`, calibration.toml, the 3 cam mp4s, env_points3d.h5, registered v7.h5, raw points3d.h5, and a v16-format trial CSV.
- Builds MuJoCo cameras from calibration extrinsics rotated through the v7 registration.
- Per-frame: undistort camera image → green-screen render the MuJoCo arm at the STAC qpos → composite (alpha 0.5) onto the camera frame → draw env markers and raw mocap keypoints.
- Stitches CamA / CamB / CamC side-by-side into one 1536×512 mp4.

The `env_points3d_path` only affects the white bar/spout overlays; it does **not** influence model-camera alignment. Sub-mm differences in env triangulation produce a couple of pixels of marker drift, not a flip.

## Inputs per session

For each `<session>` in `{A36-1_2023-07-18_16-54-01_lightOff_video, AT006_2024-02-28_15-39-40_LightOff_video, AT009_2024-04-25_13-50-34_LightOff_video, AT012_2024-06-21_12-32-30_LightOff_video, AT013_2024-06-20_13-41-20_LightOff_video}`:

```
/root/vast/eric/azim-emg-data/<session>/
    calibration.toml                                  ← anipose: 3 cams (CamA/B/C), K + dist + rvec + tvec
    <session>_points3d.h5                             ← raw triangulated mocap (T,1,3,3) in mm
    <session>_mujoco_registered_forelimb_right_v7.h5            ← BUGGY (median-resting recipe)
    <session>_mujoco_registered_forelimb_right_v7_oldrecipe.h5  ← REGENERATE with regen_v7_old_recipe.py
    <session>_tone_on.mat                             ← MATLAB trial timings (also gives reach idx)
    CamA/<session>_CamA_resample.mp4                  ← 512×512 mp4 per camera
    CamB/<session>_CamB_resample.mp4
    CamC/<session>_CamC_resample.mp4
```

Reach trial CSVs:

```
/root/vast/eric/azim-emg-data/_reach_csv_manual_refined/<sid>_reach_metrics.csv  ← USE THIS (user-curated, supersedes auto)
    columns: start_idx (= trial_start), reach_startPos, reach_endPos, reach_apexPos
/root/vast/eric/azim-emg-data/_reach_csv_manual_refined/<sid>_manual.json        ← per-trial accept/reject + manual offsets
/root/vast/eric/azim-emg-data/_reach_csv_clean/<sid>_reach_metrics.csv           ← old auto-detected (don't use)
```

The historical A36 v16 baseline used a separate CSV with different columns; preserved for reference but the smoke test now uses manual_refined for all sessions:
```
/root/vast/eric/stac-mjx/refined_reaches/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv
    columns: trial, trial_start, start, end
```

Env labels (newly redrawn, used to triangulate env markers):
```
/root/vast/eric/azim-emg-data/draw-env/draw-env-emg-project_with-A36-template.slp   ← A36 + AT006 only
/root/vast/eric/azim-emg-data/draw-env/AT006_..._video_env.slp                       ← AT006 full-session predictions
```
Env skeleton: `BarL_TL, BarL_TR, BarL_BL, BarL_BR, BarR_TL, BarR_TR, BarR_BL, BarR_BR, water_spout_tip, water_spout_1mm` (10 nodes).

**AT009/AT012/AT013 do NOT have env labels** in any slp on disk yet — reprojection can't run for them until those are labeled.

## Original A36 baseline (the reference output)

```
/root/vast/eric/stac-mjx/test_reprojection_trial01_v16_fixed_ref.mp4   (Apr 7, 2026 — the "perfect" reference)
```
Was produced by:
- v7: `/root/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5` (OLD recipe)
- env: `/root/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5`
- raw: `/root/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/points3d.h5`
- calibration: `/root/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/calibration.toml`
- video_dir: `/root/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env`
- STAC IK: `/root/vast/eric/stac-mjx/refined_STACed_data_janelia_moving_shoulder_v2/A36-1_..._trial01_ik.h5`
- trial CSV: `/root/vast/eric/stac-mjx/refined_reaches/A36-1_..._tone_on_off_trials_edited.csv` trial 1

The script that produced it (likely an earlier draft of `run_stac_janelia_moving_shoulder.py`) has been overwritten — the working surface is `run_stac_janelia_moving_shoulder_v16.py`.

## Scripts in this repo

| Path | Purpose |
|---|---|
| `/root/vast/eric/stac-mjx/run_stac_janelia_moving_shoulder_v16.py` | Original v16 driver (A36 only). Iterates over reaches in the trials CSV, runs STAC, calls `render_reprojection_video()`. |
| `/root/vast/eric/stac-mjx/regen_v7_old_recipe.py` | Regenerates `v7_oldrecipe.h5` for any session. Now supports env-anchored rotation (default — looks up `smoke_test_v16_reproj/<sid_short>_env_points3d.h5` and chains with A36's R) plus `--no-anchor` fallback (mocap-resting Procrustes) and `--anchor-ref-{env,v7}` overrides. |
| `/root/vast/eric/stac-mjx/smoke_test_v16_reproj.py` | Single-trial-per-session smoke test (5 sessions, first valid reach only). Triangulates new env labels when available, picks first valid reach from manual_refined CSV, runs STAC (50-frame clip + 5-frame pre-roll), renders v16 reprojection. `existing_ik="AUTO"` reuses prior STAC IK from the smoke-test output dir if found. |
| `/root/vast/eric/stac-mjx/run_stac_5sessions_all_reaches.py` | Batch driver: runs STAC + reprojection on **every** valid reach in `_reach_csv_manual_refined` for all 5 sessions. Idempotent — skips reaches whose `<tag>_reproj_v16.mp4` already exists, reuses `<tag>_ik.h5` if present. Output dir: `final_STACed_5sessions_v16/<sid>/`. ETA ~3-4 hours for 278 reaches. |
| `/root/vast/eric/stac-mjx/stac_mjx/reproject_stac_model.py` | `render_reprojection_video()` — composites STAC model onto camera frames + draws env/mocap overlays. |
| `/root/vast/eric/stac-mjx/stac_mjx/multicam.py` | `load_calibration`, `load_env_points3d`, `load_registration_transform`, `load_raw_shoulder_positions`, `CameraParams`, `RegistrationTransform`. |
| `/root/vast/eric/azim-emg-data/triangulate_3d_all.py` | Triangulates predicted 2D labels into `points3d.h5`. Requires `sleap_io` + `sleap_3d` (env `sleap_nn`, missing on this box currently). |
| `/root/vast/eric/azim-emg-data/register_to_mujoco.py` | **Buggy**: produces v7 with median-resting reference → mirror-flipped model. Use `regen_v7_old_recipe.py` instead. |
| `/root/vast/eric/azim-emg-data/run_3d_pipeline_for_done_events.sh` | Bash driver that triangulates + registers all 5 sessions (calls the buggy register_to_mujoco). |

## Python environments

- `/root/vast/eric/stac-mjx/.venv/bin/python` — has stac_mjx, JAX, MuJoCo, OpenCV, h5py, hydra, omegaconf, scipy. **No** sleap_io.
- `/root/miniforge3/envs/track_mjx/bin/python` — exists, no sleap_io.
- `/root/miniforge3/envs/stac-mjx-env/bin/python` — exists, no sleap_io.
- `/root/miniforge3/envs/sleap_nn` — referenced in pipeline scripts but **missing on disk** (would be needed to re-run `triangulate_3d_all.py`).

For env triangulation we bypass sleap_io and read `.slp` directly with h5py + cv2 (see `triangulate_env()` in `smoke_test_v16_reproj.py`).

## Output

Smoke test (`smoke_test_v16_reproj.py`) → `/root/vast/eric/stac-mjx/smoke_test_v16_reproj/`:
- `<sid>_env_points3d.h5` (fresh triangulation, or symlink to A36 env when no labels exist)
- `<sid>_v16_trials.csv` (synthesized v16-format CSV per session)
- `<sid>_fit.h5`, `<sid>_ik.h5`
- `<sid>_trial01_reproj_v16.mp4` ← single-trial smoke output

Full batch (`run_stac_5sessions_all_reaches.py`) → `/root/vast/eric/stac-mjx/final_STACed_5sessions_v16/<sid>/`:
- `<sid>_trial<NNN>_fit.h5`, `<sid>_trial<NNN>_ik.h5`
- `<sid>_trial<NNN>_v16_trials.csv`
- `<sid>_trial<NNN>_reproj_v16.mp4`
- `clips/<sid>_trial<NNN>_clip.h5` (intermediate 50-frame mocap clips)

The `<NNN>` index is the row number in the manual_refined CSV (3-digit zero-padded, 0-indexed). So `_trial000_` = first row, `_trial046_` = 47th row, etc. — matches the source CSV directly.

## Known gotchas

1. **Two `v7.h5` files exist for A36** in different roots; only the `CVAT_mouse_reach` one matches the working `_fixed_ref.mp4`. The `azim-emg-data` one was regenerated by the buggy median-resting script and produces a flipped model.
2. **`shoulder_reference_mm` matters more than it looks** — it's used as a fixed anchor in both the registration *and* the inverse transform inside `render_reprojection_video`. Different reference → different camera-pose math → flipped model.
3. **CSV format mismatch** — `_reach_csv_manual_refined/*_reach_metrics.csv` uses `(start_idx, reach_startPos, reach_endPos, reach_apexPos)`; the v16 reprojection expects `(trial, trial_start, start, end)`. The smoke test synthesizes a v16-format CSV when needed.
4. **Camera name case** — `calibration.toml` files differ in `name = "CamA"` vs `"camA"`; the multicam loader passes that through but `render_reprojection_video` tolerates both via case-variant directory probing.
5. **Env triangulation drift** — re-triangulating from a freshly-redrawn slp produces sub-mm shifts (~0.5 mm) → a few pixels of marker drift. The model itself doesn't move; only the white overlays.
6. **Frame numbering** — STAC IK frames map 1:1 to video frames `[trial_start + start, ..., trial_start + start + 99]`. Don't mix CSV formats or you'll render the wrong video segment behind a correct STAC pose.

## Reproducibility checklist

If you want to re-run this whole pipeline on a fresh checkout, you need:

1. **Python env at `/root/vast/eric/stac-mjx/.venv`** with `stac_mjx`, `jax`, `mujoco`, `cv2`, `h5py`, `tomllib`, `hydra`, `omegaconf`, `imageio`, `imageio_ffmpeg`, `dm_control`, `pandas`, `scipy`. (No `sleap_io` — env triangulation reads slp files via h5py directly.)
2. **MuJoCo model XML** at `/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml` (used by both STAC fit and reprojection).
3. **Reference (A36) v7.h5** at `/root/vast/eric/CVAT_mouse_reach/all_data/A36-1_2023-07-18_16-54-01_lightOff_video/A36-1_2023-07-18_16-54-01_lightOff_video_mujoco_registered_forelimb_right_v7.h5` — provides the known-good `registration_rotation` for env-anchoring. The script that produced this v7 has been overwritten; the H5 is the source of truth.
4. **Reference (A36) env_points3d.h5** at `/root/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5` — provides A36's bar/spout positions for env-Procrustes.
5. **Per-session points3d.h5** at `/root/vast/eric/azim-emg-data/<session>/<session>_points3d.h5` — comes out of `triangulate_3d_all.py` (requires `sleap_io` + `sleap_3d`, env `sleap_nn` was missing on this box at last check; you may need to regenerate from the predicted `*_predictions.slp` per camera).
6. **Per-session calibration.toml** at `/root/vast/eric/azim-emg-data/<session>/calibration.toml` — anipose format, 3 cameras.
7. **Per-session videos** at `/root/vast/eric/azim-emg-data/<session>/Cam{A,B,C}/<session>_Cam{A,B,C}_resample.mp4`. Encoded fps in metadata = 25 but real recording is ~200fps (see "Real video framerate" note above).
8. **Manual reach CSVs** at `/root/vast/eric/azim-emg-data/_reach_csv_manual_refined/<sid>_reach_metrics.csv` — user-curated, supersedes auto.
9. **(Optional) per-session env triangulations** at `/root/vast/eric/stac-mjx/smoke_test_v16_reproj/<sid>_env_points3d.h5`. AT006 has its own (triangulated from `AT006_..._video_env.slp`); A36/AT009/AT012/AT013 either have their own (A36 from CVAT_mouse_reach) or symlink to A36's as a fallback.

Reproduction order:

```bash
# 1. Triangulate env labels (only sessions that have env labels in a slp)
#    — done in smoke_test_v16_reproj.py via triangulate_env() which reads draw-env-emg-project_with-A36-template.slp
#    For sessions without env labels, symlink:
ln -sf /root/vast/eric/CVAT_mouse_reach/A36-1_..._env/points3d.h5 \
       /root/vast/eric/stac-mjx/smoke_test_v16_reproj/<sid>_env_points3d.h5

# 2. Regenerate v7 with env-anchored rotation
/root/vast/eric/stac-mjx/.venv/bin/python /root/vast/eric/stac-mjx/regen_v7_old_recipe.py

# 3. Smoke test (1 reach per session) — visually verify scapula angle / reach onset
/root/vast/eric/stac-mjx/.venv/bin/python /root/vast/eric/stac-mjx/smoke_test_v16_reproj.py

# 4. Full batch (all valid reaches)
/root/vast/eric/stac-mjx/.venv/bin/python /root/vast/eric/stac-mjx/run_stac_5sessions_all_reaches.py
```

## Smoke test invariants verified so far (2026-04-30)

- A36 with OLD v7 + original env + original calibration + original raw + original trial CSV → **byte-for-byte identical** to `test_reprojection_trial01_v16_fixed_ref.mp4`.
- A36 with `regen_v7_old_recipe.py`-produced v7 → ~2–3% pixels differ from baseline (small rotation diff, no flip).
- **AT006 with env-anchored v7 + manual_refined CSV + 50-frame clip + 5-frame pre-roll → user confirmed "looks great" (scapula angle correct, reach visible from rest in first frame).**
- AT006 with mocap-resting Procrustes v7 → scapula tilted ~10° wrong (mouse moving in early frames, "resting" median was on a moving arm).
- AT009/AT012/AT013 with A36's R direct (no env labels available; symlinked A36's env into `smoke_test_v16_reproj/<sid>_env_points3d.h5` so Procrustes returns identity → R_total = A36 R) → ~2–3° calibration-drift offset, used as a smoke-test approximation. Replace the symlinks with real triangulated env_points3d.h5 once those sessions are env-labeled.

## Pre-roll and clip duration

The reach CSV's `reach_startPos` marks the first frame of motion (or first frame where the arm is touching the bar, depending on labeling convention). To render a clip that starts with the arm visibly at rest, the smoke test uses:

- `clip_start_frame = trial_start + reach_startPos - 5`  (5-frame pre-roll = ~25ms of pre-reach context)
- `n_frames = 50`  (250ms total at the real ~200fps capture rate)

This is implemented in `first_valid_reach()` and `extract_clip()` in `smoke_test_v16_reproj.py`. For sessions with longer reaches you may want to bump n_frames; the pre-roll is fixed at 5 unless you change `pre_roll=` in `first_valid_reach`.

## "Resting" frame detection (FYI)

Several places in the pipeline try to find frames where the mouse is "at rest" (low keypoint velocity). It's used to:
- Compute a stable `shoulder_reference_mm` (fixed anchor in the calibration frame).
- Compute the Procrustes rotation from mocap-resting → MODEL_CENTERED (in fallback mode only — env-anchored bypasses this).

Velocity threshold = 25th percentile of `||Δelbow|| + ||Δwrist||` smoothed over a 21-frame uniform window, evaluated within the first 500 frames. AT006 needed this widened from 20 to 500 frames because its mouse is twitching for ~150 frames before settling. AT013 has a truly resting start (vel mean 0.001 mm/frame); AT009 doesn't (vel mean 0.085).

When env-anchoring is active, the resting computation only affects `shoulder_reference_mm`, not the rotation — so the velocity heuristic only has to be approximately right.
