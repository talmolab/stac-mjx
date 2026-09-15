# Prompt for the other vast 2-GPU job: STAC + reproject AT009 and AT013

Paste this into a fresh Claude session on the other vast machine.

---

You're picking up half of a multi-GPU STAC fitting batch. The pipeline is documented at `/root/vast/eric/stac-mjx/MOCAP_STAC_PIPELINE.md` — read it first if you want full context. This prompt has the minimum needed to just run the work.

## What you're running

Two parallel STAC + v16 reprojection processes on this machine's two GPUs:
- **GPU 0**: AT009_2024-04-25_13-50-34_LightOff (45 valid reaches)
- **GPU 1**: AT013_2024-06-20_13-41-20_LightOff (44 valid reaches)

The other vast machine is concurrently running A36 + AT006 + AT012 — don't touch those.

Each reach takes ~3.7 min on a single GPU (50-frame STAC fit + render). With 2 GPUs in parallel doing one session each: ~3 hours wall time.

## Prerequisites — verify before launching

```bash
# 1. The pipeline scripts and configs:
ls /root/vast/eric/stac-mjx/run_stac_5sessions_all_reaches.py
ls /root/vast/eric/stac-mjx/regen_v7_old_recipe.py
ls /root/vast/eric/stac-mjx/configs/stac/stac_mouse_arm.yaml
ls /root/vast/eric/stac-mjx/configs/model/mouse_arm.yaml
ls /root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml

# 2. Per-session inputs (raw points3d + regenerated v7 + calibration + videos):
for s in AT009_2024-04-25_13-50-34_LightOff_video AT013_2024-06-20_13-41-20_LightOff_video; do
  ls /root/vast/eric/azim-emg-data/$s/${s}_points3d.h5
  ls /root/vast/eric/azim-emg-data/$s/${s}_mujoco_registered_forelimb_right_v7_oldrecipe.h5
  ls /root/vast/eric/azim-emg-data/$s/calibration.toml
  ls /root/vast/eric/azim-emg-data/$s/CamA/*.mp4 /root/vast/eric/azim-emg-data/$s/CamB/*.mp4 /root/vast/eric/azim-emg-data/$s/CamC/*.mp4
done

# 3. Reach CSVs:
ls /root/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT009_reach_metrics.csv
ls /root/vast/eric/azim-emg-data/_reach_csv_manual_refined/AT013_reach_metrics.csv

# 4. The A36 reference env points (used by both AT009 and AT013 because they don't have their own env labels yet):
ls /root/vast/eric/CVAT_mouse_reach/A36-1_2023-07-18_16-54-01_lightOff_video_env/points3d.h5

# 5. GPUs visible:
nvidia-smi
```

If any of these are missing, **stop** and tell the user — the filesystem might not be shared the way we assumed.

## How the script picks the session

`run_stac_5sessions_all_reaches.py` already has all 5 sessions hardcoded in its `SESSIONS` list. It honors a `STAC_SESSION_FILTER` env var (comma-separated substrings) to only run a subset. Pin each process to a different GPU with `CUDA_VISIBLE_DEVICES`.

## Launch — two background processes, one per GPU

```bash
# GPU 0 -> AT009
CUDA_VISIBLE_DEVICES=0 STAC_SESSION_FILTER="AT009" /root/vast/eric/stac-mjx/.venv/bin/python /root/vast/eric/stac-mjx/run_stac_5sessions_all_reaches.py > /tmp/vastlrn_at009.log 2>&1 &

# GPU 1 -> AT013
CUDA_VISIBLE_DEVICES=1 STAC_SESSION_FILTER="AT013" /root/vast/eric/stac-mjx/.venv/bin/python /root/vast/eric/stac-mjx/run_stac_5sessions_all_reaches.py > /tmp/vastlrn_at013.log 2>&1 &

# Verify they're running on different GPUs
ps aux | grep "python.*run_stac_5sessions" | grep -v grep
nvidia-smi
```

## Monitor

```bash
# Tagged combined view:
( tail -f /tmp/vastlrn_at009.log | sed -u 's/^/[AT009] /' &
  tail -f /tmp/vastlrn_at013.log | sed -u 's/^/[AT013] /' ) | \
  grep -E --line-buffered "==== |done —|complete|FAILED|Error:|Traceback"
```

Each line `[AT009] [N/278] AT009_..._trial<NNN> done — elapsed Xmin` = one finished reach. The N/278 counter is the global counter inside the process; ignore it (it counts only reaches inside the filter).

Expected: ~3.7 min/reach. AT009 = 45 reaches → ~2.8 hr. AT013 = 44 reaches → ~2.7 hr. Run in parallel, total wall = ~3 hr.

## Output location

Both processes write to the **same shared output dir** as the other machine:
```
/root/vast/eric/stac-mjx/final_STACed_5sessions_v16/<session>/<session>_trial<NNN>_{fit,ik,reproj_v16}.{h5,mp4}
```

Idempotent — if the script restarts mid-batch, it skips reaches whose `_reproj_v16.mp4` already exists.

## When done

```bash
# Verify
ls /root/vast/eric/stac-mjx/final_STACed_5sessions_v16/AT009_*/*.mp4 | wc -l   # should be 45
ls /root/vast/eric/stac-mjx/final_STACed_5sessions_v16/AT013_*/*.mp4 | wc -l   # should be 44

# Tell the user when both are done.
```

## Critical reminders

- **Use `/root/vast/eric/stac-mjx/.venv/bin/python`** for everything. The venv has stac_mjx, JAX with CUDA, MuJoCo, etc.
- **Don't run AT006, A36, or AT012** — those are running on the other machine. The `STAC_SESSION_FILTER` substrings above prevent that.
- **Don't regenerate the v7 files** — they already exist with the env-anchored rotation that the user validated visually. Regenerating would lose the validated rotation.
- **Don't use `_reach_csv_clean`** — use `_reach_csv_manual_refined` (already what the script uses).
- **Don't change clip length or pre-roll** — 50 frames + 5-frame pre-roll is what was validated.
- **Don't push or commit anything** to git unless the user explicitly asks.
