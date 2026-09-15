#!/usr/bin/env python
"""Re-anchor v25's joystick position for v26, using empirical mocap instead of
the static-calibration triangulation pipeline.

v25's joystick_base position (build_v25_stac_xml.py's JOYSTICK_BASE_POS) came
from compute_joystick_geometry_v25.py: triangulate ONE static calibration
photo (static_joystick_labels_withbottom.v001.slp, shot during CFL_36's
session) and transform it into model frame via a ROTATION averaged over 12
trials spanning all three recording sessions (CFL_35/36/37 -- different
dates: 2023-12-20, 2024-01-28, 2024-04-23).

That anchor turned out to be systematically wrong: comparing it against
register_v25_mocap.py's own js_base_resting_position attr (the median
per-trial js_base position over each trial's own "resting" frames, already
in v25's model frame) shows a consistent +x/+y offset of ~5-9mm across EVERY
SINGLE one of the 15 trials, in every session -- not a per-session
remounting drift (which would point in different directions per session),
but a uniform bias, i.e. a computation-pipeline issue somewhere in the
static-photo-triangulation-plus-averaged-rotation approach. Since v25's
joystick_base only has +-6mm of x_slide/y_slide (and no rotation/z DOF at
all -- see build_v25_stac_xml.py's docstring for why it's untilted), that
~5-9mm bias pins x_slide and/or y_slide at their limit in every CFL_35 trial
and several CFL_36/37 trials (confirmed via run_stac_v25_smoke.py's
check_qpos_ranges output), and the model can never actually reach the real
js_base/js_ball target -- this is what shows up as "the joystick doesn't
match the mocap at all" in the v25 reprojection videos.

Fix: skip the static-photo pipeline for POSITION entirely and anchor
joystick_base directly on the mean js_base_resting_position across the same
12 "reliable" trials compute_joystick_geometry_v25.py already uses (the same
already-registered *_registered_v25_refined.h5 files -- no new registration
step needed). This is self-consistent with what STAC is actually trying to
fit, instead of routing through a separate calibration measurement. Recentering
this way drops residual per-trial scatter to ~0.2-4.4mm (checked across all
15 trials) -- comfortably inside the existing +-6mm slide range, which is
left unchanged.

shaft_length/ball_offset (physical rig constants, unrelated to the position
bug) are kept identical to v21/v22/v25 (14.363mm / 1.822mm).

Usage: .venv/bin/python3 compute_joystick_geometry_v26.py
"""
import h5py
import numpy as np
from pathlib import Path

TRAINING_ROOT = Path("/mnt/vast/niyo/luc3d-bugs/js_luc3d_hand/trainingData")

# Identical exclusion list to compute_joystick_geometry_v25.py (poor
# bundle-adjustment convergence -- CALIBRATION_REFINEMENT_REPORT.md section 3).
EXCLUDED_TRIALS = {
    "CFL_35_20240128_trial_0001",
    "CFL_37_20240423_trial_0001",
    "CFL_37_20240423_trial_0201",
}

SHAFT_LENGTH_M = 0.014363   # bottom -> base, physical rig constant (v21/v22/v25)
BALL_OFFSET_M = 0.001822    # base -> ball, physical rig constant (v21/v22/v25)


def main():
    all_trials = sorted(d.name for d in TRAINING_ROOT.iterdir()
                         if d.is_dir() and d.name.startswith(("CFL_35", "CFL_36", "CFL_37")))
    reliable = [t for t in all_trials if t not in EXCLUDED_TRIALS]

    bases = []
    for t in reliable:
        h5path = TRAINING_ROOT / t / f"{t}_registered_v25_refined.h5"
        if not h5path.exists():
            continue
        with h5py.File(h5path, "r") as f:
            bases.append(np.array(f.attrs["js_base_resting_position"]))
    bases = np.array(bases)
    print(f"Using {len(bases)} reliable trials: {reliable}")

    mean_base = bases.mean(0)
    print(f"mean js_base_resting_position (m) = {mean_base}")
    print(f"residual scatter (mm) vs mean, per reliable trial:")
    for t, b in zip(reliable, bases):
        d = (b - mean_base) * 1000
        print(f"  {t:32s} resid_mm=({d[0]:6.2f},{d[1]:6.2f},{d[2]:6.2f})  |xy|={np.hypot(d[0], d[1]):.2f}mm")

    bottom_m = mean_base - np.array([0.0, 0.0, SHAFT_LENGTH_M])
    print()
    print(f"joystick_base pos=\"{bottom_m[0]:.8f} {bottom_m[1]:.8f} {bottom_m[2]:.8f}\"")
    print(f"js_base local offset (on joystick body) = (0, 0, {SHAFT_LENGTH_M:.6f})")
    print(f"js_ball local offset (on joystick body) = (0, 0, {SHAFT_LENGTH_M + BALL_OFFSET_M:.6f})")


if __name__ == "__main__":
    main()
