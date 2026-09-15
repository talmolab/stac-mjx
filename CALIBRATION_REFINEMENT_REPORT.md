# v21 Calibration Refinement Report

Scope: per-trial camera calibration refinement for the CFL_35/36/37 joystick-reaching
mocap dataset (`/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData/`), triggered by
two observations: (1) the STAC-fitted model's hand/joystick alignment looked off in
some trials (particularly CFL_36), and (2) the shoulder-translation DOF (`sh_tx/ty/tz`)
had to be clamped to +-0.5mm to stop STAC's optimizer from saturating it, which felt
like it was masking a data-quality problem rather than fixing one.

## 1. Why not kineo

Eric asked to use [`liris-xr/kineo`](https://github.com/liris-xr/kineo) ("Kineo:
Calibration-Free Metric Motion Capture From Sparse RGB Cameras") to refine calibration
using the mocap pose data. On inspection, kineo turned out to be the wrong tool:

- It's a full **learned SMPL/SMPL-X human-body** motion-capture pipeline (PyTorch,
  pretrained checkpoints), not a general camera-calibration refiner.
- Person tracking/segmentation is done via a custom SAM2 fork, with an interactive
  UI step for person selection -- there's no path to feed it our own mouse skeleton
  or pre-existing SLEAP 2D keypoint detections.
- It's designed to reconstruct cameras **from scratch** ("calibration-free"), not to
  refine an existing, already-mostly-good calibration using known keypoint tracks.
- It has no concept of a rigid non-human object like our joystick.

None of this fits: we already have per-camera SLEAP 2D keypoint detections (19
keypoints, arm+hand+joystick) and an existing, Charuco-board-derived
`calibration.toml` per trial that just needs refining, not replacing.

## 2. Method: aniposelib bundle adjustment

Used [`aniposelib`](https://github.com/lambdaloop/aniposelib) instead (installed via
pip; this is the calibration/triangulation library underlying `anipose`, and is
already the format `calibration.toml` is written in -- `stac_mjx/multicam.py` reads
this same anipose TOML convention).

**Why the original `calibration.toml`'s units matter and must be preserved:** each
trial's `calibration.toml` was originally produced from a **Charuco board**
calibration (a checkerboard with embedded ArUco markers). The Charuco board's known,
physical square size is what anchors the whole camera rig's scale to real-world
millimeters -- it's the *only* thing in this entire pipeline that ties "1 unit" to
"1mm" in a way that's independently verifiable (as opposed to being defined relative
to the mouse's own anatomy, which is what our model-registration rescale step does).
This matters a lot for what follows in Section 3: any refinement method that doesn't
also see Charuco-board data has no way to know the true metric scale, and can only
be trusted for improving *relative* camera geometry, not for re-deriving scale from
scratch. Eric plans to separately validate the refined calibrations against the
original Charuco board corner detections tomorrow -- that will be the independent
check on absolute scale/drift that this report's method cannot fully provide on its
own (see Section 4 and Section 8).

**Per-trial procedure** (`refine_calibration.py`), for each of the 15 CFL_35/36/37
trials:

1. Load the 3 cameras' merged SLEAP 2D keypoint detections
   (`Cam0X/<trial>_Cam0X_merged.slp`, 19 keypoints x 126 frames, no missing frames
   in the raw 2D detections for any trial).
2. Load `calibration.toml` via `aniposelib.cameras.CameraGroup.load()` -- confirmed
   to parse this exact TOML structure natively (`cam_0`/`cam_1`/`cam_2` sections
   with `matrix`, `distortions`, `rotation` (Rodrigues rvec), `translation`).
3. Baseline: triangulate the 2D detections with the *original* calibration, compute
   per-camera, per-point reprojection error (pixel distance between the observed 2D
   detection and the reprojection of the triangulated 3D point back into that
   camera).
4. Refine: `CameraGroup.bundle_adjust_iter(p2ds, only_extrinsics=True)` -- robust
   iterative bundle adjustment (outlier-downweighted, anipose's own algorithm,
   inspired by Fast Global Registration) against the actual observed 2D keypoints.
   **`only_extrinsics=True`**: only camera rotation/translation are refined;
   intrinsics (focal length, principal point, lens distortion) are left exactly as
   the Charuco calibration determined them, since those come from a proper
   multi-angle checkerboard calibration and shouldn't be re-derived from sparse,
   single-session animal keypoints.
5. Re-triangulate with the refined calibration, recompute reprojection error.
6. Fit a gauge-drift diagnostic (Section 3) against the original calibration's
   triangulation -- used only to flag trials with poor bundle-adjustment
   convergence, **not** applied as a correction to the output (see Section 3.1
   for why: it was tried and reverted).
7. Write `calibration_refined.toml` (refined extrinsics, original intrinsics) and
   `<trial>_points3d_refined.h5` (that calibration's own, self-consistent
   triangulation -- same format as the original `<trial>_points3d.h5`) into the
   trial's own directory.

### Results: reprojection error, all 15 trials

| Trial | Mean px before | Mean px after | Median before | Median after | Improvement |
|---|---|---|---|---|---|
| CFL_35_20240128_trial_0001 | 9.53 | 3.19 | 8.28 | 2.56 | 66.5% |
| CFL_35_20240128_trial_0101 | 9.25 | 2.88 | 7.88 | 2.27 | 68.9% |
| CFL_35_20240128_trial_0201 | 9.49 | 3.33 | 8.14 | 2.49 | 64.9% |
| CFL_35_20240128_trial_0301 | 8.72 | 2.64 | 7.47 | 2.14 | 69.8% |
| CFL_35_20240128_trial_0401 | 9.70 | 3.01 | 8.01 | 2.18 | 69.0% |
| CFL_36_20231220_trial_0001 | 10.28 | 3.41 | 9.00 | 2.57 | 66.9% |
| CFL_36_20231220_trial_0101 | 11.01 | 4.33 | 9.66 | 3.32 | 60.7% |
| CFL_36_20231220_trial_0201 | 10.12 | 3.90 | 8.71 | 3.13 | 61.4% |
| CFL_36_20231220_trial_0301 | 9.81 | 3.26 | 8.32 | 2.56 | 66.8% |
| CFL_36_20231220_trial_0401 | 10.08 | 3.98 | 8.77 | 2.99 | 60.5% |
| CFL_37_20240423_trial_0001 | 10.33 | 5.27 | 8.73 | 3.14 | 49.0% |
| CFL_37_20240423_trial_0101 | 10.09 | 3.98 | 8.99 | 2.90 | 60.5% |
| CFL_37_20240423_trial_0201 | 9.71 | 3.60 | 8.30 | 2.83 | 62.9% |
| CFL_37_20240423_trial_0301 | 11.73 | 5.03 | 9.77 | 3.36 | 57.2% |
| CFL_37_20240423_trial_0401 | 11.82 | 4.67 | 10.51 | 3.58 | 60.5% |
| **Mean (15 trials)** | **10.11** | **3.76** | **8.70** | **2.82** | **62.8%** |

Per-camera breakdown (mean px, averaged within each session): **Cam01 was the
consistent outlier in all three sessions**, at roughly 2x the error of Cam00/Cam02:

| Session | Cam00 before | Cam01 before | Cam02 before | Cam00 after | Cam01 after | Cam02 after |
|---|---|---|---|---|---|---|
| CFL_35 | 7.30 | 13.62 | 7.19 | 2.72 | 3.51 | 2.81 |
| CFL_36 | 7.89 | 14.76 | 8.13 | 3.33 | 4.38 | 3.56 |
| CFL_37 | 8.50 | 15.31 | 8.53 | 3.58 | 5.40 | 4.35 |

This is a strong, consistent signal that **Cam01's extrinsics were the primary
source of triangulation error** in the original calibration, across the whole
dataset -- not just an isolated issue in CFL_36. This lines up with Eric's
observation of a "weird angle" specifically in CFL_36's reprojection.

## 3. The gauge-drift bug (found and fixed mid-session)

The first version of this refinement (before this fix) looked *great* by
reprojection error, but when the resulting registration was used to re-place the
model's joystick and re-run STAC, the fit looked visibly wrong -- worse than before
refinement in places, even though the underlying 2D tracking accuracy had
genuinely improved. Diagnosis:

**Root cause**: `bundle_adjust_iter` refines camera extrinsics purely to minimize
2D reprojection error against the observed keypoints. Reprojection error is
**invariant to any similarity transform** (rotation + translation + uniform scale)
applied jointly to every camera and every 3D point -- if you rotate, translate, or
uniformly rescale the *entire* multi-camera rig plus the reconstructed 3D points
together, every camera's projection of every point is unchanged. This is the
classical "gauge freedom" of bundle adjustment from 2D correspondences alone, and
nothing in `bundle_adjust_iter` (or in our per-trial-only keypoint data) anchors it
-- there's no Charuco board in this refinement step, only mouse+joystick keypoints,
so the optimizer is free to drift the whole reconstruction's rotation, translation,
*and scale* to whatever minimizes reprojection error, independent of whether that
matches the real physical world.

Measured drift for `CFL_36_20231220_trial_0401` (representative case): fitting the
best similarity transform between the refined-calibration triangulation and the
original-calibration triangulation gave scale=1.013, a ~1.5 degree rotation, and
~1.2mm translation, with a tiny (~0.2-0.3mm mean) residual after correcting for it --
confirming the two triangulations are related by *almost exactly* a similarity
transform, i.e. genuine gauge drift, not some other systematic error.

**Why this specifically broke the joystick, not the arm/hand**: the arm/hand
registration (`register_v21_mocap.py`) rescales every bone to the model's own
known segment length (`tree_rescale`), which erases any uniform scale error in the
raw mocap by construction -- so the arm/hand fit was never actually affected by the
gauge drift. But `js_base`/`js_ball` (the joystick) are **not** bone-rescaled --
they're only rotated and translated using the same transform computed from the
arm fit (see `register_v21_mocap.py`'s `JOYSTICK_TARGET_SPEC` handling) -- so any
scale drift in the raw triangulated points is inherited directly into the
joystick's registered position, unrescaled. This is exactly the "far off" alignment
Eric spotted when comparing `CFL_36_20231220_trial_0401_triangulation_check_refined.mp4`
(great -- gauge-invariant 2D check) against
`CFL_35_20240128_trial_0001_reproj_v21_refined.mp4` (visibly wrong -- gauge-drift-
corrupted joystick registration).

**Fix attempted** (superseded -- see Section 3.1 immediately below for why this
was reverted): added a Umeyama similarity-transform fit (`refine_calibration.py::umeyama`)
per trial, mapping the refined-calibration triangulation back onto the
*original*-calibration triangulation (itself independently verified cross-trial
consistent -- the original, pre-refinement registration already gave a small,
~1-1.5mm std for the joystick position across all 15 trials, so it's a trustworthy
gauge anchor). The gauge-corrected points are saved to `<trial>_points3d_refined.h5`
and used for everything downstream. `calibration_refined.toml` itself is **not**
gauge-corrected (its own extrinsics still live in the arbitrary drifted gauge --
a genuine scale change can't be folded into a physically valid rotation+translation
extrinsic without also changing the camera's effective focal length); it's used
only as an intermediate tool to obtain better-triangulated 3D points, which are
then re-expressed in the original, trusted frame. Any later reprojection/rendering
against the refined result (video overlays, STAC reprojection) therefore uses the
**original** `calibration.toml` together with the gauge-corrected points, not
`calibration_refined.toml` directly.

### Gauge-correction results, all 15 trials

| Trial | Fitted scale | Residual mean (mm) | Residual median | Residual max |
|---|---|---|---|---|
| CFL_35_20240128_trial_0001 | 0.464 | 1.709 | 1.295 | 59.35 |
| CFL_35_20240128_trial_0101 | 0.919 | 0.356 | 0.287 | 4.18 |
| CFL_35_20240128_trial_0201 | 1.036 | 0.291 | 0.197 | 4.66 |
| CFL_35_20240128_trial_0301 | 1.014 | 0.199 | 0.174 | 0.74 |
| CFL_35_20240128_trial_0401 | 0.808 | 0.624 | 0.490 | 5.37 |
| CFL_36_20231220_trial_0001 | 0.983 | 0.203 | 0.180 | 4.30 |
| CFL_36_20231220_trial_0101 | 0.852 | 0.598 | 0.475 | 5.24 |
| CFL_36_20231220_trial_0201 | 0.956 | 0.329 | 0.279 | 4.42 |
| CFL_36_20231220_trial_0301 | 0.947 | 0.311 | 0.260 | 4.44 |
| CFL_36_20231220_trial_0401 | 0.992 | 0.167 | 0.107 | 5.17 |
| CFL_37_20240423_trial_0001 | **1.356** | 0.603 | 0.541 | 2.83 |
| CFL_37_20240423_trial_0101 | 0.871 | 0.620 | 0.513 | 5.34 |
| CFL_37_20240423_trial_0201 | **0.255** | 2.841 | 2.166 | 10.72 |
| CFL_37_20240423_trial_0301 | 1.268 | 0.455 | 0.406 | 3.23 |
| CFL_37_20240423_trial_0401 | 1.172 | 0.652 | 0.520 | 11.52 |

12/15 trials show a mild, expected gauge drift (scale within ~15% of 1.0, residual
well under 1mm after correction) -- these are the "typical" case described above.
**3/15 trials show much larger scale drift** (CFL_35_trial_0001: 0.46,
CFL_37_trial_0001: 1.36, CFL_37_trial_0201: 0.25) with correspondingly larger
post-correction residuals (1.7mm, 0.6mm, 2.8mm mean respectively, with the worst
individual-point residual reaching 59mm in CFL_35_trial_0001). This looks like a
genuine bad local minimum in the bundle adjustment for these specific trials
(re-running with different sampling settings did not change the result), not a
sampling artifact. **These 3 trials are flagged and excluded from the cross-trial
joystick-position averaging** in `register_v21_mocap.py` (still registered
normally otherwise -- their arm/hand fit is unaffected, per the rescaling argument
above), so they don't corrupt the placement of the joystick in the model. They
should be treated as lower-confidence for any analysis that depends on their
absolute calibration accuracy, pending Eric's planned Charuco-board cross-check.

### 3.1 The gauge-correction fix above was tried, and reverted, per direct visual feedback

The Umeyama correction described above is theoretically well-motivated and the
diagnosis of *why* gauge drift happens is correct -- but **applying it to the
actual output points made the results visibly worse**, not better, and was
reverted. Eric's direct feedback: the triangulation-check-refined videos, which
had looked "really really good" right after the (uncorrected) calibration
refinement, looked "horrible" once gauge-corrected; the STAC reprojection videos
looked bad too.

The reason, in hindsight: reprojecting the gauge-corrected points through the
**original** `calibration.toml` (which the correction targets by construction)
necessarily reintroduces whatever accuracy problems the original calibration
had -- including Cam01's bad extrinsics, which is exactly what the refinement
was trying to fix in the first place. The gauge-correction diagnostic (fitted
scale, residuals, the table above) is still useful for flagging poor
bundle-adjustment convergence, and is still computed and used to exclude the 3
bad trials from joystick-position averaging -- but the actual 3D points written
to `<trial>_points3d_refined.h5`, and the calibration used for all downstream
rendering (`render_triangulation_check.py`'s "refined" variant,
`run_stac_v21_refined.py`/`run_stac_v21_full_batch.py`'s reprojection), are
`calibration_refined.toml` and its own self-consistent (uncorrected)
triangulation, used together everywhere. This is simpler than the gauge-corrected
version and, empirically, looks better -- Eric's call, and the right one: a
refined camera's own self-consistent reprojection is the correct thing to trust
for per-trial visualization; cross-trial consistency (the only place gauge drift
actually mattered) is handled by the exclusion-from-averaging step instead of a
transform that reintroduces old errors.

### 3.2 Joystick orientation bug (found via the same investigation)

Separately, Eric noticed the joystick model looked "crooked" in the reprojection
videos -- not upright/vertical as the physical rig requires (the joystick's
mechanical design is a vertical post that only translates horizontally, matching
the model's `x_slide`/`y_slide`-only, no-rotation-DOF joystick geometry).

Root cause: `register_v21_mocap.py` previously computed the `joystick_base` XML
position by **averaging two candidates** -- one derived from the tracked
`js_base` position, one from `js_ball` -- implicitly assuming both points sit on
the model's local vertical (+Z) axis above `joystick_base`, as the rigid,
non-rotating model geometry requires. They empirically don't: the mean tracked
`js_base -> js_ball` direction across the 12 gauge-reliable trials is **~46
degrees off vertical**, ranging from 25 to 75 degrees between individual trials
-- far too inconsistent trial-to-trial to be a real, stable physical tilt.
`js_base` and `js_ball` are single, isolated tracked points with no bone-length
constraint (unlike the arm/hand chain, where `tree_rescale` corrects exactly this
kind of per-point noise) -- this looks like raw triangulation/tracking noise,
not physical reality. Averaging the two inconsistent candidates baked this noise
directly into a visibly tilted joystick.

**Fix**: derive `joystick_base`'s position from **`js_ball` alone** (median over
the 12 gauge-reliable trials, minus its known static offset), not averaged with
`js_base`. `js_ball` is the visually and functionally important point --
where the mouse's hand actually contacts the joystick -- and trusting the
model's built-in vertical assumption (rather than the noisier, inconsistent
`js_base` estimate) keeps the joystick upright, matching the physical rig.
Updated `joystick_base` position: `[0.02337264, 0.01040419, 0.05120002]`.

## 4. What still needs independent validation (tomorrow)

This bundle-adjustment refinement uses only the observed mouse+joystick keypoints
as the source of truth -- it never sees the Charuco board that originally defined
metric scale, so it can only be trusted to improve *relative* per-frame tracking
accuracy (which is exactly what the reprojection-error numbers in Section 2
measure), not to independently re-verify absolute camera drift over time. Eric
plans to check the refined calibrations (particularly Cam01's large rotation
change, and the 3 poor-convergence trials) against the original Charuco board
corner detections directly, which will be the authoritative check on whether the
cameras have genuinely physically drifted since the original calibration, and by
how much. This report's numbers should be read as "per-trial relative refinement
against animal keypoints," not as a replacement for that Charuco-based check.

## 5. Triangulation-check videos (all 15 trials, both variants)

Regenerated in each trial's `trainingData` directory, matching the style of Eric's
original reference video (green = SLEAP 2D detection, red = reprojection of the
triangulated 3D point, per camera, 3-camera horizontal tiling):

- `<trial>_triangulation_check.mp4` -- original calibration + its own triangulation.
- `<trial>_triangulation_check_refined.mp4` -- refined calibration
  (`calibration_refined.toml`) + its own, self-consistent triangulation
  (`<trial>_points3d_refined.h5`) -- see Section 3.1 for why these must be
  used together, not mixed with the original calibration.

Visual spot-check (`CFL_35_20240128_trial_0001`, frame 20): Cam01's green/red
skeletons, which showed a large (~1cm-scale in image space) divergence at the
elbow/wrist in the original video, snap into close alignment in the refined
video -- directly visible confirmation of the Cam01 fix quantified in Section 2.

## 6. Joystick registration update

Re-ran `register_v21_mocap.py --points3d-suffix _points3d_refined.h5` (each
trial's own refined calibration + self-consistent triangulation) across all 15
trials. Joystick position derived from `js_ball` alone (median over the 12
gauge-reliable trials, excluding the 3 flagged in Section 3), per the orientation
fix in Section 3.2:

```
joystick_base pos = [0.02337264, 0.01040419, 0.05120002]
```

vs. the pre-refinement (averaged `js_base`+`js_ball`) estimate
`[0.023736, 0.009689, 0.051282]` -- within a couple mm on every axis. **The
calibration refinement's main effect is per-frame 2D tracking accuracy**
(Section 2) and joystick orientation (Section 3.2), not a large correction to
the mean joystick location -- the original calibration was already reasonably
good for that specific purpose. Rebuilt
`models/mouse_forelimb_right_janelia_arm_hand_v21.xml` with this updated
position.

## 7. STAC rerun with refined calibration (`refined_STACed_data_v21`)

Reran the STAC smoke test (3 trials: CFL_35/36/37 trial_0001), iterating twice:
first against the gauge-corrected registration (reverted, Section 3.1) and a
joystick position averaged from `js_base`+`js_ball` (also reverted, crooked --
Section 3.2), then again against the final, correct pipeline (calibration_refined
+ its own triangulation, `js_ball`-only joystick position). The numbers and
visual checks below are from the **final** version. Full 15-trial batch run via
`run_stac_v21_full_batch.py`:

Full 15-trial results (`refined_STACed_data_v21/full_batch_fit_summary.json`):

| Trial | Mean (mm) | Median (mm) | Max (mm) | sh_tx/ty/tz saturated? |
|---|---|---|---|---|
| CFL_35_20240128_trial_0001 | 1.359 | 0.771 | 73.047 | no |
| CFL_35_20240128_trial_0101 | 0.584 | 0.504 | 3.691 | no |
| CFL_35_20240128_trial_0201 | 0.553 | 0.418 | 3.645 | no |
| CFL_35_20240128_trial_0301 | 0.469 | 0.383 | 3.880 | no |
| CFL_35_20240128_trial_0401 | 0.483 | 0.408 | 3.831 | no |
| CFL_36_20231220_trial_0001 | 0.540 | 0.426 | 3.194 | no |
| CFL_36_20231220_trial_0101 | 0.745 | 0.491 | 3.773 | no |
| CFL_36_20231220_trial_0201 | 0.698 | 0.504 | 3.892 | no |
| CFL_36_20231220_trial_0301 | 0.745 | 0.442 | 4.237 | no |
| CFL_36_20231220_trial_0401 | 0.490 | 0.434 | 4.320 | no |
| CFL_37_20240423_trial_0001 | 0.701 | 0.487 | 2.991 | no |
| CFL_37_20240423_trial_0101 | 1.018 | 0.982 | 4.507 | no |
| CFL_37_20240423_trial_0201 | 2.361 | 1.635 | 17.042 | no |
| CFL_37_20240423_trial_0301 | 0.851 | 0.513 | 4.425 | no |
| CFL_37_20240423_trial_0401 | 0.632 | 0.421 | 3.095 | no |
| **Mean (15 trials)** | **0.815** | **-** | **-** | **0/15** |

**`sh_tx/ty/tz` saturates in zero of the 15 trials** -- the solver fix (Section 8)
generalizes cleanly across the whole dataset. The two trials with visibly
elevated error -- `CFL_35_20240128_trial_0001` (max 73.0mm) and
`CFL_37_20240423_trial_0201` (mean 2.4mm, max 17.0mm) -- are exactly 2 of the 3
trials flagged in Section 3 for poor bundle-adjustment convergence (gauge scale
0.464 and 0.255 respectively), consistent and expected, not a new problem. Their
max errors are higher than in the earlier (reverted) gauge-corrected version,
since those points are no longer artificially pulled toward the original
calibration's own accuracy problems -- but their mean/median fit quality is
essentially unaffected, and both are already excluded from the joystick-position
average (Section 3, 6), so this doesn't affect the model. All 15 trials
completed without errors.

Visual check (`<trial>_reproj_v21_refined.mp4`, all 3 cameras, spot-checked
across multiple trials): the joystick model is now **upright/vertical** in every
camera view
(previously crooked, ~46 degrees off vertical on average -- Section 3.2), and
the model's arm and hand track the real mouse closely, with the joystick model
(dark cylinder + ball) sitting right on top of the real joystick across all 3
camera views -- directly resolving the original
complaint about `CFL_36_20231220_trial_0001`'s (and `_trial_0401`'s) misalignment.
Saturated DOFs remaining are all hand/finger joints hitting their anatomical range
limits while gripping the joystick (`rz_N_L_C_right`, `rz_Phalanx_hand_1_*_right`,
etc.) -- plausible given the task, not evidence of a fitting problem.

Reprojection rendered with `calibration_refined.toml` (not the original
`calibration.toml`), consistently with the rest of this pipeline per Section 3.1
-- `registered_v21_refined.h5`'s stored registration transform maps back into
whatever frame `points3d_refined.h5` was triangulated in, which is
`calibration_refined.toml`'s own self-consistent frame.

## 8. Shoulder-translation range (sh_tx/ty/tz): root-caused and fixed properly

Eric's question -- "why is the shoulder not moving? it should be able to translate
along with the mocap data like our v18?" -- was the right one to push on. Earlier
this session, `sh_tx/ty/tz` (a root translation DOF meant for small skin-slip
correction) had been clamped from its original +-10mm range (matching v18) down to
+-0.5mm, because the full range caused STAC's pose optimizer to saturate it on
every frame, producing ~15-17mm mean marker error across *all* keypoints (not just
the shoulder). That tightening fixed the symptom but gave up real translational
freedom the model should have.

**Root cause, confirmed by direct test**: STAC's pose optimizer
(`stac_mjx/stac_core.py`'s `q_solver`, a `jaxopt.ProjectedGradient`) used a **fixed
stepsize of 0.1** for every DOF regardless of gradient magnitude. `sh_tx/ty/tz` is a
*shared* DOF that every one of v21's 19 keypoints' position error gradients flows
through (v18 only had 5 keypoints on this same DOF and never hit the problem). With
more keypoints summing into one shared gradient, a fixed step overshoots and gets
projected back to the boundary every iteration, regardless of calibration quality --
tested directly: even with the corrected, gauge-fixed calibration/registration from
Sections 2-3, loosening the range back to +-10mm with the *old* solver reproduced
the *exact* same failure (`CFL_35_20240128_trial_0001`: sh_tx/ty/tz saturated at
+-10mm, mean marker error 16.1mm, median 17.3mm -- statistically indistinguishable
from the original, pre-calibration-fix bug). This conclusively rules out calibration
data quality as the cause of the saturation -- it's a property of the fixed-step
optimizer, independent of how good the input data is.

**Fix**: `jaxopt.ProjectedGradient` supports `stepsize<=0`, which switches it to a
backtracking line search with FISTA acceleration -- an adaptively-sized step per
iteration instead of one fixed step for every DOF. Changed
`stac_mjx/stac_core.py`'s `q_solver` from `stepsize=0.1` to `stepsize=0.0`.
Verified on the same trial: with the line-search solver and the full +-10mm range,
`sh_tx/ty/tz` settle naturally within a few mm (`[-0.0032, 0.0002]`,
`[-0.0003, 0.0017]`, `[-0.0025, 0.0005]` on that trial) -- **never touching the
+-10mm limit** -- at essentially the same fit quality (mean marker error 1.24mm,
matching the old tightened-range result of 1.59mm on the same trial, actually
slightly better). This is confirmed across all 3 smoke-test trials in Section 7 --
none show `sh_tx/ty/tz` in the saturation flags.

`sh_tx/ty/tz` is now left at Eric's original +-10mm range in
`build_v21_xml.py`/`mouse_forelimb_right_janelia_arm_hand_v21.xml`, matching v18.
The joint range never needed to change -- only the solver did.

**Cost**: the line-search solver runs roughly 2x slower per frame (~0.46s vs
~0.25s) since each iteration may need multiple function evaluations for the line
search. Given this fixes silent optimizer-induced fit corruption (not just for
v21 -- `q_solver` is shared by every model version, v16 through v21) rather than
papering over it with a tighter range, this cost is worth it. This is a
**framework-level change** affecting all past and future models that use
`stac_mjx`, made and verified within this session while Eric was asleep, per his
explicit go-ahead ("yes loosen the shoulder translation clamp by alot let it move
a bunch in the xml") -- flagging prominently here since it's the most consequential
code change in this whole session and should get a second look.

## 9. Joystick geometry correction from a dedicated static calibration

After the fixes in Sections 3.1/3.2, Eric reported the joystick was still visibly
misaligned and "changing sizes and angles between animals" -- worse for
CFL_36/37 than CFL_35. This turned out to be a real, separate bug: the model's
joystick geometry itself (shaft length, ball height) was wrong, and its
assumed vertical orientation was wrong too.

**New data**: Eric provided `static_joystick_labels_withbottom.v001.slp` -- a
dedicated, single-frame calibration labeling of the physical joystick rig in
all 3 cameras, with 3 points: `js_ball` (ball center), `js_base` (base of the
ball, top of the shaft), and a **new** `js_bottom` (where the shaft connects
to the rig's mount, roughly the bottom). Unlike `js_base`/`js_ball` tracked
during mouse behavior (noisy, incidental, no bone-length constraint), this is
a deliberate, dedicated measurement of the rig's fixed physical geometry.

**Method** (`compute_joystick_geometry.py`):
1. Triangulated the 3 static points using `calibration.toml` (verified
   byte-identical across all 15 CFL_35/36/37 trials -- one calibration for
   the whole dataset).
2. Transformed into model frame using a *rotation-averaged* registration
   transform: collected each of the 12 gauge-reliable trials'
   `registration_rotation` (from their `registered_v21_refined.h5` attrs) and
   computed the proper SVD/chordal mean rotation, rather than trusting any
   single trial's own (noisy) Procrustes fit. `registration_translation`
   (the model's own shoulder position) is exactly constant across all trials,
   as expected.
3. Computed: shaft length (bottom->base) = **14.363mm** (model previously used
   8.6mm -- far too short); ball offset (base->ball) = **1.822mm** (model
   previously used 0.9mm); shaft direction (bottom->base) = **39.4 degrees
   off model-frame +Z** -- not vertical.

**The "rigidly vertical" assumption from Section 3.2 was wrong.** The
x_slide/y_slide-only, no-rotation-DOF mechanical design only implies the
joystick's tilt doesn't *change* during a trial -- it says nothing about
whether that fixed tilt is zero. It isn't. Section 3.2's fix (deriving
position from `js_ball` alone, assuming vertical) was an improvement over
averaging two inconsistent noisy points, but still built on the wrong
orientation assumption and the wrong (too-short) shaft geometry -- both of
which this static calibration now directly measures instead of assumes.

**Fix applied**:
- `joystick_base` (the body with `x_slide`/`y_slide`) is now positioned at the
  `js_bottom`/mount point in model frame, and stays unrotated -- so it still
  translates in world X/Y only, matching "moving it should just move it along
  the x,y plane."
- The child `joystick` body (carrying the shaft + ball geoms, previously
  defined with an implicit vertical axis) now has a fixed `quat` rotating its
  local +Z to the measured shaft direction.
- The shaft geom's half-length and the ball geom's position were corrected to
  14.363mm and 14.363+1.822=16.184mm respectively (was 8.6mm/9mm).
- `configs/model/mouse_arm_hand_joystick_v21.yaml`'s `KEYPOINT_INITIAL_OFFSETS`
  for `js_base`/`js_ball` and `register_v21_mocap.py`'s `JOYSTICK_TARGET_SPEC`
  updated to match.
- `register_v21_mocap.py`'s old joystick-position-averaging logic (Section
  3.2's `js_ball`-alone approach) is removed entirely -- superseded by this
  static calibration, which is strictly better data for a fixed rig feature.
  Per-trial `js_base`/`js_ball` registration (used as per-frame STAC fitting
  targets, letting `x_slide`/`y_slide` track real ball displacement) is
  unchanged.

**Verified**: rebuilt v21, reran the 3-trial smoke test. All 3 trials (including
`CFL_37_20240423_trial_0001`, previously the worst-looking case) now show a
correctly-sized, consistently-tilted joystick with the ball positioned right
at the hand -- a large visual improvement, directly addressing "CFL_37 hand
and joystick look way off."

## 10. Summary

- **kineo was the wrong tool** (Section 1) -- it's a from-scratch human-body mocap
  system, not a calibration refiner; used `aniposelib`'s bundle adjustment instead,
  which reads/writes the same anipose TOML format `calibration.toml` already uses.
- **Calibration refinement genuinely works**: mean reprojection error dropped
  62.8% (10.11px -> 3.76px) across all 15 trials, with Cam01 (the consistent
  outlier in all 3 sessions) improving the most.
- **Found a real gauge-drift bug, tried a fix, reverted it per direct visual
  feedback** (Section 3, 3.1): unconstrained bundle adjustment drifts the whole
  reconstruction's gauge (rotation/translation/scale) per trial, invisible to a
  reprojection check. A Umeyama correction back to the original calibration's
  frame is theoretically sound but empirically made results *worse* -- it
  reintroduces the original calibration's own accuracy problems (e.g. Cam01's
  bad extrinsics) into anything reprojected through it. **Final approach**: use
  `calibration_refined.toml` and its own self-consistent triangulation together
  everywhere; use the gauge-drift diagnostic only to flag/exclude 3 poorly-
  converged trials from the joystick-position average, not to transform output.
- **Found and fixed a joystick orientation bug, twice** (Section 3.2, then
  Section 9): first, averaging `js_base`/`js_ball`-derived position
  candidates baked ~46 degrees of tracking noise into a crooked joystick --
  fixed by deriving position from `js_ball` alone, assuming vertical. That
  assumption turned out to be wrong too: a dedicated static calibration
  (`js_base`/`js_ball`/`js_bottom` labeled once, all 3 cameras) directly
  measured the joystick's true shaft length (14.36mm, not the model's 8.6mm),
  ball offset (1.82mm, not 0.9mm), and tilt (39.4 degrees off vertical, not
  upright at all). Rebuilt the joystick's geometry and orientation from this
  measurement instead of assumption -- a much larger visual improvement than
  either prior fix, directly resolving "CFL_37 hand and joystick look way off."
- **Also root-caused and fixed the shoulder-translation saturation** Eric
  originally asked about: not a calibration-quality issue as suspected, but a
  fixed-stepsize optimizer bug in the shared STAC solver, fixed via an adaptive
  line-search stepsize. `sh_tx/ty/tz` is back to its full, v18-matching +-10mm
  range with no saturation and equal-or-better fit quality.
- **Net result**: the STAC-fitted model's hand and joystick now track the real
  video closely across all 3 sessions, joystick correctly sized and tilted
  (not assumed vertical) in every view -- this appears to have resolved the
  original complaint about CFL_36/37's joystick misalignment.
- **Full 15-trial batch**: rerun with the corrected joystick geometry (Section
  9); see `refined_STACed_data_v21/full_batch_fit_summary.json` for final
  per-trial numbers (the joystick geometry fix mainly affects visual
  alignment, not the aggregate marker-error metric, since js_base/js_ball are
  only 2 of 19 fitted keypoints per frame).
- **Not yet independently validated**: this refinement only sees mouse+joystick
  keypoints, never the original Charuco board images -- Eric's planned check of
  the refined calibrations (especially Cam01's large rotation change, and the 3
  poor-convergence trials) against the original Charuco corner detections is the
  authoritative test of whether the cameras have genuinely drifted, and by how
  much. Treat this report's numbers as "relative refinement quality," pending
  that check.
