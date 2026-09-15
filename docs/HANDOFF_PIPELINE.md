# Pose → STAC/IK Pipeline Handoff (CFL joystick-reaching dataset, v22 / v25 / v26)

**Written:** 2026-08-19, by Eric Leonardis, for whoever picks this project up next.

This document explains, end-to-end and with exact file paths, how raw 2D pose
data becomes the STAC-fitted IK output in
`/root/vast/eric/stac-mjx/refined_STACed_data_v22/` and
`/root/vast/eric/stac-mjx/refined_STACed_data_v25/` — the mouse arm+hand+joystick
reaching dataset (sessions `CFL_35`, `CFL_36`, `CFL_37`, 5 trials each = 15
trials total). It also documents the newer `v26` fix that should probably be
your actual starting point (see §5.5), and lays out what's still broken and
worth fixing next, especially the "model stands upright" problem.

Read this together with the two detailed engineering logs already in this repo:
- `CALIBRATION_REFINEMENT_REPORT.md` — the full story of calibration refinement
  and joystick placement debugging (referenced heavily below; read it for the
  "why", this doc is more of a "where/what/how").
- `STAC_EMG_DIAGNOSTIC_REPORT.md` — a related but separate diagnostic on an
  older EMG/reach dataset (A36/AT00x sessions, see §12), not the CFL dataset.
- `MOCAP_STAC_PIPELINE.md` — documents a *different*, older pipeline (the
  `azim-emg-data` A36/AT006/AT009/AT012/AT013 reach sessions). It is **not**
  the CFL joystick pipeline this doc covers, but the reprojection-video
  machinery it uses (`reproject_stac_model.py`) is the same code.
- `PLAN_supination.md` — historical design doc for adding a forearm-supination
  hinge joint + knuckle keypoints to an earlier arm-only model
  (`mouse_forelimb_right_janelia_moving_shoulder.xml`, 7 DOF, no supination).
  The `forearm_supination` hinge it proposes is exactly what shows up later in
  the v24/v25/v26 skeletons (`build_v24_stac_xml.py`'s source model already has
  a `forearm_supination` joint on `radius_right`) — read this if you need to
  understand *why* that joint exists or want to add similar rotation-disambiguating
  keypoints elsewhere (e.g. for the torso fix in §11.1).
- `README.md` — top-level package install/usage instructions (conda env,
  `stac_mjx.load_configs`/`load_data`/`run_stac` basic API, headless rendering
  setup via `MUJOCO_GL=osmesa`). Read this first if setting up the environment
  from scratch on a new machine.

---

## 0. Background & concepts (read this first if the pipeline is unfamiliar)

Everything below this section is file-by-file "where/what/how." This section is
the "why" — the mental model that makes the rest of the document make sense.
Skip it if you're already comfortable with MuJoCo, multi-camera triangulation,
and STAC-style pose fitting.

### 0.1 The big picture: pixels in, joint angles out

At the highest level, this pipeline turns 2D pixel coordinates of the animal's
keypoints (from 3 synchronized cameras) into a full time series of MuJoCo joint
angles (`qpos`) for a rigged skeleton model. The data passes through five
conceptually distinct representations, in order:

1. **Pixel coordinates** (2D, per camera) — where SLEAP thinks a keypoint
   (e.g. `Wrist`) is in each camera's image, in pixels. This is
   `*.arm/hand.predictions.slp` → `*_merged.slp` (§2.1, §2.3).
2. **Triangulated 3D, "calibration frame"** (millimeters) — combine the 3
   cameras' 2D detections with known camera geometry (calibration) to solve
   for one 3D point per keypoint per frame (§0.4). This is
   `*_points3d[_refined].h5`. The coordinate frame here is **arbitrary** — it's
   whatever frame the Charuco calibration board happened to define; it has no
   relationship to the MuJoCo model's own coordinate frame.
3. **Registered 3D, "model frame"** (meters) — rotate/scale/translate the
   calibration-frame 3D points so they line up with a *specific* MuJoCo
   model's own rest-pose geometry (e.g. that model's own idea of where the
   shoulder sits and how long the arm bones are). This is
   `*_registered_v{N}_refined.h5` (§5), and it is **model-version-specific**
   — see §0.3.
4. **STAC/IK optimization** — given registered mocap in model frame, solve for
   (a) a fixed 3D marker offset per keypoint (where a marker sits relative to
   its parent bone, once per trial) and (b) per-frame joint angles that make
   the model's marker positions (offset + forward kinematics from `qpos`)
   match the mocap as closely as possible. This is what `stac_mjx` itself does
   (§7).
5. **`qpos` time series** — the actual output: one MuJoCo joint-angle vector
   per frame, in `*_ik.h5`. Reprojection videos, downstream kinematic
   analysis, and any RL work all consume this, not the raw mocap.

Steps 1–3 happen **outside** `stac-mjx`, in `luc3d-bugs` and its adjacent
SLEAP/CVAT projects (§2). Steps 4–5 are what living inside this repo's
`stac_mjx/` package does. Keeping straight *which stage* a given `.h5`/`.slp`
file represents is the single most useful thing to internalize before editing
anything — most of the "which frame is this file in" confusion in
`CALIBRATION_REFINEMENT_REPORT.md` traces back to conflating stages 2 and 3.

### 0.2 MuJoCo/mjx concepts used throughout this doc

If you haven't used MuJoCo before, a few terms recur constantly:

- **Body** — a rigid link in the kinematic tree (e.g. `humerus_right`), with a
  position/orientation defined relative to its parent body.
- **Joint** — a degree of freedom (DOF) connecting a body to its parent: e.g.
  a `hinge` (1 rotational DOF, with a `range` in the XML) or `free` (6 DOF,
  translation + rotation, used for a body not rigidly attached to anything).
  **A body with no `<joint>` element at all is welded rigidly to its parent —
  it cannot move independently.** This is exactly the mechanism behind the
  "v25 stands upright" issue in §11.1: every spine body has zero joints, so
  none of them can move, full stop — it isn't a fitting failure, there is
  simply no DOF there for anything to fit.
- **Site** — a massless reference point attached to a body, used as the
  attachment point for a tracked marker/keypoint. STAC fits a per-keypoint 3D
  offset from a body's origin to where that keypoint's marker actually sits
  in real life (`KEYPOINT_INITIAL_OFFSETS` in the yaml configs is the initial
  guess for this; §7 is where it gets refined).
- **`qpos`** — the vector of every joint's current value (an angle for each
  hinge, a distance for each slide, 7 numbers for a freejoint) that fully
  determines the model's pose. This is the thing STAC solves for, frame by
  frame — it is the main content of `*_ik.h5`.
- **Forward kinematics (FK)** — given a `qpos`, compute where every
  body/site ends up in world space. STAC's optimizer calls FK many times per
  step and compares the resulting site positions against the registered
  mocap targets.
- **`mjx` vs. `mujoco`** — `mujoco` is the standard C physics engine; `mjx` is
  a JAX reimplementation used here specifically so FK is auto-differentiable
  and can run inside a gradient-based optimizer (`jaxopt`), batched over many
  frames at once. STAC never calls `mj_step` / simulates physics — it is
  purely a *kinematics* fitting problem, which is why `<tendon>`/`<actuator>`
  blocks are stripped out of every STAC-safe XML (§5.5): they're irrelevant
  to (and in some cases incompatible with) a pure-FK optimization.

### 0.3 Why does every model version need its own registration frame?

This is the single easiest mistake to make (see the "critical lesson" callout
in §4), so it's worth explaining rather than just asserting. A MuJoCo model's
"rest pose" — its geometry when every joint is at `qpos = 0` — is set by
however the source Blender/anatomical-atlas rig happened to be authored, and
that differs between model versions (different artists, different export
passes, different body trees). Registration (§5) works by rotating, scaling,
and translating the *same* real-world mocap points so that they best match
*one specific model's own* rest-pose geometry — it's a per-model-version best
fit, not a universal transform. Because v21's rest pose and v22's rest pose
happen to be oriented roughly 87° apart in this arbitrary "model space," the
very same physical elbow position ends up at completely different (x, y, z)
coordinates once registered into each model's frame. **A registered
position/rotation computed for one model version is meaningless — not just
approximately wrong, but arbitrary — in another model version's frame.**
Always recompute registration (and anything downstream of it, like the
joystick anchor in §4) per model version; never copy a computed value across
versions. Two real, previously-shipped bugs (documented in
`CALIBRATION_REFINEMENT_REPORT.md`) came from exactly this mistake.

### 0.4 Camera calibration & triangulation, briefly

- **Intrinsics** describe a single camera's own lens geometry (focal length,
  principal point, lens distortion) — independent of where the camera
  physically sits.
- **Extrinsics** describe where a camera sits and points in 3D space: a
  rotation (`rotation`, stored as a **Rodrigues vector** — a compact
  3-number encoding of a 3D rotation, where the vector's direction is the
  rotation axis and its magnitude is the rotation angle) and a translation.
- **Triangulation** — given the same physical point's 2D pixel location in
  ≥2 cameras with known intrinsics + extrinsics, solve for the single 3D
  point whose reprojection into every camera best matches the observed 2D
  pixels. `aniposelib`'s `CameraGroup.triangulate()` does this via
  least-squares over all cameras simultaneously (more robust than naive
  pairwise triangulation, and it degrades gracefully if one camera's
  detection is missing/low-confidence for a frame).
- **Charuco board** — a calibration target combining a chessboard pattern
  with unique per-square ArUco markers. Because the real board's square size
  (in mm) is known, solving for the camera geometry that correctly
  reprojects the board's known corner layout pins down **true metric scale**
  for the whole rig. This Charuco step is the *only* place absolute,
  physical-world scale enters the entire pipeline (§2.2) — every later
  transform (bundle adjustment, registration) is supposed to preserve that
  scale, and when one doesn't, that's exactly the "gauge drift" bug in §3.
- **Bundle adjustment** — jointly re-optimizing camera extrinsics (and
  optionally intrinsics) together with 3D point positions, to minimize total
  reprojection error across every camera and frame at once, rather than
  trusting the original single-shot Charuco calibration forever. This is
  what `refine_calibration.py` does (§3) — but critically, it uses only the
  *animal's own tracked keypoints* as correspondences, with no Charuco board
  in the loop, which is precisely why it has no way to anchor absolute
  scale/rotation/translation and can "gauge drift" (§3, §0.5).

### 0.5 Glossary

| Term | Meaning here |
|---|---|
| STAC | The two-phase fitting approach this pipeline uses to turn markerless mocap into kinematics: fit fixed per-marker offsets once per trial, then solve per-frame joint angles ("Simultaneous Tracking And Calibration"-style approach, as used broadly in markerless rodent motion-capture work). |
| IK | Inverse kinematics — given target marker positions, solve for the joint angles that reproduce them via forward kinematics. |
| DOF | Degree of freedom — one independent way a joint can move (one hinge angle, one slide distance, one of a freejoint's 6 components, etc). |
| SLEAP | The 2D pose-estimation tool used to detect keypoints in video frames; also the name of its project/label/prediction file format (`.slp`). |
| CVAT | A general-purpose video/image annotation tool; used here to hand-label the arm keypoints before converting them to SLEAP format (§2.3). |
| (Orthogonal) Procrustes | Finding the best-fit pure rotation (no scaling or shearing) that aligns one set of 3D points onto another — used in registration (§5) to align mocap onto a model's rest pose. |
| Rodrigues vector | A 3-number representation of a 3D rotation (direction = axis, magnitude = angle); the convention MuJoCo/OpenCV/aniposelib calibration files use. |
| `qpos` | MuJoCo's joint-position state vector — the thing STAC solves for, per frame. |
| `mjx` | The JAX-based reimplementation of MuJoCo used here for differentiable, batchable forward kinematics (see §0.2). |
| Bundle adjustment | Joint optimization of camera parameters and 3D point positions together, to minimize total reprojection error. |
| Gauge drift / gauge freedom | A family of equally-valid solutions that differ by some transform (here: a 3D similarity transform — rotation + translation + scale) that the optimization's own objective (reprojection error) is mathematically blind to, so nothing in the optimization prevents drifting along that transform. See §3 for the concrete case. |

---

## 1. The repos you need

| Repo | Role |
|---|---|
| `/root/vast/eric/stac-mjx/` | This repo. STAC/IK fitting code (`stac_mjx/`), model-building scripts, run scripts, config, and all `refined_STACed_data_*` outputs. |
| `/root/vast/eric/luc3d-bugs/js_luc3d_hand/` | **Source of truth for everything pose/calibration-related.** SLEAP 2D detections, camera calibration, triangulated 3D points, per-trial registration outputs — **and** the actual arm and hand+joystick SLEAP labeling/training projects and model checkpoints, all under `trainingData/` (see §2.3). |
| `/root/vast/eric/CVAT_mouse_reach/` | A separate, larger CVAT-labeled arm project. Looks like it should be "the" arm model source, but verified (§2.3) to **not** be what actually produced any of this dataset's `.arm.predictions.slp` files — included here only so you don't go down that path by mistake. |
| `/root/vast/eric/janelia_model/` | Source MuJoCo model XMLs (musculoskeletal atlas exports from the Janelia/vnl-playground modeling side) that `build_v*_stac_xml.py` scripts in stac-mjx read from and adapt for STAC. |

---

## 2. Raw data layout (`luc3d-bugs`)

Root: `/root/vast/eric/luc3d-bugs/js_luc3d_hand/`

```
js_luc3d_hand/
  trainingData/                  ← THE data used by v22/v25/v26 (15 CFL_35/36/37 trials + 5 newer CFL_44 trials, unused so far — see §10)
  trainingData_eric/              ← identical duplicate of trainingData/ (same trial set) — appears to be a personal working copy, not a separate dataset
  scripts/                        ← misc video-conversion / EMG-matching scripts, not part of the STAC pipeline
  matched_EMG_v25/                ← EMG data matched to v25 IK output (separate downstream analysis, not covered here)
  original_data_CFL_35/           ← raw/pre-processing source for CFL_35 (didn't need to touch this for v22/v25)
  handSkeleton.json, Use_Skeleton.json  ← SLEAP skeleton definitions (node names/edges) used when labeling 2D hand keypoints
```

### 2.1 Per-trial directory (the unit everything operates on)

Example: `/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData/CFL_36_20231220_trial_0001/`

```
CFL_36_20231220_trial_0001/
  Cam00/
    CFL_36_20231220_trial_0001_Cam00.mp4                      ← ORIGINAL VIDEO for this camera (use for visual eval)
    CFL_36_20231220_trial_0001_Cam00.arm.predictions.slp      ← raw SLEAP 2D predictions, arm skeleton (Shoulder/Elbow/Wrist)
    CFL_36_20231220_trial_0001_Cam00.hand.predictions.slp     ← raw SLEAP 2D predictions, hand skeleton (digit keypoints)
    CFL_36_20231220_trial_0001_Cam00_merged.slp                ← arm+hand+joystick merged into one 19-keypoint SLP (via merge_arm_hand.py)
  Cam01/  (same structure)
  Cam02/  (same structure)
  calibration.toml                                             ← ORIGINAL anipose-format camera calibration (Charuco-board derived), shared/identical across all 15 CFL_35/36/37 trials
  calibration_refined.toml                                     ← bundle-adjustment-refined extrinsics (see §3), per-trial (NOT identical across trials — this is per-trial refined)
  CFL_36_20231220_trial_0001_calib_refine_stats.json           ← before/after reprojection error for this trial's refinement
  CFL_36_20231220_trial_0001_points3d.h5                       ← raw triangulated 3D points (original calibration.toml), (T,1,19,3) mm
  CFL_36_20231220_trial_0001_points3d_refined.h5               ← 3D points re-triangulated with calibration_refined.toml — THE ONE USED DOWNSTREAM
  CFL_36_20231220_trial_0001_registered_v21_refined.h5          ← mocap registered into v21 model frame
  CFL_36_20231220_trial_0001_registered_v22_refined.h5          ← mocap registered into v22 model frame — INPUT to v22's STAC run
  CFL_36_20231220_trial_0001_registered_v25_refined.h5          ← mocap registered into v24/v25 model frame — INPUT to v25's STAC run
  CFL_36_20231220_trial_0001_joystick_recorded.slp              ← per-trial joystick tracking labels (js_base/js_ball) feeding into merge/triangulation
  CFL_36_20231220_trial_0001_joystick_alignment_check.mp4       ← QA video
  CFL_36_20231220_trial_0001_triangulation_check.mp4            ← QA: green=2D SLEAP detection, red=reprojected 3D triangulation, original calibration
  CFL_36_20231220_trial_0001_triangulation_check_refined.mp4    ← same QA, refined calibration — use THIS one to sanity check new data
```

19-keypoint order used throughout (`merge_arm_hand.py`'s construction, confirmed via `points3d.h5`'s `n_nodes=19`):
```
Shoulder, Elbow, Wrist,
D1_K, D1_T, D2_K, D2_M, D2_T, D3_K, D3_M, D3_T, D4_K, D4_M, D4_T, D5_K, D5_M, D5_T,
js_base, js_ball
```
(`Dn_K` = knuckle/MCP, `Dn_M` = mid/PIP, `Dn_T` = fingertip; digit 1/thumb has no mid point.)

`merge_arm_hand.py` (the script that builds `Cam0X_merged.slp` from the separate arm/hand SLEAP predictions files) lives at:
```
/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData/merge_arm_hand.py
```

### 2.2 Camera calibration

- **Original**: `calibration.toml` per trial, anipose/aniposelib TOML format (`cam_0`/`cam_1`/`cam_2` sections: `matrix`, `distortions`, `rotation` (Rodrigues rvec), `translation`). Derived from a **Charuco board** calibration — the *only* thing in the whole pipeline anchoring absolute metric scale. Verified byte-identical across all 15 CFL_35/36/37 trials (one shared rig calibration for the whole dataset).
- **Refined**: `calibration_refined.toml` per trial, produced by `/root/vast/eric/stac-mjx/refine_calibration.py` — see §3 for the full method and why it's per-trial (extrinsics only; intrinsics untouched).
- Loader used by the STAC/reprojection code: `stac_mjx/multicam.py::load_calibration()` — reads this exact anipose TOML convention.

### 2.3 How the 19-keypoint 2D detections were produced: SLEAP models, training labels, and the sleap-io merge

The 19-keypoint 2D detections in `<trial>_Cam0X_merged.slp` don't come from one model — they come from **two independently labeled and trained SLEAP projects** (arm, and hand+joystick), combined per-frame by a small `sleap_io` script. If accuracy in some viewing angle/behavior is poor, or you need to extend the keypoint set, this is where you'd add labels and retrain.

**Exactly which trained model produced a given `*.arm.predictions.slp` / `*.hand.predictions.slp` file — verified directly, not inferred from directory names.** Every SLEAP prediction file embeds its own provenance metadata: which checkpoint ran inference, the inference config, the timestamp, and the machine it ran on. This is ground truth and is worth checking directly whenever you're not sure what produced a given file:

```python
import sleap_io as sio
labels = sio.load_slp(path_to_predictions_slp)
print(labels.provenance["model_paths"])   # exact checkpoint directory used for inference
print(labels.provenance["model_type"])    # e.g. "single_instance"
print(labels.skeletons[0].node_names)     # keypoints that model predicts
```

Running this over **all 60** `*.arm.predictions.slp` and **all 60** `*.hand.predictions.slp` files under `trainingData/` (20 trials × 3 cameras — the 15 CFL_35/36/37 trials plus the 5 unused `CFL_44` trials, §11.5) shows one consistent mapping: **a separate `single_instance` model was trained per camera per body-part (6 checkpoints total), not one shared multi-view model**:

| Camera | Role | Checkpoint directory (under `trainingData/models/`) | Skeleton (node names) | Trained on (`trainingData/<file>`) | Train-split frames |
|---|---|---|---|---|---|
| Cam00 | arm | `260708_101218.single_instance.n=134` | `shoulder, elbow, wrist, hand` | `Cam00_Working_akira.slp` | 120 |
| Cam01 | arm | `260708_131506.single_instance.n=180` | `shoulder, elbow, wrist, hand` | `Cam01_Working_akira.slp` | 162 |
| Cam02 | arm | `260709_100153.single_instance.n=162` | `shoulder, elbow, wrist, hand` | `Cam02_Working_akira.slp` | 145 |
| Cam00 | hand+joystick | `260702_164636.single_instance.n=120` | 17-node hand skeleton (§2.1) | `Cam00_Working.slp` | 108 |
| Cam01 | hand+joystick | `260712_134207.single_instance.n=183` | 17-node hand skeleton | `Cam01_Working.slp` | 164 |
| Cam02 | hand+joystick | `260706_161230.single_instance.n=153` | 17-node hand skeleton | `Cam02_Working.slp` | 137 |

The actual model weights are the `best.ckpt` file inside each of those directories, e.g.
`/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData/models/260708_101218.single_instance.n=134/best.ckpt`
(~80–110MB each; UNet backbone, single-instance confmap head, `sleap_nn` v0.2.0 — see that directory's `initial_config.yaml`/`training_config.yaml` for the full architecture/augmentation config, and `training_log.csv`/`metrics.{train,val}.0.npz` for training curves).

**`trainingData/models/` is the actual model store** — no separate model repo to go hunting for. It holds 19 total training-run directories: the 6 above (the ones actually used for the shipped predictions) plus 13 earlier/abandoned attempts at other `n=<frame count>` values (e.g. `260701_160144.centroid.n=120` through `260712_131451.single_instance.n=183`) — dead-end iterations kept for history, not used by any current `*.predictions.slp`.

The **raw labeling projects** these checkpoints were trained from live directly in `trainingData/` (and are duplicated in `trainingData_eric/`, same pattern as noted above):
```
Cam00_Working_akira.slp / Cam01_Working_akira.slp / Cam02_Working_akira.slp   ← arm labels (shoulder/elbow/wrist/hand "akira" skeleton), one project per camera
Cam00_Working.slp / Cam01_Working.slp / Cam02_Working.slp                     ← hand+joystick labels (17-node skeleton), one project per camera
Cam0X_Working.backup.slp, Cam0X_Working_vids.slp, Cam0X_Working_akira_vids.slp  ← backups / video-embedded variants of the same projects
```
The 4-node "akira" skeleton (`shoulder → elbow → wrist → hand`) referenced by these labels matches `akira_skeleton.json`/`akira_test_session` naming seen elsewhere in `luc3d-bugs` — it is the **actual arm skeleton used for this dataset's predictions**, not a discarded variant.

**Correction to an earlier pass over this codebase**: `/root/vast/eric/CVAT_mouse_reach/` (CVAT-labeled arm data, converted via `sleap_io_Convert_CVAT_to_SLP.ipynb`, with its own much larger `models/` directory — `n=7730` through `n=14057` labeled frames) looks like a very plausible source for "the arm model" at first glance, and an earlier research pass over this repo reported it as such. **Direct inspection of the provenance metadata on all 60 `.arm.predictions.slp` files shows this is wrong** — none of them were produced by any `CVAT_mouse_reach` checkpoint. `CVAT_mouse_reach` is a separate labeling effort that, as far as the actual prediction files can show, was not used to produce the CFL_35/36/37/44 arm predictions this pipeline consumes. The real arm model is the smaller, native, per-camera "akira"-skeleton SLEAP model documented above. If you're chasing arm-tracking quality issues, retrain from `Cam0X_Working_akira.slp`, not `CVAT_mouse_reach`. (This is a good general lesson: when in doubt about provenance, check the file's own embedded metadata rather than trusting directory names or an earlier summary — including this one.)

**Why per-camera, single-instance models instead of one shared multi-view model?** Each camera sees the mouse/rig from a very different angle, and `single_instance` mode assumes exactly one visible animal instance per frame with no cross-camera identity matching needed at the 2D stage — matching detections across cameras is handled later, geometrically, by triangulation against the known camera calibration (§0.4), not by the pose model itself. Training one small model per camera's own viewing geometry is simpler to label for and avoids needing a single model to generalize across very different viewpoints.

**Merging arm + hand predictions into one 19-keypoint skeleton** — `/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData/merge_arm_hand.py` (identical copy in `trainingData_eric/`), using the `sleap_io` package (imported as `sio`):
1. For each trial × camera, `sio.load_slp(...)` the separate `<trial>_<cam>.arm.predictions.slp` and `<trial>_<cam>.hand.predictions.slp`.
2. Build a merged `Skeleton`: nodes = `["shoulder", "elbow"] + hand_skeleton.node_names` (19 total), edges = `shoulder→elbow→Wrist` plus the hand skeleton's own internal edges.
3. Per frame present in both: take `shoulder`/`elbow` xy+score from the **arm** model's instance, all 17 hand-model points from the **hand** model's instance — the arm model's own (less precise) wrist/hand predictions are discarded in favor of the hand model's `Wrist` — concatenate via `np.vstack`, wrap as a new `sio.PredictedInstance.from_numpy(...)`.
4. `sio.save_slp(...)` writes `<trial>_<cam>_merged.slp` — **this is the file everything downstream (`refine_calibration.py`, triangulation, §2.1) actually reads.**
5. **Gotcha on a fresh machine**: the script hardcodes `BASE = Path(r"Z:\eric\luc3d-bugs\js_luc3d_hand\trainingData")` (a Windows/SMB-mapped path) — change to `/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData` before rerunning on Linux.

**If you need more/better labels**: label additional frames in whichever project is weak (CVAT for arm, SLEAP directly for hand/digits/joystick), retrain with `sleap train` (see `train-script.sh`/`jobs.yaml` for the hand project's invocation pattern), run inference on your new videos to get fresh `*.arm.predictions.slp`/`*.hand.predictions.slp`, then rerun `merge_arm_hand.py` (with `BASE` fixed as above) to regenerate `*_merged.slp` before feeding it into §3 onward.

---

## 3. Calibration refinement (full detail: `CALIBRATION_REFINEMENT_REPORT.md`)

Script: `/root/vast/eric/stac-mjx/refine_calibration.py`

**Method** (per trial, all 15 CFL_35/36/37 trials):
1. Load 3 cameras' `Cam0X_merged.slp` (19 keypoints × 126 frames).
2. Load `calibration.toml` via `aniposelib.cameras.CameraGroup.load()`.
3. Baseline reprojection error against original calibration.
4. `CameraGroup.bundle_adjust_iter(p2ds, only_extrinsics=True)` — refine rotation/translation only; intrinsics left as Charuco-derived.
5. Re-triangulate → `<trial>_points3d_refined.h5`; write `calibration_refined.toml`.

**Result**: mean reprojection error dropped 62.8% (10.11px → 3.76px) across all 15 trials. Cam01 was a consistent ~2x-worse outlier in every session, and improved the most.

**Known caveat — "gauge drift"** (see also §0.4/§0.5 for the general concept): `bundle_adjust_iter` is only constrained by reprojection error, which is invariant to a joint similarity transform (rotation+translation+**scale**) applied to every camera and 3D point together. Concretely: if you took the true, correct set of cameras and 3D points and rotated all of them 10° around some axis, shifted them all by 1cm, and scaled everything by 1.001x — every camera-to-point ray is still pointing the same relative direction, so every 2D reprojection is still pixel-for-pixel identical, and `bundle_adjust_iter`'s loss function cannot tell the difference between that transformed solution and the original one. It has no reason to prefer one over the other, so which one it happens to converge to depends on essentially arbitrary optimizer initialization/path — that's "gauge drift": the calibration can silently rotate/shift/rescale by a few percent or degrees, per trial, with reprojection error staying just as good throughout. So `calibration_refined.toml`'s absolute scale/frame can drift trial-to-trial in a way invisible to reprojection error, and two different trials' `calibration_refined.toml` files are not guaranteed to agree on what "1 meter" or "straight up" means even though each is internally self-consistent. A Umeyama gauge-correction fix (rigidly re-aligning each trial's drifted frame back to a common reference, using `scipy`'s/a custom Umeyama similarity-transform solver) was tried and **reverted** — it made results visibly worse because it reintroduces the original (uncorrected) calibration's own errors, i.e. the "fix" was itself another source of exactly this kind of drift, just relative to a different, no-more-correct reference. Current approach: use `calibration_refined.toml` + its own self-consistent triangulation together, everywhere (never mix a refined calibration from one trial with triangulated points from another), and use the gauge-drift diagnostic only to **exclude** 3 poorly-converged trials from cross-trial averages (joystick position, rotation averaging):
```
EXCLUDED_TRIALS = {
    "CFL_35_20240128_trial_0001",
    "CFL_37_20240423_trial_0001",
    "CFL_37_20240423_trial_0201",
}
```
This exclusion set is hardcoded identically in `compute_joystick_geometry_v22.py`, `compute_joystick_geometry_v25.py`, and (implicitly, re-derived) `register_v25_mocap.py`.

**Still not done** (flagged in the report, §4 there): independent validation of the refined calibrations against the *original Charuco board corner detections* — the refinement above never sees the Charuco board, so it can only be trusted for *relative* per-trial accuracy, not for verifying the cameras haven't physically drifted over time. If picking this project back up, this is a good first thing to check, especially for Cam01 (large rotation change) and the 3 excluded trials.

---

## 4. Joystick (0,0) / geometry estimation — how the joystick's position is determined

This is the piece the user specifically asked about. It's **not** derived from per-frame mocap tracking of `js_base`/`js_ball` alone (that was tried first and found too noisy — see report §3.2) — it's derived from a **dedicated static calibration recording**:

```
/root/vast/eric/stac-mjx/static_joystick_labels_withbottom.v001.slp
```
A single frame, 3 cameras, 3 hand-labeled points on the physical joystick rig: `js_ball` (ball center), `js_base` (top of shaft / base of ball), `js_bottom` (where the shaft mounts to the rig — new in this recipe, added specifically to fix an orientation bug, see report §9).

**Method** (`compute_joystick_geometry_v22.py`, `compute_joystick_geometry_v25.py`, `compute_joystick_geometry_v26.py` — all near-identical, only the source registration `.h5` files differ per model version):
1. Triangulate the 3 static SLP points using the shared `calibration.toml` (`aniposelib.cameras.CameraGroup.triangulate`).
2. Average `registration_rotation` and `shoulder_reference_mm` across the reliable trials' own `*_registered_v{22,25}_refined.h5` attrs — a proper SVD/chordal rotation mean, not any single trial's noisy fit.
3. Transform the triangulated points into model frame using that averaged rotation.
4. Compute `shaft_length = |base - bottom|` (**14.363 mm**, constant across v21/v22/v25/v26 — it's a physical rig measurement), `ball_offset = |ball - base|` (**1.822 mm**, also constant), and `shaft_direction` (bottom→base, **not assumed vertical** — this was a prior bug, see report §3.2/§9).
5. Print values consumed **manually** (copy-pasted, not auto-wired) into `build_v{22,25,26}_stac_xml.py`'s hardcoded `JOYSTICK_BASE_POS` and the corresponding `configs/model/*.yaml`'s `KEYPOINT_INITIAL_OFFSETS` for `js_base`/`js_ball`.

**Critical lesson, repeated in every one of these scripts' docstrings — do not skip this if you add a new model version:**
> Each model version's world frame is its **own independent** Procrustes best-fit of the same real mocap data against that model's own, differently-postured rest pose. v21 vs v22's frames differ by ~87°, empirically. **Never copy a computed `joystick_base` position/quat from one model version's build script into another's** — always recompute via `compute_joystick_geometry_v{N}.py` pointed at that version's own `*_registered_v{N}_refined.h5` files. Two real bugs (report §3.2, and the v25→v26 fix below) were caused exactly by getting this wrong.

**v25 → v26 correction** (this postdates the v25 output the user asked about, and is directly relevant): `compute_joystick_geometry_v26.py` found that v25's static-calibration-photo-derived anchor was systematically off by 5–9 mm in *every* trial — a uniform bias — compared to each trial's own `js_base_resting_position` attribute (already computed and stored by `register_v25_mocap.py`, from real per-trial registered mocap, not the separate static photo). Since `joystick_base` only has ±6 mm of `x_slide`/`y_slide` travel, this bias pinned the slide joints at their limit on most trials. **Fix**: v26 anchors `joystick_base` on the **mean of the 12 reliable trials' own `js_base_resting_position`** (from `register_v25_mocap.py`'s output, already in v25's model frame) instead of re-deriving from the static photo. This drops residual scatter to ~0.2–2.7mm. **If you're building on v25, check `refined_STACed_data_v26/` first — it's a strict improvement on the joystick placement specifically, using the same v24/v25 skeleton.**

---

## 5. Mocap → model-frame registration

**Why this step has to exist at all**: the 3D points coming out of triangulation (§3) live in the calibration board's arbitrary coordinate frame, in millimeters — a frame that has no relationship whatsoever to how the MuJoCo model's own rest pose happens to be oriented and scaled in its own XML coordinate frame, in meters (§0.1 stages 2 vs. 3, §0.3). STAC's optimizer (§7) needs its *target* mocap points to already be in the model's own frame — it isn't solving for that alignment itself, only for joint angles given an already-aligned target. Registration is the step that produces that alignment, once per trial, before STAC ever runs.

Scripts: `register_v21_mocap.py`, `register_v22_mocap.py`, `register_v24_mocap.py`, `register_v25_mocap.py` (one per model version, because — per §0.3 — each model version's rest pose defines a different target frame; **read `register_v25_mocap.py`'s docstring first, it's the most complete/current recipe**).

Per trial, `<trial>_points3d_refined.h5` (raw triangulated, mm, calibration frame) → `<trial>_registered_v{N}_refined.h5` (meters, model/world frame, same 19-keypoint order), with these steps (see `register_v25_mocap.py::prepare_trial`/`register_trial` for the actual code):
1. Fill NaNs (linear interpolation), Savitzky-Golay smooth (window=15, polyorder=3).
2. Find a low-velocity "resting window" in the first ~60 frames (combined Wrist+Elbow speed, 25th percentile threshold).
3. Center on `shoulder_reference_mm` (median over resting window), convert mm→m.
4. **Per-trial own scale factor**: rescale uniformly so median resting arm span (shoulder→elbow→wrist) matches the model's own rest-pose arm span. This uniform rescale is why `js_base`/`js_ball` positions are NOT independently bone-rescaled — they ride along on the same rotation+scale+translation as the arm/hand fit, which is why joystick placement is specifically sensitive to registration quality (report §3, "why gauge drift broke the joystick, not the arm/hand").
5. **Per-trial own Procrustes rotation** (`scipy.linalg.orthogonal_procrustes`) fit on the 17 arm/hand keypoints only (not js_base/js_ball) against the model's own rest-pose target positions. (Earlier recipes shared one averaged rotation across all trials — found to be wrong, since per-trial calibration frames diverge 8-21° even within a session; see `build_v22_stac_xml.py`'s docstring.)
6. `js_ball` per-frame position is stabilized: replaced with `js_base + median(early-frame js_ball - js_base offset)` (raw per-frame js_ball tracking under hand occlusion is unreliable).
7. Write output `.h5` with `tracks` dataset + attrs: `registration_rotation`, `registration_translation` (= model's own shoulder position, constant across trials), `shoulder_reference_mm`, `scale_factor`, `js_base_resting_position`, `js_ball_resting_position`, `max_bone_length_error_m`, `recipe` (string tag), etc.

Model targets (`KP_TARGET_SPEC`) for v25/v24: `Shoulder→humerus_right`, `Elbow→ulna_right`, `Wrist→N_L_C_right` (fused carpal complex), digit K/M/T → `Phalanx_hand_{1,2,3}_{1..5}_right` bodies, with small nonzero offsets for the T (fingertip) keypoints (e.g. `D1_T: [0, 0.00066, 0]`).

---

## 5.5 MuJoCo model lineage (which XML is which)

All STAC-safe model XMLs live in `/root/vast/eric/stac-mjx/models/`. Source (non-STAC-safe) XMLs live in `/root/vast/eric/janelia_model/v{N}/`.

| Version | Source | Body scope | Joystick? | Notes |
|---|---|---|---|---|
| v17/v18 | `janelia_model` | arm only | no | early iterations |
| v19 | " | + hand | no | first hand graft (see `[[v19 hand model rebuild]]` memory) |
| v21 | " | arm+hand | yes | first working joystick model; `sh_tx/ty/tz` shoulder-translation solver bug found & fixed here (report §8, framework-level fix in `stac_mjx/stac_core.py::q_solver`, affects **all** model versions) |
| **v22** | `janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22.xml` | arm+hand (v21 body names) | yes, cylinder geom | `build_v22_stac_xml.py`. Re-derived joystick placement in v22's own frame (§4) |
| v24 | `janelia_model/v24/forearm_v24.xml` | **full skeleton**: spine/skull/mandible/clavicle/scapula/arm/hand (49 bodies), Blender-exported musculoskeletal atlas | **no** (dropped) | `build_v24_stac_xml.py`. Body names get `_right` suffix; wrist target becomes `N_L_C_right` |
| **v25** | `models/mouse_forelimb_right_janelia_arm_hand_v24_stac.xml` (i.e. v24's STAC-safe output) | v24 body + joystick re-added | yes, **capsule** geoms (not cylinder — mjx collision support + cost reasons, see `build_v25_stac_xml.py` docstring) | `build_v25_stac_xml.py`. Joystick untilted by design (stability lesson from v22) |
| v26 | same base as v25 | same as v25 | yes | `build_v26_stac_xml.py` — **only** change from v25 is a corrected `joystick_base` anchor (§4). Probably your best starting point if extending the joystick-tracking work. |

Both `build_v22_stac_xml.py` and `build_v24/25/26_stac_xml.py` do the same generic prep: comment out `<tendon>`/`<actuator>` blocks (STAC is kinematics-only, `mjx.kinematics`, never calls `mj_step`), validate via `mjx.put_model()` at build time (catches unsupported collision-geom-pair combos like cylinder-vs-mesh before a smoke test wastes time discovering it), and set `meshdir` correctly for the new file location.

**v22 config**: `configs/model/mouse_arm_hand_joystick_v22.yaml` + `configs/stac/stac_mouse_arm_hand_joystick_v22.yaml`
**v25 config**: `configs/model/mouse_arm_hand_joystick_v25.yaml` + `configs/stac/stac_mouse_arm_hand_joystick_v25.yaml`

Both configs: 19 `KP_NAMES` (17 arm/hand + `js_base`/`js_ball`), `ROOT_OPTIMIZATION_KEYPOINT: Wrist`, `N_ITERS: 6`, `N_SAMPLE_FRAMES: 50`, `SCALE_FACTOR/MOCAP_SCALE_FACTOR: 1.0` (no additional rescaling beyond what registration's own per-trial `scale_factor` already applied, §5), `n_fit_frames: 126` (CFL trials are fixed 126-frame / 5.04s@25fps clips), `mujoco.solver: newton`.

Two more config keys worth knowing about, since they control how forgiving the offset fit is (§7): `KEYPOINT_INITIAL_OFFSETS` (the starting guess for each keypoint's marker-to-body offset, in meters, refined during `fit_offsets` — e.g. fingertip offsets around `0.0007–0.0015` m along the body's local +Y, joystick offsets `js_base: [0,0,0.014363]`/`js_ball: [0,0,0.016184]` from §4) and `SITES_TO_REGULARIZE` + `M_REG_COEF` (a list of keypoints whose fitted offset gets pulled back toward its initial guess by a penalty of weight `M_REG_COEF`, to keep low-confidence/noisy keypoints — currently just the fingertips and `js_base`/`js_ball` — from drifting to an implausible offset during fitting). **`M_REG_COEF` is currently `0` in both v22 and v25's configs**, i.e. regularization is defined but switched off — offset fitting is currently fully unconstrained for every keypoint. If a future version's fingertip/joystick offsets look implausible (e.g. a fingertip marker offset that ends up inside a different finger), turning this back on for the noisier keypoints is the first knob to try before assuming it's a mocap or model problem.

---

## 6. Running STAC + IK

Run scripts (one per model version, all near-identical): `run_stac_v22_smoke.py`, `run_stac_v25_smoke.py`, `run_stac_v26_smoke.py`.

```bash
cd /root/vast/eric/stac-mjx
.venv/bin/python run_stac_v22_smoke.py       # produces refined_STACed_data_v22/
.venv/bin/python run_stac_v25_smoke.py       # produces refined_STACed_data_v25/
# N_FRAMES_LIMIT=15 .venv/bin/python run_stac_v25_smoke.py   # fast smoke test on a 15-frame clip first
```

What each script does, per trial (all 15 `CFL_35/36/37_*` dirs under `luc3d-bugs/.../trainingData/`):
1. Requires `<trial>_registered_v{N}_refined.h5` to already exist (built by `register_v{N}_mocap.py` — run that first if missing).
2. `kp_data, sorted_kp_names = stac_mjx.load_mocap(cfg, base_path)` — loads the registered h5.
3. `fit_path, ik_path = stac_mjx.run_stac(cfg, kp_data, sorted_kp_names, base_path=out_dir)` — this is the actual STAC/IK call (see §7 for internals). Writes `<trial>_fit.h5` (offset-fitting phase) and `<trial>_ik.h5` (per-frame IK, the main output).
4. Runs QA checks: joint-limit saturation (`check_qpos_ranges`), marker fit error in mm (`marker_error_stats`), fingertip-to-joystick-ball min distance (`hand_joystick_min_distance_mm`).
5. Calls `stac_mjx.reproject_stac_model.render_reprojection_video(...)` (§9) → `<trial>_reproj_v{N}_smoke.mp4`.
6. Dumps all per-trial QA numbers to `v{N}_smoke_fit_summary.json` at the output root.

### 7. What `stac_mjx.run_stac()` actually does internally

**The core problem STAC is solving**, in one sentence: mocap gives you where a keypoint like `Wrist` is in 3D space every frame, but it doesn't tell you *where on the model's wrist body* that marker actually sits, nor what joint angles produced that position — both have to be inferred jointly, and STAC splits that into two phases specifically because the marker-offset part should be constant across a whole trial (a physical marker/labeled-point doesn't move relative to the bone it's tracking) while joint angles are different every single frame. Solving both simultaneously per-frame would be heavily underdetermined (many offset+angle combinations explain a single frame equally well); fixing the offset first, using information from every frame in the trial at once, then solving only for angles per frame, breaks that ambiguity.

Entry point: `stac_mjx/main.py::run_stac(cfg, kp_data, kp_names, base_path)`. Two phases:
- **`fit_offsets`** (`Stac.fit_offsets`, uses `cfg.stac.n_fit_frames` frames — 126 here, i.e. the whole clip) — fits a single fixed per-keypoint 3D offset (marker position relative to its parent body, in that body's local frame) that has to work across *all* those frames simultaneously, starting from the initial guesses in `KEYPOINT_INITIAL_OFFSETS` (and pulled back toward that guess by `SITES_TO_REGULARIZE`/`M_REG_COEF` when regularization is enabled — see §5.5). Saved to `<trial>_fit.h5`.
- **`ik_only`** (`Stac.ik_only`) — per-frame inverse kinematics: with the offsets now fixed, solve independently for each frame's joint angles (`qpos`) that make the model's marker positions (parent-body FK position + that fixed offset) match the mocap target as closely as possible. Saved to `<trial>_ik.h5`, which contains `qpos`, `kp_data` (mocap input, MuJoCo frame), `marker_sites` (FK-computed site positions for comparison — this is literally "where the model thinks each keypoint is," useful for computing your own error metrics), `offsets`, and (if `infer_qvels: true`) `qvel`.

**What the loss functions actually minimize**: both phases minimize a sum-of-squared-distances error between predicted marker position (FK site position + offset) and the mocap target, summed over every tracked keypoint (and, for `fit_offsets`, over every sampled frame too), plus the offset-regularization penalty described above when it's turned on. There's no term rewarding a "natural-looking" pose beyond what the joint `range` limits in the XML already constrain — if a keypoint's mocap is noisy or wrong for a frame, the optimizer will still push the joint angles as far as needed (up to their range limits) to chase it, which is exactly the mechanism behind the "joint saturates at its range limit" symptom described throughout this document (§4's joystick slide-limit bug, §11.2's finger-joint ranges) — saturation is usually either a genuinely-out-of-range real pose, bad input data for that frame, or an optimizer issue, and distinguishing between those three is mostly what `CALIBRATION_REFINEMENT_REPORT.md`'s debugging effort consisted of.

Core optimizer: `stac_mjx/stac_core.py` — `q_opt`/`q_loss` (per-frame pose, `jaxopt.ProjectedGradient` — "projected" meaning each gradient step is clipped back into the joints' valid `range`, which is also *why* a saturating joint gets stuck exactly at its limit rather than overshooting past it) and `m_opt`/`m_loss` (offset fitting, same solver family). Both are implemented with `mjx` forward kinematics (§0.2) so the whole loss is differentiable end-to-end and JAX can batch/JIT-compile it. **Framework-level fix already applied** (report §8): `q_solver`'s stepsize was changed from a fixed `0.1` to `0.0` (switches `jaxopt.ProjectedGradient` from a fixed-size gradient step to adaptive backtracking line search — it tries progressively smaller steps along the gradient direction until the loss actually decreases, instead of always taking the same-size step regardless of whether that step helps) — this fixed a bug where any DOF shared across many keypoints at once (like `sh_tx/ty/tz`, which every arm/hand keypoint's position depends on) would saturate at its joint-range limit regardless of data quality, because the fixed step size was too large relative to that DOF's sensitivity and kept overshooting past the true optimum. Cost: ~2x runtime per frame, since line search evaluates the loss multiple times per step. This affects **every** model version that uses `stac_mjx`, not just v21+, since it's a fix in the shared library code, not any model-specific config.

---

## 8. Output structure

`refined_STACed_data_v22/` and `refined_STACed_data_v25/` (and `_v26/`) each contain one subdirectory per trial:
```
refined_STACed_data_v25/
  CFL_35_20240128_trial_0001/
    CFL_35_20240128_trial_0001_fit.h5             ← offset-fitting phase output
    CFL_35_20240128_trial_0001_ik.h5              ← per-frame IK: qpos, qvel, kp_data, marker_sites, offsets — THE main artifact
    CFL_35_20240128_trial_0001_reproj_v25_smoke.mp4  ← reprojection overlay video (see §9)
  CFL_36_20231220_trial_0001/  ...
  ... (15 trial dirs total)
  v25_smoke_fit_summary.json     ← per-trial QA: joint saturation flags, marker error (mm), fingertip-ball distance
  v25_full_run.log               ← full stdout/stderr log of the run, including a "trials ranked by joystick fit error" summary at the end — useful for picking best-fitting trials
```

Reading an `_ik.h5` file: `stac_mjx.io.load_stac_data(path)` returns `(cfg, StacData)` where `StacData` has `.qpos`, `.qvel`, `.kp_data`, `.marker_sites`, `.offsets`, `.kp_names` (see `stac_mjx/io.py::StacData`, `load_stac_data`).

---

## 9. Reprojection videos (for visual evaluation against the original camera footage)

Function: `stac_mjx/reproject_stac_model.py::render_reprojection_video(...)`

```python
render_reprojection_video(
    stac_h5_path=str(ik_path),                                  # the *_ik.h5 from run_stac
    calibration_path=str(trial_dir / "calibration_refined.toml"),
    video_dir=str(trial_dir),                                    # dir containing Cam00/Cam01/Cam02 subdirs with the ORIGINAL mp4s
    env_points3d_path=None,                                      # CFL dataset has no fixed env markers (bars/spout) — only used by the older A36/EMG pipeline
    registered_h5_path=str(reg_h5),                               # the *_registered_v{N}_refined.h5
    raw_points3d_path=str(trial_dir / f"{trial_name}_points3d_refined.h5"),
    clip_csv_path=None,                                          # None + trial_number=None because each CFL trial dir IS one fixed-length clip (no multi-trial session CSV needed)
    trial_number=None,
    output_path=str(reproj_out),
    fps=25,
    render_size=(540, 720),
)
```

What it does per frame, and why each step is in this specific order:
1. **Undistort the camera image** using calibration intrinsics/distortion — real camera lenses bend light non-linearly near the edges of the frame; undistorting first means every later step can pretend the camera is an ideal pinhole camera, which is the assumption `mjx`'s virtual camera rendering (and the calibration's `matrix`/rotation/translation, §0.4) makes.
2. **Render the MuJoCo model at the fitted `qpos`** from a virtual camera placed at that real camera's (now-undistorted) intrinsics/extrinsics, against a solid green background — a synthetic "if the fit were perfect, this is what the arm would look like from this exact camera" image.
3. **Alpha-composite the rendered arm onto the real camera frame** (the green background makes this a simple chroma-key-style composite) — this overlays the model directly on top of the real mouse in the real footage, in the same camera view, which is what makes visual misalignment immediately obvious (a joystick rendered floating next to the real one, a finger passing through where the real digit isn't, etc.).
4. **Draw raw mocap keypoints as an overlay** (plus environment markers, if provided) — this lets you tell apart "the mocap input was already wrong for this frame" from "the mocap was fine but STAC's fit is wrong," which is exactly the ambiguity that `CALIBRATION_REFINEMENT_REPORT.md`'s debugging kept having to resolve.
5. **Tile all 3 cameras side by side into one video** — a misalignment that's ambiguous from one camera's viewing angle (e.g. depth-wise error, which is hard to see face-on) is often obvious from another, so watching all three at once catches more than any single camera would.

Camera geometry: `stac_mjx/multicam.py` — `load_calibration` (parses `calibration.toml`), `load_registration_transform` (reads `registration_rotation`/`registration_translation`/`shoulder_reference_mm` attrs off a `registered_*.h5`, used to place MuJoCo's virtual cameras consistently with wherever the mocap was registered — i.e. this function is what bridges "model frame" back to "the real camera's actual viewpoint" for rendering purposes), `load_raw_shoulder_positions`, `CameraParams`, `RegistrationTransform`.

**This is your primary tool for visually evaluating fit quality — more trustworthy than any single aggregate number.** `v22_smoke_fit_summary.json`/`v25_smoke_fit_summary.json`'s numeric error stats are a useful triage signal (sort trials by error, spot obvious outliers), but the reprojection video is what actually caught every real bug documented in `CALIBRATION_REFINEMENT_REPORT.md` (crooked joystick, wrong camera, etc.). The reason a single aggregate marker-error number can hide a real problem: it's an average over all 19 keypoints, so a badly-wrong joystick (2 of 19 keypoints) barely moves the mean while looking obviously broken on video — always watch the video for any keypoint you specifically care about, don't trust the summary statistic alone.

---

## 10. Recipe: adding NEW 2D pose data

If you have new 2D SLEAP predictions for a new session/trial (same rig: 3 cameras, same 19-node arm+hand+joystick skeleton), here's the path through the pipeline. (Note: `luc3d-bugs/js_luc3d_hand/trainingData/` already has 5 unused `CFL_44_20240618_trial_*` trials with `Cam0X_merged.slp` files already present but never triangulated/registered/fit against v22/v25/v26 — a good first test case, since the merge step is already done for them.)

1. **Get merged 2D SLEAP predictions per camera**: `<trial>_Cam0X_merged.slp` for Cam00/Cam01/Cam02, 19 keypoints in the standard order (§2.1). If you only have separate arm/hand SLEAP prediction files, run `merge_arm_hand.py` (`/root/vast/eric/luc3d-bugs/js_luc3d_hand/trainingData/merge_arm_hand.py`) to combine them.
2. **Calibration**: if this is a new rig setup, you need a Charuco-board-derived `calibration.toml` (anipose format) for this session — this is not something `refine_calibration.py` can produce from scratch, it only *refines* an existing one. If it's the same physical rig as CFL_35/36/37, you may be able to reuse the existing `calibration.toml`.
3. **Triangulate raw 3D points**: produces `<trial>_points3d.h5` (T,1,19,3, mm). (The exact triangulation driver for the CFL dataset wasn't located in this session's investigation — check `luc3d-bugs/js_luc3d_hand/scripts/` or ask around; the general anipose recipe is `CameraGroup.triangulate()`, same call `compute_joystick_geometry_v*.py` uses.)
4. **Refine calibration for this trial**: run `/root/vast/eric/stac-mjx/refine_calibration.py` (edit its `TRIALS`/`TRAINING_ROOT` to include the new trial, or generalize it to take a trial name as an argument) → produces `calibration_refined.toml` + `<trial>_points3d_refined.h5` + `<trial>_calib_refine_stats.json` in the trial's own directory. **Sanity check the `_calib_refine_stats.json` reprojection error and the `_triangulation_check_refined.mp4`** before trusting this trial's data — if reprojection error doesn't drop like the report's Section 2 table, don't proceed until you understand why (see the "gauge drift" caveat in §3).
5. **Register into your chosen model's frame**: run the model version's `register_v{N}_mocap.py` (recommend `register_v25_mocap.py` as the template — it's the most complete/current recipe, or better, adapt it for v26 if you rebuild that model's registration). Add the new trial to its trial-discovery glob (currently hardcoded to `CFL_35/36/37` prefixes — you'll need to either rename your trial to match or edit the prefix filter). This produces `<trial>_registered_v{N}_refined.h5`.
6. **(Optional, only if you're changing rigs/model version) Recompute joystick geometry**: only needed if the joystick rig itself changed, or you're building a new model version. Otherwise reuse the existing `JOYSTICK_BASE_POS` already baked into the model XML.
7. **Run STAC + IK**: adapt `run_stac_v25_smoke.py` (or `_v26_`) — add your new trial name(s) to its trial list, run it. Produces `<trial>_fit.h5`, `<trial>_ik.h5`, `<trial>_reproj_v{N}.mp4` under a new/existing `refined_STACed_data_v{N}/<trial>/` directory.
8. **Evaluate**: watch the reprojection video against the original `Cam0X.mp4`, check `check_qpos_ranges` saturation flags and `marker_error_stats` in the run's fit-summary JSON.

If you hand this document plus the relevant script files to Claude along with a new trial's raw SLP/video files, that should be enough context to either run this recipe directly or generalize the hardcoded `CFL_35/36/37`-prefix trial discovery in `register_v25_mocap.py`/`run_stac_v25_smoke.py`/`refine_calibration.py` into something that takes an arbitrary trial list.

---

## 11. Known issues / suggested next steps

### 11.1 The model stands upright — root cause and fix options

**Root cause, confirmed directly in the v25 model XML** (`models/mouse_forelimb_right_janelia_arm_hand_v25_stac.xml`): the entire torso/spine/skull chain (`Armature` → `T13` → `T12` → ... → `T2` → `scapula_right`) has **no `<joint>` elements at all** — every one of those bodies is a rigid, fixed-offset child of the next, and the root `Armature` body itself has **no freejoint** (`pos="0 0 0" quat="1 0 0 0"`, permanently welded to world). Only the arm chain from `scapula_right` onward (`humerus_right`, `ulna_right`, hand) has actual `<joint>` elements (`shoulder_protraction`, `elbow_flexion`, etc.).

This means: STAC has **zero ability to move, rotate, or reposition the torso/spine/head** — they sit exactly wherever the original Blender/anatomical-atlas export (`janelia_model/v24/forearm_v24.xml`, built by `postprocess_xml.py` on the modeling side) posed them, which is an anatomical reference stance, not the animal's actual crouched/seated reaching posture. Since only arm+hand+joystick keypoints are tracked (no torso/spine mocap markers), there's no data and no DOF to correct this — the "upright" appearance is baked into the source XML's rest pose and is completely untouched by the STAC fit.

**This is specific to v24/v25/v26** (the full-skeleton model). v17-v22 are arm-only rigs with no torso body at all, so this issue doesn't apply to those model versions — the visible "upright" problem is a consequence of grafting the joystick-tracking arm/hand work onto v24's fuller skeleton.

**Options for future work**, roughly in order of effort:
1. **Cheapest**: just re-author `Armature`'s rest `quat` (and/or the `T13`...`T2` chain's rest quats) in the source Blender export or directly in the XML to a natural crouched/seated posture instead of the atlas's anatomical stance. This is a one-time static correction, not something STAC needs to solve — nothing tracks the torso, so there's no reason it needs to be dynamic, it just needs to *start* in a better pose.
2. **More correct but more work**: add a handful of DOF at the base of the spine (e.g. a small set of hinge/slide joints between `Armature` and `T13`, or a full 6-DOF freejoint on `Armature` if you want the whole animal repositionable) so a separate optimization stage (not necessarily STAC's per-frame IK — could be a simple one-time pose solve, or informed by video-based full-body tracking if that ever exists) can posture the torso to match the real animal. This requires **new tracked keypoints on the torso/spine** to have anything to fit against — currently there are none in the CFL dataset's 19-keypoint scheme.
3. If it turns out a full-body IK is wanted eventually, this is really a separate mocap problem (need to label spine/torso landmarks in the SLEAP data, which doesn't exist yet for this dataset) rather than a STAC/model-building fix.

Given the reaching/joystick task is specifically an **arm+hand** behavior, option 1 (just fix the static rest pose) is almost certainly the right first move unless someone specifically needs the torso to move.

### 11.2 Finger joint ranges look very tight in v25

`v25_full_run.log` shows many finger joints (`pip_flexion_2`, `dip_flexion_3`, `tip_flexion_4`, etc.) saturating at `jnt_range=[-5, 5]` degrees on nearly every trial. A ±5° range is extremely tight for finger flexion during active grasping — worth checking whether this is an intentional anatomical constraint from the Blender atlas or an export artifact that should be loosened (similar in spirit to the `sh_tx/ty/tz` saturation bug already found and fixed for v21, report §8 — that one turned out to be an optimizer bug, not a real range problem, but it's worth checking whether this one is a genuine range issue this time).

### 11.3 v26 already fixes v25's joystick placement — check it first

See §4/§5.5. If continuing joystick-tracking work, start from `models/mouse_forelimb_right_janelia_arm_hand_v26_stac.xml` / `refined_STACed_data_v26/`, not v25 — v26 is a strict improvement on the same skeleton.

### 11.4 Charuco-board validation still pending

Flagged in `CALIBRATION_REFINEMENT_REPORT.md` §4/§10 as not yet done: independently validate `calibration_refined.toml`'s per-trial extrinsics (especially Cam01's large rotation change, and the 3 excluded trials) against the original Charuco board corner detections. This is the only way to confirm whether the refinement's improvements reflect real camera-drift correction versus an artifact of fitting only to animal keypoints.

### 11.5 `CFL_44` trials exist but are unused

5 additional trials (`CFL_44_20240618_trial_{0001,0101,0201,0301,0401}`) exist under `trainingData/` with `Cam0X_merged.slp` already built, but were never triangulated/registered/run through STAC for any model version. Good first test of the "new data" recipe in §10.

---

## 12. Appendix: exhaustive path reference

**Repos**
```
/root/vast/eric/stac-mjx/                                              STAC/IK code + outputs
/root/vast/eric/luc3d-bugs/js_luc3d_hand/                               raw pose/calibration data + hand/joystick SLEAP project
/root/vast/eric/CVAT_mouse_reach/                                      arm (shoulder/elbow/wrist) SLEAP project (from CVAT labels)
/root/vast/eric/janelia_model/v{17..26}/                                source (non-STAC) model XMLs
```

**SLEAP pose models, labels, and merge (§2.3) — checkpoints actually used, verified via each prediction file's own provenance metadata**
```
trainingData/models/260708_101218.single_instance.n=134/best.ckpt   Cam00 arm model  (shoulder,elbow,wrist,hand)
trainingData/models/260708_131506.single_instance.n=180/best.ckpt   Cam01 arm model
trainingData/models/260709_100153.single_instance.n=162/best.ckpt   Cam02 arm model
trainingData/models/260702_164636.single_instance.n=120/best.ckpt   Cam00 hand+joystick model (17-node skeleton)
trainingData/models/260712_134207.single_instance.n=183/best.ckpt   Cam01 hand+joystick model
trainingData/models/260706_161230.single_instance.n=153/best.ckpt   Cam02 hand+joystick model
  (all paths relative to /root/vast/eric/luc3d-bugs/js_luc3d_hand/; trainingData/models/ holds 13 more abandoned training runs besides these 6)
trainingData/Cam0{0,1,2}_Working_akira.slp                          arm labeling projects (source of the 3 arm models above)
trainingData/Cam0{0,1,2}_Working.slp                                hand+joystick labeling projects (source of the 3 hand models above)
trainingData/train-script.sh, trainingData/jobs.yaml                sleap train invocation / job config
trainingData/merge_arm_hand.py                                      sleap_io merge: arm + hand predictions -> 19-kp *_merged.slp
handSkeleton.json, Use_Skeleton.json                                 hand+joystick skeleton definition (17 nodes) referenced by the labeling projects
/root/vast/eric/CVAT_mouse_reach/                                    a SEPARATE, larger (n=7730-14057) CVAT-labeled arm project — NOT used in any of the 60 arm.predictions.slp files verified above; don't confuse it with the real source
```

**Raw + intermediate data (per trial, under `luc3d-bugs/js_luc3d_hand/trainingData/<trial>/`)**
```
Cam0{0,1,2}/<trial>_Cam0X.mp4                            original video
Cam0{0,1,2}/<trial>_Cam0X.arm.predictions.slp            raw 2D SLEAP, arm
Cam0{0,1,2}/<trial>_Cam0X.hand.predictions.slp           raw 2D SLEAP, hand
Cam0{0,1,2}/<trial>_Cam0X_merged.slp                     merged 19-kp 2D SLEAP  (built by trainingData/merge_arm_hand.py)
calibration.toml                                         original Charuco calibration (shared across all 15 trials)
calibration_refined.toml                                 per-trial refined extrinsics (refine_calibration.py)
<trial>_calib_refine_stats.json                          before/after reprojection error
<trial>_points3d.h5                                      raw triangulated 3D (mm)
<trial>_points3d_refined.h5                              triangulated with refined calibration — used downstream
<trial>_registered_v{21,22,25}_refined.h5                mocap registered into model N's frame — STAC input
<trial>_joystick_recorded.slp                            joystick tracking labels
<trial>_triangulation_check[_refined].mp4                QA overlay videos
```

**Joystick static calibration**
```
/root/vast/eric/stac-mjx/static_joystick_labels_withbottom.v001.slp
/root/vast/eric/stac-mjx/compute_joystick_geometry_v22.py
/root/vast/eric/stac-mjx/compute_joystick_geometry_v25.py
/root/vast/eric/stac-mjx/compute_joystick_geometry_v26.py
/root/vast/eric/stac-mjx/CALIBRATION_REFINEMENT_REPORT.md
```

**Calibration refinement**
```
/root/vast/eric/stac-mjx/refine_calibration.py
```

**Registration**
```
/root/vast/eric/stac-mjx/register_v21_mocap.py
/root/vast/eric/stac-mjx/register_v22_mocap.py
/root/vast/eric/stac-mjx/register_v24_mocap.py
/root/vast/eric/stac-mjx/register_v25_mocap.py
```

**Model building**
```
/root/vast/eric/stac-mjx/build_v22_stac_xml.py
/root/vast/eric/stac-mjx/build_v24_stac_xml.py
/root/vast/eric/stac-mjx/build_v25_stac_xml.py
/root/vast/eric/stac-mjx/build_v26_stac_xml.py
/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v22_stac.xml
/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v24_stac.xml
/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v25_stac.xml
/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v26_stac.xml
```

**Configs**
```
/root/vast/eric/stac-mjx/configs/model/mouse_arm_hand_joystick_v22.yaml
/root/vast/eric/stac-mjx/configs/model/mouse_arm_hand_joystick_v25.yaml
/root/vast/eric/stac-mjx/configs/model/mouse_arm_hand_v24.yaml
/root/vast/eric/stac-mjx/configs/stac/stac_mouse_arm_hand_joystick_v22.yaml
/root/vast/eric/stac-mjx/configs/stac/stac_mouse_arm_hand_joystick_v25.yaml
/root/vast/eric/stac-mjx/configs/stac/stac_mouse_arm_hand_v24.yaml
```

**Run scripts**
```
/root/vast/eric/stac-mjx/run_stac_v22_smoke.py
/root/vast/eric/stac-mjx/run_stac_v25_smoke.py
/root/vast/eric/stac-mjx/run_stac_v26_smoke.py
```

**Core library**
```
/root/vast/eric/stac-mjx/stac_mjx/__init__.py            public API: enable_xla_flags, load_mocap, load_configs, run_stac, viz_stac
/root/vast/eric/stac-mjx/stac_mjx/main.py                 run_stac(), load_configs()
/root/vast/eric/stac-mjx/stac_mjx/stac.py                 Stac class: fit_offsets(), ik_only()
/root/vast/eric/stac-mjx/stac_mjx/stac_core.py             q_opt/q_loss (per-frame IK solver), m_opt/m_loss (offset fit)
/root/vast/eric/stac-mjx/stac_mjx/io.py                    load_mocap, load_stac_data, save_data_to_h5, StacData/Config dataclasses
/root/vast/eric/stac-mjx/stac_mjx/multicam.py              load_calibration, load_registration_transform, load_raw_shoulder_positions, CameraParams
/root/vast/eric/stac-mjx/stac_mjx/reproject_stac_model.py  render_reprojection_video()
/root/vast/eric/stac-mjx/stac_mjx/viz.py                   viz_stac() direct-render helper
```

**Outputs**
```
/root/vast/eric/stac-mjx/refined_STACed_data_v22/<trial>/{trial}_fit.h5, {trial}_ik.h5, {trial}_reproj_v22_smoke.mp4
/root/vast/eric/stac-mjx/refined_STACed_data_v22/v22_smoke_fit_summary.json
/root/vast/eric/stac-mjx/refined_STACed_data_v25/<trial>/{trial}_fit.h5, {trial}_ik.h5, {trial}_reproj_v25_smoke.mp4
/root/vast/eric/stac-mjx/refined_STACed_data_v25/v25_smoke_fit_summary.json
/root/vast/eric/stac-mjx/refined_STACed_data_v25/v25_full_run.log
/root/vast/eric/stac-mjx/refined_STACed_data_v26/  (same structure, v26)
```

**Python environment**
```
/root/vast/eric/stac-mjx/.venv/bin/python3     stac_mjx, jax, mujoco, mjx, h5py, hydra, omegaconf, dm_control, sleap_io, aniposelib
```
