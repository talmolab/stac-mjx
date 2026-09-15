# Plan: Adding Forearm Supination/Pronation to the Mouse Arm Model

## Background

The current model (`mouse_forelimb_right_janelia_moving_shoulder.xml`) has **7 DOF** total:
- 3 shoulder translations (sh_tx, sh_ty, sh_tz) as slide joints
- 3 shoulder rotations (sh_rotation, sh_extension, sh_elv) as hinge joints
- 1 elbow hinge

The kinematic chain is:
```
shoulder_base (sh_tx/ty/tz)
  -> clavicle (fixed)
    -> scapula (fixed)
      -> humerus (sh_rotation, sh_extension, sh_elv)
        -> ulna (elbow) -- radius mesh + all hand meshes rigidly attached
          -> wrist (landmark body, no joint, pos="0 0.01092 0")
```

**Problem:** The ulna and radius are one rigid body. The 3 tracked keypoints (Shoulder, Elbow, Wrist) all lie along the arm axis, so rotation about the forearm axis is invisible -- the wrist stays in the same place regardless of supination angle.

**Solution:** Add two new tracked keypoints (left knuckle, right knuckle) that form a triangle with the existing Wrist point on the back of the hand. This triangle's orientation around the forearm axis directly encodes the supination angle. Add a hinge joint for supination so STAC can fit this rotation.

---

## Part 1: XML Model Changes

### 1.1 Add a supination hinge joint on the wrist body

The simplest approach: add a hinge joint to the existing `wrist` body. The joint axis is the forearm long axis (Y in ulna-local coords, pointing elbow-to-wrist). Everything distal (wrist site, hand meshes, knuckle sites) rotates together.

```xml
<body name="wrist" pos="0 0.01092 0">
  <joint name="forearm_supination" type="hinge" pos="0 0 0" axis="0 1 0"
         range="-90 90" limited="true" damping="0.01"/>
  <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>
  <site name="Wrist" size="0.0005" pos="0 0 0"/>
  <site name="KnuckleL" size="0.0005" pos="-0.002 0.005 0"/>
  <site name="KnuckleR" size="0.0005" pos="0.002 0.005 0"/>
  <!-- all Retopo_* hand mesh geoms stay here -->
</body>
```

**Why a hinge and not a new joint type:** Supination/pronation is rotation about a single axis (the forearm long axis). A standard MuJoCo hinge joint is exactly the right primitive. What makes it "special" is just its placement and axis orientation -- along the bone rather than across it.

**Alternative (more anatomical):** Create a separate `radius_hand` body between ulna and wrist, put the supination joint there, and move the radius mesh + hand meshes onto it. This is more correct (the ulna doesn't rotate during supination) but adds complexity. Start with the simpler approach above.

### 1.2 Add knuckle sites

Two new sites on the wrist body for left and right knuckles on the back of the palm. Initial positions are approximate -- STAC's `fit_offsets` phase will learn the true offsets from the parent body.

- `KnuckleL`: left knuckle (e.g., digit 5 metacarpophalangeal joint)
- `KnuckleR`: right knuckle (e.g., digit 2 metacarpophalangeal joint)
- Rough initial offsets: ~2mm lateral, ~5mm distal from wrist

Look at the hand mesh STLs in a 3D viewer to refine these estimates.

### 1.3 Verify the supination axis

Load the model in MuJoCo viewer, confirm the Y axis of the ulna body points from elbow toward wrist. If not, adjust the hinge axis to match the actual forearm long axis (compute as the vector from elbow joint pos to wrist pos in ulna-local frame).

### 1.4 Joint range and damping

Start with `range="-90 90"` degrees. Mouse forearm supination may be less than this. Tighten the range after initial STAC runs show actual range of motion. Low damping (`0.01`) helps regularize without over-constraining.

### 1.5 Move the hand meshes

Currently the ~103 `Retopo_*` hand mesh geoms are children of the `ulna` body. They need to move to the `wrist` body so they rotate with the supination joint. This is just reparenting -- the mesh positions relative to the wrist stay the same.

---

## Part 2: SLEAP Training and 3D Data Pipeline

### 2.1 Train 2D SLEAP models for knuckle tracking

Define two new keypoints in your SLEAP project:
- `KnuckleL` -- left knuckle on dorsal (back) side of paw
- `KnuckleR` -- right knuckle on dorsal side of paw

Label training frames across diverse supination angles so the models learn to track knuckles through the full rotation range. Train models for each camera view.

### 2.2 Triangulate to 3D

Run SLEAP inference on all camera views, then triangulate KnuckleL and KnuckleR to 3D using the same multi-camera calibration pipeline used for existing keypoints.

**Quality check:** The wrist-knuckle triangle should maintain roughly consistent edge lengths across frames (it's a rigid structure). Large edge-length variation indicates triangulation errors.

### 2.3 Update the registered H5 format

Current shape: `(n_frames, 1, 3, 3)` -- 3 keypoints x 3 coordinates.
New shape: `(n_frames, 1, 5, 3)` -- 5 keypoints x 3 coordinates.

Keypoint order: `[Shoulder, Elbow, Wrist, KnuckleL, KnuckleR]`

The registration pipeline needs to include the new keypoints in the same coordinate frame as existing ones.

---

## Part 3: STAC Configuration and Code Changes

### 3.1 Update model config (`configs/model/mouse_arm.yaml`)

```yaml
KEYPOINT_MODEL_PAIRS:
  Shoulder: humerus
  Elbow: ulna
  Wrist: wrist
  KnuckleL: wrist
  KnuckleR: wrist

KEYPOINT_INITIAL_OFFSETS:
  Shoulder: [0, 0, 0]
  Elbow: [0, 0, 0]
  Wrist: [0, 0, 0]
  KnuckleL: [-0.002, 0.005, 0]
  KnuckleR: [0.002, 0.005, 0]
```

Both knuckle keypoints map to the `wrist` body. When the supination joint rotates, the wrist body rotates, and both knuckle sites rotate with it. STAC optimizes the supination angle to place the knuckle sites at their observed 3D positions.

### 3.2 Update STAC config (`configs/stac/stac_mouse_arm.yaml`)

- Keypoint count goes from 3 to 5
- Add KnuckleL and KnuckleR to name lists and color pairs

### 3.3 Create new run script

Create `run_stac_janelia_supination_v1.py` based on `run_stac_janelia_moving_shoulder_v16.py`:
- Point to new XML model with supination joint
- Point to new registered H5 with 5 keypoints
- Update config overrides for 5 keypoints
- Output directory: `refined_STACed_data_janelia_supination_v1/`

### 3.4 Verify STAC core handles new DOF

The STAC optimizer in `stac_core.py` works generically over qpos and site positions. Adding the supination joint should "just work" since:
- `qpos` shape: (N, 7) -> (N, 8) -- 8th DOF is `forearm_supination`
- `kp_data` shape: (N, 9) -> (N, 15) -- 5 keypoints x 3 coords
- `marker_sites`: (N, 3, 3) -> (N, 5, 3)
- `offsets`: (3, 3) -> (5, 3)

Check `stac_core.py` and `utils.py` for any hardcoded assumptions about joint/keypoint counts.

---

## Part 4: Validation

### 4.1 XML sanity check
- Load new model in MuJoCo viewer
- Manually vary the supination joint -- hand meshes and knuckle sites should rotate around the forearm axis while the ulna stays fixed
- Verify wrist site doesn't move significantly during supination (it's on-axis)

### 4.2 STAC test run
- Run on a single trial (e.g., reach 01)
- Check optimizer convergence (loss decreasing)
- Inspect `_stac.mp4` -- does the hand rotate realistically during reaches?

### 4.3 Reprojection check
- Render reprojection video with the fitted supination model
- Verify projected knuckle points align with 2D SLEAP detections
- Compare wrist/elbow/shoulder reprojection quality to v2 baseline (should be equal or better)

### 4.4 Biomechanical sanity
- Plot supination angle over time for reaching trials
- Should correlate with visible paw rotation in video
- Should be smooth (not noisy oscillation) and stay within anatomical range
- Check independence from other DOF (not compensating for other errors)

---

## Output Data Changes

Refined HDF5 output will change:
| Field | Old shape | New shape | Notes |
|-------|-----------|-----------|-------|
| qpos | (N, 7) | (N, 8) | +forearm_supination |
| qvel | (N, 7) | (N, 8) | |
| xpos | (N, 8, 3) | (N, 8, 3) | unchanged if using Option A (no new body) |
| xquat | (N, 8, 4) | (N, 8, 4) | unchanged if Option A |
| marker_sites | (N, 3, 3) | (N, 5, 3) | +KnuckleL, KnuckleR |
| kp_data | (N, 9) | (N, 15) | 5 kps x 3 coords |
| offsets | (3, 3) | (5, 3) | |
| kp_names | 3 entries | 5 entries | +KnuckleL, KnuckleR |
| names_qpos | 7 entries | 8 entries | +forearm_supination |

---

## Suggested Order of Operations

1. **Build the new XML model** (Part 1) -- can start now, before new tracking data
2. **Test in MuJoCo viewer** -- verify supination joint works correctly with hand meshes
3. **Train SLEAP knuckle models** (Part 2.1) -- can run in parallel with steps 1-2
4. **Triangulate and register new keypoints** (Part 2.2-2.3)
5. **Update configs and run script** (Part 3)
6. **Run STAC and validate** (Part 4)

Steps 1-2 and step 3 are independent and can proceed in parallel.
