# STAC-to-EMG Pipeline Diagnostic Report

**Date:** 2026-04-10
**Goal:** Trace the STAC model through vnl-playground training to SCAMPER evaluation.
Identify all discrepancies preventing replication of Triceps r=0.758, Biceps r=0.645.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [The STAC Model Used for Reference Data](#2-the-stac-model-used-for-reference-data)
3. [Joint DOF Order: Every XML Variant Compared](#3-joint-dof-order-every-xml-variant-compared)
4. [Joint Limits: Every XML Variant Compared](#4-joint-limits-every-xml-variant-compared)
5. [Default Physics Parameters: Every XML Variant Compared](#5-default-physics-parameters-every-xml-variant-compared)
6. [Muscle Forces: Every XML Variant Compared](#6-muscle-forces-every-xml-variant-compared)
7. [Reference Data Provenance](#7-reference-data-provenance)
8. [Checkpoint Configurations Compared](#8-checkpoint-configurations-compared)
9. [SCAMPER Evaluation Configuration](#9-scamper-evaluation-configuration)
10. [Timeline of All Model Changes](#10-timeline-of-all-model-changes)
11. [Root Cause Analysis](#11-root-cause-analysis)
12. [Recommended Actions](#12-recommended-actions)

---

## 1. Pipeline Overview

```
3D Keypoints (CVAT)
    |
    v
STAC-MJX  (stac-mjx/)
    | Uses: models/mouse_forelimb_right_janelia.xml
    | Produces: per-trial h5 files with qpos, names_qpos
    v
Reference Clips  (vnl-playground/.../reference_data/)
    | 46 h5 files, each 100 frames x 4 DOFs
    | Joint order stored as names_qpos: [sh_elv, sh_extension, sh_rotation, elbow]
    v
RL Training  (vnl-playground/)
    | Uses: xmls/mouse_forelimb_right.xml (MUSCLE model)
    | Loads qpos BY INDEX from h5, assigns to simulation BY INDEX
    | Produces: checkpoint (policy params + config.json)
    v
SCAMPER Evaluation  (SCAMPER/)
    | Uses: SAME mouse_forelimb_right.xml via editable vnl-playground install
    | (NOT akira_muscle.xml -- that's a dead copy in scratch/repos/)
    | Loads checkpoint, runs rollouts, extracts ctrl[:,sim_idx]
    | Compares simulated muscle activation vs real EMG
    v
EMG Correlation (r-value)
```

**Critical invariant:** The joint order in the STAC XML, the reference h5 files, and the training XML must all agree, because qpos is loaded and assigned **by positional index, not by joint name**.

---

## 2. The STAC Model Used for Reference Data

The reference h5 files embed their provenance. From `A36-1_..._trial01_ik.h5`:

```
config.model.MJCF_PATH: /root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia.xml
```

**This is the STAC model.** It is a kinematics-only variant (no muscles/tendons) of the Janelia forelimb model.

| Property | Value | Source |
|----------|-------|--------|
| Path | `stac-mjx/models/mouse_forelimb_right_janelia.xml` | h5 config |
| Size | 22,825 bytes | `ls -la` |
| Modified | 2026-04-07 06:07 | filesystem |
| Git status | **Untracked** (never committed) | `git status` |
| Muscles | **None** (removed for kinematics) | Lines 349-351 comment |

---

## 3. Joint DOF Order: Every XML Variant Compared

This is the most critical table. The reference data stores qpos in the order the STAC XML defines joints. Training assigns qpos to the simulation XML by index. **Any order mismatch = wrong joint angles.**

### DOF Position 0 (first hinge joint)

| XML File | Location | DOF 0 Name | DOF 0 Axis |
|----------|----------|------------|------------|
| `akira_muscle.xml` | SCAMPER (frozen) | **sh_elv** | Y |
| `assets_high_res/mouse_forelimb_right.xml` | vnl-playground | **sh_elv** | Y |
| `mouse_forelimb_right.xml` (production) | vnl-playground | **sh_elv** | Y |
| `akira_torque.xml` | stac-mjx | **sh_elv** | Y |
| `arm_model_v3_torque.xml` | stac-mjx | **sh_elv** | Y |
| `mouse_forelimb_right_janelia.xml` (STAC) | stac-mjx | **sh_rotation** | Y |
| `mouse_forelimb_right_test.xml` | vnl-playground | **sh_rotation** | Y |
| `mouse_forelimb_right_janelia_moving_shoulder.xml` | stac-mjx | **sh_rotation** | Y |
| `mouse_forelimb_right_janelia_moving_shoulder_v2.xml` | stac-mjx | **sh_rotation** | Y |

### Full DOF Order Comparison

| XML File | DOF 0 | DOF 1 | DOF 2 | DOF 3 |
|----------|-------|-------|-------|-------|
| **akira_muscle.xml** (SCAMPER) | sh_elv | sh_extension | sh_rotation | elbow |
| **assets_high_res/** (vnl) | sh_elv | sh_extension | sh_rotation | elbow |
| **mouse_forelimb_right.xml** (vnl prod) | sh_elv | sh_extension | sh_rotation | elbow |
| **akira_torque.xml** (stac) | sh_elv | sh_extension | sh_rotation | elbow_joint |
| **arm_model_v3_torque.xml** (stac) | sh_elv | sh_extension | sh_rotation | elbow_joint |
| **janelia.xml** (STAC model) | sh_rotation | sh_extension | sh_elv | elbow |
| **_test.xml** (vnl test) | sh_rotation | sh_extension | sh_elv | elbow |
| **_moving_shoulder.xml** (stac) | sh_rotation | sh_extension | sh_elv | elbow |
| **_moving_shoulder_v2.xml** (stac) | sh_rotation | sh_extension | sh_elv | elbow |

### Summary: Two Families of Joint Order

| Family | Order | Members |
|--------|-------|---------|
| **"Akira" order** (original) | sh_elv, sh_extension, sh_rotation, elbow | akira_muscle, akira_torque, arm_v3, assets_high_res, **vnl production** |
| **"Janelia" order** (swapped) | sh_rotation, sh_extension, sh_elv, elbow | **STAC model**, vnl test, all moving_shoulder variants |

### The Paradox

The STAC model (`mouse_forelimb_right_janelia.xml`) uses **Janelia order** (sh_rotation first).
But the reference h5 files say `names_qpos = ['sh_elv', 'sh_extension', 'sh_rotation', 'elbow']`.

This means **STAC saved names in alphabetical/logical order but the actual qpos array follows the XML order** (sh_rotation, sh_extension, sh_elv, elbow). The `names_qpos` metadata is a label list that reflects the MuJoCo joint ordering -- but wait:

**Verification from h5 data ranges (all 46 clips, 4600 frames):**

| H5 DOF Index | names_qpos label | Observed Range (deg) | Matches which joint? |
|--------------|------------------|---------------------|---------------------|
| 0 | sh_elv | [-61.3, 0.0] | sh_elv [-70, 0] in Akira; OR sh_rotation [-70, 40] in Janelia? |
| 1 | sh_extension | [4.2, 40.0] | sh_extension [-70, 40] in both |
| 2 | sh_rotation | [-25.0, 72.9] | sh_rotation [-25, 89] in Akira; OR sh_elv [-60, 89] in Janelia? |
| 3 | elbow | [-19.8, 51.3] | elbow [-70/80, 80/90] in both |

**Key evidence:**
- DOF 0 range is [-61.3, 0.0] -- this is **consistent with sh_elv [-70, 0]** (Akira order). It is NOT consistent with sh_rotation [-70, 40] (Janelia order), because it never goes positive.
- DOF 2 range is [-25.0, 72.9] -- this is **consistent with sh_rotation [-25, 89]** (Akira order). It is NOT consistent with sh_elv [-60, 89] (Janelia order), because the lower bound is -25, not -60.

**Conclusion: The h5 names_qpos labels ARE correct, and the data follows Akira order (sh_elv first).**

This means one of two things:
1. The STAC model had Akira order when the reference data was generated (it was later modified to Janelia order), OR
2. STAC reorders qpos by name before saving (unlikely -- the code saves raw MuJoCo qpos)

Checking `stac_mjx/io.py` lines 256-301: `save_data_to_h5()` saves the `names_qpos` parameter directly, which comes from `mj_model.joint(i).name`. The qpos array comes from MuJoCo's internal ordering. **There is no reordering.**

**Therefore: The STAC XML was in Akira order when the reference data was generated, and was later changed to Janelia order.** The XML was never version-controlled, so we cannot confirm when this change happened.

---

## 4. Joint Limits: Every XML Variant Compared

All ranges in degrees.

| Joint | akira_muscle (SCAMPER) | akira_torque (stac) | arm_v3_torque (stac) | high_res (vnl) | production (vnl) | test (vnl) | janelia (STAC) | moving_sh_v1 (stac) | moving_sh_v2 (stac) |
|-------|----------------------|--------------------|--------------------|---------------|-----------------|-----------|---------------|--------------------|--------------------|
| sh_elv | **[-70, 0]** | **[-70, 0]** | **[-70, 0]** | **[-70, 0]** | **[-70, 0]** | [-60, 89] | [-60, 89] | [-60, 89] | [-90, 90] |
| sh_extension | [-70, 40] | [-70, 40] | [-70, 40] | [-70, 40] | [-70, 40] | [-70, 40] | [-70, 40] | [-70, 40] | [-90, 90] |
| sh_rotation | **[-25, 89]** | **[-25, 89]** | **[-25, 89]** | **[-25, 89]** | **[-25, 89]** | [-70, 40] | [-70, 40] | [-70, 40] | [-90, 90] |
| elbow | [-80, 90] | [-80, 90] | [-80, 90] | [-70, 80] | [-70, 80] | [-70, 80] | [-70, 80] | [-70, 80] | [-90, 100] |

### Joint Limit Families

| Family | sh_elv | sh_rotation | elbow | Members |
|--------|--------|-------------|-------|---------|
| **Original Akira** | [-70, 0] | [-25, 89] | [-80, 90] | akira_muscle, akira_torque, arm_v3 |
| **Akira + narrowed elbow** | [-70, 0] | [-25, 89] | [-70, 80] | high_res, **vnl production** |
| **Swapped limits** | [-60, 89] | [-70, 40] | [-70, 80] | **STAC janelia**, vnl test, moving_sh_v1 |
| **Wide open** | [-90, 90] | [-90, 90] | [-90, 100] | moving_sh_v2 |

**Critical observation:** In the "Swapped limits" family, the limits for sh_elv and sh_rotation appear to have been *swapped along with the joint order*. The Janelia XMLs put sh_rotation first (where sh_elv used to be), and gave it the range [-70, 40] -- which is similar to the original sh_elv range. Similarly, sh_elv (now in position 2) got range [-60, 89], similar to original sh_rotation [-25, 89].

This suggests someone relabeled/reordered the joints but adjusted the ranges to follow the new positions rather than the original joint names.

---

## 5. Default Physics Parameters: Every XML Variant Compared

| XML File | armature | damping | stiffness | solver |
|----------|----------|---------|-----------|--------|
| **akira_muscle.xml** (SCAMPER) | **4e-8** | **1e-5** | 1e-12 | CG (6 iter, 6 ls) |
| **assets_high_res/** (vnl) | **4e-8** | **1e-5** | 1e-12 | CG (6 iter, 6 ls) |
| **mouse_forelimb_right.xml** (vnl prod) | **4e-10** | **1e-6** | 1e-12 | CG (6 iter, 6 ls) |
| **mouse_forelimb_right_test.xml** (vnl) | **4e-10** | **1e-5** | 1e-12 | CG (6 iter, 6 ls) |
| **janelia.xml** (STAC) | **4e-10** | **1e-6** | 1e-12 | CG (6 iter, 6 ls) |
| **moving_shoulder.xml** (stac) | **4e-10** | **1e-6** | 1e-12 | CG (6 iter, 6 ls) |
| **moving_shoulder_v2.xml** (stac) | **4e-10** | **1e-6** | 1e-12 | CG (6 iter, 6 ls) |
| **akira_torque.xml** (stac) | **4e-6** | **0.001** | **0.001** | (default) |
| **arm_v3_torque.xml** (stac) | **4e-6** | **0.001** | **0.001** | (default) |

### Physics Parameter Evolution

| Parameter | akira_muscle (original) | high_res | vnl production | STAC janelia | akira_torque |
|-----------|------------------------|----------|---------------|-------------|-------------|
| armature | 4e-8 | 4e-8 | **4e-10** (100x smaller) | **4e-10** | 4e-6 (100x larger) |
| damping | 1e-5 | 1e-5 | **1e-6** (10x smaller) | **1e-6** | 0.001 (100x larger) |
| stiffness | 1e-12 | 1e-12 | 1e-12 | 1e-12 | **0.001** (huge) |

**Note:** Training configs override armature and damping at runtime. The XML defaults matter for:
1. MuJoCo model compilation (muscle length range computation via `<lengthrange>`)
2. Any code that doesn't apply runtime overrides

---

## 6. Muscle Forces: Every XML Variant Compared

The STAC janelia XML has **no muscles** (kinematics only). The comparison below covers all muscle-bearing XMLs.

### Arm Muscles (EMG-relevant)

| Muscle | akira_muscle (SCAMPER) | high_res (vnl) | test (vnl) | production (vnl) | Actuator Index |
|--------|----------------------|---------------|-----------|-----------------|---------------|
| Pec_C | 0.8 | 0.2 | 0.2 | 0.2 | 0 |
| Lat | 0.8 | 0.2 | 0.2 | 0.2 | 1 |
| AD | 1.2 | 0.4 | 0.4 | 0.4 | 2 |
| PD | 0.87 | 0.2 | 0.2 | 0.2 | 3 |
| MD | 0.9 | 0.2 | 0.2 | 0.2 | 4 |
| **Triceps_Lateral** | **0.6** | **0.2** | **0.2** | **0.1** | **5** |
| **Triceps_Long** | **0.9** | **0.2** | **0.2** | **0.1** | **6** |
| **Brachialis** | **0.6** | **0.2** | **0.2** | **0.1** | **7** |
| **Biceps_Long** | **0.52** | **0.2** | **0.2** | **0.1** | **8** |

### Rotator Cuff Muscles

| Muscle | akira_muscle | high_res | test | production | Actuator Index |
|--------|-------------|----------|------|-----------|---------------|
| Supraspinatus | N/A | 0.6 | 0.5 | 0.5 | 9 |
| Infraspinatus | N/A | 0.5 | 0.5 | 0.5 | 10 |
| Subscapularis | N/A | 0.6 | 0.6 | 0.6 | 11 |

### Muscle Force Evolution

| Muscle | akira_muscle | high_res / test | production | Change |
|--------|-------------|----------------|-----------|--------|
| Triceps_Lateral | 0.6 | 0.2 | **0.1** | 6x -> 2x reduction |
| Triceps_Long | 0.9 | 0.2 | **0.1** | 9x -> 2x reduction |
| Brachialis | 0.6 | 0.2 | **0.1** | 6x -> 2x reduction |
| Biceps_Long | 0.52 | 0.2 | **0.1** | 5.2x -> 2x reduction |

**The production XML has halved the arm muscle forces compared to high_res/test.** The original akira_muscle had much higher forces. The EMG-correlated checkpoint (r=0.758) was trained at some point in this evolution.

### Muscle Actuator Order (for hardcoded indices in EMG comparison)

All muscle-bearing XMLs define actuators in the same order:
```
0: Pec_C, 1: Lat, 2: AD, 3: PD, 4: MD,
5: Triceps_Lateral, 6: Triceps_Long, 7: Brachialis, 8: Biceps_Long
[9: Supraspinatus, 10: Infraspinatus, 11: Subscapularis -- only in newer XMLs]
```

The SCAMPER and vnl-playground EMG comparison code hardcodes `sim_idx=5` for Triceps_Lateral and `sim_idx=8` for Biceps_Long. **This is correct for all XML variants** (muscle order is consistent).

**However:** The akira_muscle.xml only has 9 actuators (no rotator cuff), while newer XMLs have 12. Any code using `ctrl[:, 5]` or `ctrl[:, 8]` will still get the right muscle, but the action space dimension differs (9 vs 12).

---

## 7. Reference Data Provenance

### H5 Files in stac-mjx/ (STAC outputs)

| File | names_qpos | qpos shape | Date | STAC XML |
|------|-----------|-----------|------|----------|
| `mouse_arm_fit_new.h5` | [sh_elv, sh_extension, sh_rotation, **elbow_joint**] | (1000, 4) | Feb 11, 2025 | akira_torque* |
| `mouse_arm_fit_new_full.h5` | [sh_elv, sh_extension, sh_rotation, **elbow_joint**] | (123548, 4) | Feb 11, 2025 | akira_torque* |
| `mouse_arm_ik_only_new.h5` | [sh_elv, sh_extension, sh_rotation, **elbow_joint**] | (1000, 4) | Feb 13, 2025 | akira_torque* |
| `reaches_*_reach*_ik.h5` | [sh_elv, sh_extension, sh_rotation, **elbow**] | varies | recent | janelia* |

*Inferred from joint name: `elbow_joint` = akira_torque/arm_v3 model; `elbow` = janelia model.

### H5 Files in vnl-playground/reference_data/ (training inputs)

| Property | Value |
|----------|-------|
| Files | 46 trial-level h5 files |
| Format | `A36-1_2023-07-18_..._trialNN_ik.h5` |
| Created | 2026-03-02 23:52 (all files same date) |
| names_qpos | `['sh_elv', 'sh_extension', 'sh_rotation', 'elbow']` (all files) |
| qpos shape | (100, 4) per file |
| Embedded MJCF_PATH | `stac-mjx/models/mouse_forelimb_right_janelia.xml` |

### Aggregate Reference Data Ranges vs XML Limits

| DOF | H5 Label | Observed Range (deg) | akira_muscle limit | vnl production limit | STAC janelia limit (at DOF position) |
|-----|----------|---------------------|-------------------|---------------------|--------------------------------------|
| 0 | sh_elv | [-61.3, 0.0] | sh_elv: [-70, 0] | sh_elv: [-70, 0] | sh_rotation: [-70, 40] |
| 1 | sh_extension | [4.2, 40.0] | sh_ext: [-70, 40] | sh_ext: [-70, 40] | sh_ext: [-70, 40] |
| 2 | sh_rotation | [-25.0, 72.9] | sh_rot: [-25, 89] | sh_rot: [-25, 89] | sh_elv: [-60, 89] |
| 3 | elbow | [-19.8, 51.3] | elbow: [-80, 90] | elbow: [-70, 80] | elbow: [-70, 80] |

**Conclusion from range analysis:**
- DOF 0 never exceeds 0 deg -> matches `sh_elv [-70, 0]` in Akira order, NOT `sh_rotation [-70, 40]` in Janelia order
- DOF 2 has lower bound -25 -> matches `sh_rotation [-25, 89]` in Akira order, NOT `sh_elv [-60, 89]` in Janelia order

**The reference data was generated with Akira-order joints (sh_elv first), matching the vnl production XML.**

---

## 8. Checkpoint Configurations Compared

### Source Checkpoint (r=0.758)

| Field | `janelia-mouse-intention-20260408-182117` | File |
|-------|------------------------------------------|------|
| Created | 2026-04-08 18:21 (20:01 dir mod) | `ls -la` |
| walker_xml_path | `.../mouse_forelimb_right.xml` | config.json |
| joint_damping | **1e-6** | config.json |
| joint_armature | 4e-10 | config.json |
| joint_stiffness | null | config.json |
| force_scale | 1.0 | config.json |
| control_cost | 0.1 | config.json |
| control_diff_cost | 0.0 | config.json |
| qvel_init | "zeros" | config.json |
| episode_length | 100 | config.json |
| clip_length | 50 | config.json |

### S1-07 Physics Sweep

| Field | `intention-S1-07-physics` | File |
|-------|---------------------------|------|
| Created | 2026-04-09 19:19 (20:13 dir mod) | `ls -la` |
| walker_xml_path | `.../mouse_forelimb_right.xml` | config.json |
| joint_damping | **1e-7** | config.json |
| joint_armature | 4e-10 | config.json |
| control_cost | 0.1 | config.json |
| control_diff_cost | 0.0 | config.json |
| qvel_init | "zeros" | config.json |

### All Checkpoint Physics Comparison

| Checkpoint | damping | armature | stiffness | force_scale | control_cost | control_diff | qvel_init | Date |
|-----------|---------|----------|-----------|-------------|-------------|-------------|-----------|------|
| **janelia-...-20260408-182117** (r=0.758) | **1e-6** | 4e-10 | null | 1.0 | 0.1 | 0.0 | zeros | Apr 8 |
| intention-S1-05-physics | **1e-6** | 4e-10 | null | **1.5** | 0.1 | 0.0 | zeros | Apr 9 |
| **intention-S1-07-physics** | **1e-7** | 4e-10 | null | 1.0 | 0.1 | 0.0 | zeros | Apr 9 |
| intention-S1-17-physics | **1e-7** | 4e-10 | null | 1.0 | 0.1 | 0.0 | zeros | Apr 9 |
| intention-S1-22-physics | **1e-7** | 4e-10 | **1e-6** | 1.0 | 0.1 | 0.0 | zeros | Apr 9 |
| replicate-cc0.1 | **1e-7** | 4e-10 | null | 1.0 | 0.1 | 0.0 | **reference** | Apr 10 |
| replicate-cc0.01 | **1e-7** | 4e-10 | null | 1.0 | **0.01** | 0.0 | zeros | Apr 10 |
| replicate-20260408-fixed-joints | **1e-6** | 4e-10 | null | 1.0 | 0.1 | **0.02** | zeros | Apr 10 |
| replicate-s1-07-fixed-joints | **1e-7** | 4e-10 | null | 1.0 | 0.1 | **0.02** | zeros | Apr 10 |
| low-cc-smooth-fixed-joints | **1e-6** | 4e-10 | null | 1.0 | **0.02** | **0.01** | zeros | Apr 10 |
| janelia-...-20260410-065440 | **1e-6** | **4e-9** | null | 1.0 | **0.05** | **0.05** | zeros | Apr 10 |

### EMG Pareto Sweep Checkpoints

| Checkpoint | damping | armature | cc | cdc | qvel_init | latent |
|-----------|---------|----------|------|------|-----------|--------|
| emg-pareto-lat4-d1e-7-a4e-10-cc0.02-cdc0.0 | 1e-7 | 4e-10 | 0.02 | 0.0 | zeros | 4 |
| emg-pareto-lat5-d1e-7-a4e-10-cc0.05-cdc0.0 | 1e-7 | 4e-10 | 0.05 | 0.0 | reference | 5 |
| emg-pareto-lat5-d1e-7-a4e-10-cc0.05-cdc0.05 | 1e-7 | 4e-10 | 0.05 | 0.05 | reference | 5 |
| emg-pareto-lat5-d1e-7-a4e-10-cc0.1-cdc0.0 | 1e-7 | 4e-10 | 0.1 | 0.0 | zeros | 5 |
| emg-pareto-lat5-d1e-7-a4e-10-cc0.1-cdc0.05 | 1e-7 | 4e-10 | 0.1 | 0.05 | reference | 5 |
| emg-pareto-lat5-d1e-7-a4e-9-cc0.05-cdc0.0 | 1e-7 | **4e-9** | 0.05 | 0.0 | reference | 5 |
| emg-pareto-lat5-d1e-7-a4e-9-cc0.05-cdc0.05 | 1e-7 | **4e-9** | 0.05 | 0.05 | reference | 5 |
| emg-pareto-lat5-d1e-7-a4e-9-cc0.1-cdc0.0 | 1e-7 | **4e-9** | 0.1 | 0.0 | reference | 5 |
| emg-pareto-lat5-d1e-7-a4e-9-cc0.1-cdc0.05 | 1e-7 | **4e-9** | 0.1 | 0.05 | reference | 5 |
| emg-pareto-lat5-d1e-7-a8e-10-cc0.1-cdc0.0 | 1e-7 | **8e-10** | 0.1 | 0.0 | zeros | 5 |
| emg-pareto-v2-lat5-d1e-7-a4e-10-cc0.1-cdc0.0 | 1e-7 | 4e-10 | 0.1 | 0.0 | zeros | 5 |

### Key Difference: Source Checkpoint vs S1-07

| Parameter | Source (r=0.758) | S1-07 | Difference |
|-----------|-----------------|-------|------------|
| joint_damping | **1e-6** | **1e-7** | 10x less damping in S1-07 |
| Everything else | identical | identical | - |

The source checkpoint that produced r=0.758 used **damping=1e-6**, not 1e-7. The S1-07 checkpoint was a physics sweep variant with 10x lower damping.

---

## 9. SCAMPER Evaluation Configuration

### Which XML Does SCAMPER Actually Use?

SCAMPER has a **dead copy** of the old XML at:
```
/root/vast/eric/SCAMPER/scratch/repos/vnl-playground/.../akira_muscle.xml  (NOT USED)
```

But at runtime, SCAMPER imports `vnl_playground` as an **editable install** pointing to `/root/vast/eric/vnl-playground/`. The import chain:

1. `SCAMPER/notebooks/neurips_figures/utils.py` line 84: `from vnl_playground.tasks.mouse.consts import JANELIA_MOUSE_XML_PATH`
2. `vnl-playground/vnl_playground/tasks/mouse/consts.py`: `JANELIA_MOUSE_XML_PATH = .../xmls/mouse_forelimb_right.xml`
3. `utils.py` line 88: `env_cfg.walker_xml_path = JANELIA_MOUSE_XML_PATH`

**SCAMPER uses the SAME `mouse_forelimb_right.xml` as training.** The `akira_muscle.xml` in `scratch/repos/` is never imported.

| Property | akira_muscle.xml (DEAD COPY) | mouse_forelimb_right.xml (ACTUALLY USED) |
|----------|--------------------------|-----------------------------------|
| Modified | 2026-03-16 | 2026-04-10 |
| Actuators | 9 (no rotator cuff) | 12 (with rotator cuff) |
| Triceps_Lat force | 0.6 | 0.1 |
| Status | **Orphaned, never loaded** | **Active** |

### SCAMPER Config: `mouse-prior-janelia.yaml`

| Parameter | SCAMPER config | Source checkpoint config |
|-----------|---------------|------------------------|
| joint_damping | 1e-6 | 1e-6 |
| joint_armature | 4e-10 | 4e-10 |
| control_cost weight | **0.01** | **0.1** |
| control_diff_cost weight | 0.0 | 0.0 |
| clip_length | 50 | 50 |
| episode_length | 100 | 100 |
| qvel_init | zeros | zeros |

The SCAMPER config has `control_cost=0.01` (10x lower than the source checkpoint's 0.1). This only matters for reward computation during rollouts, not for policy inference.

---

## 10. Timeline of All Model Changes

| Date | Event | Source |
|------|-------|--------|
| 2025-02-05 | `mujoco_aligned_tracking.h5` created (raw keypoint data) | filesystem |
| 2025-02-11 | `mouse_arm_fit_new_full.h5` created (STAC with akira_torque, `elbow_joint`) | h5 names_qpos |
| 2025-02-13 | `mouse_arm_ik_only_new.h5` created | filesystem |
| 2026-02-25 | `akira_muscle.xml` deleted from vnl-playground git | git log |
| 2026-03-02 23:52 | **46 reference h5 files created** (using STAC janelia model, Akira joint order) | filesystem |
| 2026-03-03 | vnl-playground `consts.py` updated to reference `mouse_forelimb_right.xml` | git log |
| 2026-03-16 | SCAMPER `akira_muscle.xml` frozen copy created | filesystem |
| 2026-04-02 05:24 | `assets_high_res/mouse_forelimb_right.xml` last modified | stat |
| 2026-04-04 16:54 | `arm_model_v3_torque.xml` created in stac-mjx | filesystem |
| 2026-04-07 06:05 | `akira_torque_moving_shoulder.xml` created in stac-mjx | filesystem |
| 2026-04-07 06:07 | `akira_torque.xml`, `mouse_forelimb_right_janelia.xml` modified in stac-mjx | filesystem |
| 2026-04-07 06:31 | `janelia_4dof.xml`, `janelia_moving_shoulder.xml` created in stac-mjx | filesystem |
| 2026-04-07 18:51 | `moving_shoulder_v1.xml`, `moving_shoulder_v2.xml` created in stac-mjx | filesystem |
| **2026-04-08 18:21** | **Source checkpoint trained** (`janelia-mouse-intention-20260408-182117`, damping=1e-6) | checkpoint |
| **2026-04-09 19:19** | **S1-07 checkpoint trained** (damping=1e-7) | checkpoint |
| 2026-04-09 | Multiple janelia-intention checkpoints created (physics sweeps) | ls -la |
| 2026-04-10 16:15 | `mouse_forelimb_right_test.xml` last modified (Janelia order, 0.2 forces) | stat |
| **2026-04-10 16:58** | **`mouse_forelimb_right.xml` (production) last modified** (Akira order, 0.1 forces) | stat |
| 2026-04-10 | Replicate checkpoints created, EMG pareto sweep launched | ls -la |

### Critical Gap

The production XML was modified on **Apr 10 16:58** -- AFTER all training. We don't know:
1. What the XML looked like during the Apr 8 training (source of r=0.758)
2. Whether the joint order was always Akira order, or was briefly Janelia order
3. Whether muscle forces were 0.1 or 0.2 during training

The `_test.xml` was created at 16:15 (43 min before the production XML was last modified), suggesting they were being compared/edited simultaneously on Apr 10.

---

## 11. Root Cause Analysis

### Confirmed Issues

| # | Issue | Evidence | Impact |
|---|-------|----------|--------|
| 1 | **STAC janelia XML has different joint order than reference data** | h5 names_qpos=[sh_elv,...] but STAC XML DOF 0=sh_rotation | Any new STAC run with current janelia XML will produce WRONG reference data |
| 2 | **Production XML was modified after training** | stat shows Apr 10 16:58 > checkpoint dates Apr 8-9 | Cannot confirm training-time state of XML |
| 3 | ~~SCAMPER uses different XML than training~~ | **CORRECTED:** SCAMPER imports vnl-playground as editable install, uses same `mouse_forelimb_right.xml` | Not an issue -- `akira_muscle.xml` in scratch/repos/ is a dead copy |
| 4 | **Muscle forces differ 2-9x across XMLs** | See Table 6 | Different force -> different activation magnitude for same joint torque |
| 5 | **Test XML has Janelia joint order** | test.xml DOF 0=sh_rotation, names match janelia STAC model | If test XML was ever used for training, qpos would be swapped |
| 6 | **Default physics params changed across XMLs** | armature: 4e-8 -> 4e-10 (100x); damping: 1e-5 -> 1e-6 (10x) | Affects lengthrange computation even when overridden at runtime |
| 7 | **Source checkpoint used damping=1e-6, S1-07 used 1e-7** | config.json comparison | 10x physics difference between the checkpoint that produced r=0.758 and S1-07 |

### STAC-Specific Issues

| # | Issue | Evidence | Impact |
|---|-------|----------|--------|
| S1 | **STAC XML never version-controlled** | git status shows untracked | Cannot trace model history |
| S2 | **STAC janelia XML joint order INCONSISTENT with h5 outputs** | XML: sh_rotation first; h5: sh_elv first | STAC XML was modified AFTER generating reference data |
| S3 | **Two different elbow joint names** | akira models: `elbow_joint`; janelia models: `elbow` | Old STAC h5 files have `elbow_joint`, new ones have `elbow` |
| S4 | **STAC config points to modified janelia XML** | `configs/model/mouse_arm.yaml` -> `mouse_forelimb_right_janelia.xml` | Any future STAC run will use wrong joint order |
| S5 | **No joint order validation in pipeline** | reference_clips.py loads by index; io.py saves names but doesn't validate | Silent failure when XML order changes |

### Most Likely Scenario for r=0.758

1. STAC ran with janelia XML **in Akira order** (sh_elv first) around Mar 2 -> produced 46 reference h5 files
2. vnl-playground XML (`mouse_forelimb_right.xml`) was **also in Akira order** at that time -> training used correct joint alignment
3. Training on Apr 8 with damping=1e-6, cc=0.1 -> produced checkpoint `20260408-182117`
4. EMG comparison with that checkpoint -> r=0.758 for Triceps
5. After Apr 8, both XMLs were modified (joint order, limits, forces) -> replication became impossible

---

## 12. Recommended Actions

### Immediate (to recover r=0.758)

1. **Verify the production XML is currently in Akira order** -- it IS (confirmed: DOF 0 = sh_elv). So the vnl-playground production XML is correct.

2. **Re-run EMG comparison using the source checkpoint** (`janelia-mouse-intention-20260408-182117`) with the current production XML. The config says damping=1e-6, which matches. If r=0.758 is recovered, the production XML is the correct one.

3. **Do NOT use S1-07 for EMG replication** -- it has damping=1e-7 (10x different from the source).

### For STAC

4. **Fix the STAC janelia XML joint order** back to Akira order:
   ```
   Current (WRONG):  sh_rotation, sh_extension, sh_elv, elbow
   Should be:        sh_elv, sh_extension, sh_rotation, elbow
   ```
   This means also swapping the axis assignments and ranges back.

5. **Commit all XML files to git immediately.** The STAC janelia XML, all variants, and the vnl-playground XMLs should be version-controlled.

6. **Add joint order validation to the pipeline.** When loading reference h5 files, compare `names_qpos` from the h5 against the MuJoCo model's joint order. Raise an error on mismatch.

### For SCAMPER

7. ~~Update SCAMPER's XML~~ **NOT NEEDED** -- SCAMPER already uses the production `mouse_forelimb_right.xml` via editable vnl-playground install. The `akira_muscle.xml` in `scratch/repos/` is a dead copy.

8. **Verify actuator indices** are still correct for the 12-actuator model (indices 5=Triceps_Lateral and 8=Biceps_Long are correct -- the 3 rotator cuff muscles are appended at indices 9-11).

### For Future Reproducibility

9. **Save XML hash in checkpoint config.json** so we can always verify which XML was used.

10. **Add name-based qpos mapping** in `reference_clips.py` and `imitation.py` so joint order mismatches are detected or automatically resolved.

---

## Appendix A: File Locations Reference

| File | Path |
|------|------|
| STAC model (kinematics) | `/root/vast/eric/stac-mjx/models/mouse_forelimb_right_janelia.xml` |
| STAC config | `/root/vast/eric/stac-mjx/configs/model/mouse_arm.yaml` |
| STAC io code | `/root/vast/eric/stac-mjx/stac_mjx/io.py` |
| VNL production XML (muscles) | `/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/mouse_forelimb_right.xml` |
| VNL test XML | `/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/mouse_forelimb_right_test.xml` |
| VNL high-res XML | `/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/xmls/assets_high_res/mouse_forelimb_right.xml` |
| VNL reference data | `/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_data/` |
| VNL reference_clips.py | `/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/reference_clips.py` |
| VNL imitation.py | `/root/vast/eric/vnl-playground/vnl_playground/tasks/mouse/imitation.py` |
| Source checkpoint | `/root/vast/eric/vnl-playground/checkpoints/janelia-mouse-intention-20260408-182117/` |
| S1-07 checkpoint | `/root/vast/eric/vnl-playground/checkpoints/intention-S1-07-physics/` |
| SCAMPER XML (frozen) | `/root/vast/eric/SCAMPER/scratch/repos/vnl-playground/vnl_playground/tasks/mouse/xmls/akira_muscle.xml` |
| SCAMPER config | `/root/vast/eric/SCAMPER/scamper/config/mouse-prior-janelia.yaml` |
| SCAMPER EMG notebook | `/root/vast/eric/SCAMPER/notebooks/neurips_figures/emg_comparison.ipynb` |
| SCAMPER EMG script | `/root/vast/eric/SCAMPER/scripts/reproduce_emg_correlation.py` |
| EMG data | `/root/vast/eric/mouse-reach-mjx-neurips/emg/` |
| Trial CSV | `/root/vast/eric/mouse-reach-mjx-neurips/trial_info/A36-1_2023-07-18_16-54-01_lightOff_tone_on_off_trials_edited.csv` |

## Appendix B: How qpos Flows Through the Pipeline

```
STAC (stac_mjx/stac.py)
  mj_model.joint(i).name -> names_qpos list
  mj_data.qpos[7:] -> qpos array (skip root pos+quat for freejoint, or all 4 DOFs for fixed-base)
      |
      v
stac_mjx/io.py:save_data_to_h5()
  h5["names_qpos"] = names_qpos   # joint names in XML order
  h5["qpos"] = qpos               # angles in XML order
      |
      v
vnl-playground/reference_clips.py:__init__()
  self._names_qpos = f["names_qpos"]   # loaded but NOT used for reordering
  all_qpos.append(f["qpos"][:n])       # loaded by index
  self._data_arrays["qpos"] = stack(all_qpos)
      |
      v
vnl-playground/imitation.py
  data.replace(qpos=reference.qpos)     # assigned by index to MuJoCo sim
  joint_targets = reference.joints - data.qpos  # reward computed by index
      |
      v
Policy learns: ctrl[t] = f(obs including qpos mismatch)
      |
      v
SCAMPER/rollout.py
  ctrl = policy(obs)  # muscle activations
  ctrl[:, 5] -> Triceps_Lateral
  ctrl[:, 8] -> Biceps_Long
      |
      v
EMG correlation: r = corrcoef(sim_mean, emg_mean)
```

**Every arrow is index-based. A joint order mismatch at the top propagates silently through the entire pipeline.**
