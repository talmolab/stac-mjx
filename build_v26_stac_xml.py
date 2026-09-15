#!/usr/bin/env python
"""Build a STAC-safe copy of v26: identical to v25 (build_v25_stac_xml.py)
except for the joystick_base anchor position, which is re-derived to fix a
systematic mislocation found in v25's reprojection videos ("the joystick
doesn't match the mocap at all").

Root cause (see compute_joystick_geometry_v26.py's docstring for the full
diagnosis): v25's anchor came from triangulating ONE static calibration photo
and transforming it via a rotation averaged across 12 trials spanning three
different recording sessions. Compared against register_v25_mocap.py's own
js_base_resting_position attr (each trial's own median js_base position,
already in v25's model frame -- i.e. the actual thing STAC is trying to fit),
v25's anchor is off by a consistent ~5-9mm in every single one of the 15
trials, in every session -- a uniform bias, not per-session rig drift. Since
joystick_base only has +-6mm of x_slide/y_slide and no rotation/z DOF, that
bias pins the slide joints at their limit in most trials, capping the fit
before it can reach the real target.

v26 keeps the untilted, 2-DOF design exactly as-is (still no evidence STAC
needs the tilt, still following v22's stability lesson -- see
build_v25_stac_xml.py's docstring) and keeps the physical shaft_length/
ball_offset constants unchanged. The ONLY change is anchoring joystick_base
on the mean js_base_resting_position across the same 12 reliable trials,
computed directly from mocap already registered for v25 (no new registration
needed) instead of from the separate static-photo pipeline. This drops
residual per-trial scatter to ~0.2-2.7mm (compute_joystick_geometry_v26.py's
own output) -- comfortably inside the unchanged +-6mm range.

Usage: .venv/bin/python3 build_v26_stac_xml.py
"""
import re
from pathlib import Path

SRC = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v24_stac.xml")
DST = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v26_stac.xml")

# From compute_joystick_geometry_v26.py (mean js_base_resting_position across
# the 12 reliable trials, v24/v25 model frame -- NOT the static-photo pipeline
# v25 used, see module docstring above).
JOYSTICK_BASE_POS = "-0.00569542 -0.02594662 0.04388028"
SHAFT_LEN = 0.014363   # bottom -> base, physical rig constant (unchanged from v21/v22/v25)
BALL_OFFSET = 0.001822  # base -> ball (unchanged from v21/v22/v25)
SHAFT_HALF_LEN = 0.0071815  # = SHAFT_LEN / 2, capsule half-length convention (unchanged)

HAND_COL_BODIES = [
    "Metacarpal_hand_1_right", "Metacarpal_hand_2_right", "Metacarpal_hand_3_right",
    "Metacarpal_hand_4_right", "Metacarpal_hand_5_right", "Multangular_right",
    "Phalanx_hand_1_1_right", "Phalanx_hand_1_2_right", "Phalanx_hand_1_3_right",
    "Phalanx_hand_1_4_right", "Phalanx_hand_1_5_right",
    "Phalanx_hand_2_1_right", "Phalanx_hand_2_2_right", "Phalanx_hand_2_3_right",
    "Phalanx_hand_2_4_right", "Phalanx_hand_2_5_right",
    "Phalanx_hand_3_2_right", "Phalanx_hand_3_3_right", "Phalanx_hand_3_4_right", "Phalanx_hand_3_5_right",
    "Sesamoid_hand_a_right", "Sesamoid_hand_b_right", "Sesamoid_hand_c_right", "Sesamoid_hand_d_right",
    "Sesamoid_hand_e_right", "Sesamoid_hand_f_right", "Sesamoid_hand_g_right", "Sesamoid_hand_h_right",
    "Sesamoid_hand_i_right", "Sesamoid_hand_j_right", "Sesamoid_hand_k_right",
    "Sesamoid_hand_l_right", "Sesamoid_hand_m_right", "Sesamoid_hand_n_right",
]

JOYSTICK_BLOCK = f'''    <body name="joystick_base" pos="{JOYSTICK_BASE_POS}">
      <joint name="x_slide" type="slide" axis="1 0 0" limited="true" range="-0.006 0.006" damping="0.0099" stiffness="14.8"/>
      <joint name="y_slide" type="slide" axis="0 1 0" limited="true" range="-0.006 0.006" damping="0.0082" stiffness="11.2"/>
      <geom name="circular_base" type="capsule" size="0.0015 0.00025" pos="0 0 0" rgba="0.6 0.6 0.6 1" contype="8" conaffinity="4"/>
      <body name="joystick" pos="0 0 0">
        <geom name="joystick_geom" type="capsule" size="0.001 {SHAFT_HALF_LEN}" pos="0 0 {SHAFT_HALF_LEN}" rgba="0 0 1 1" contype="8" conaffinity="4" density="40000"/>
        <geom name="joystick_ball" type="capsule" size="0.002 0.0001" pos="0 0 {SHAFT_LEN + BALL_OFFSET:.6f}" rgba="0 0 1 1" contype="8" conaffinity="4"/>
      </body>
    </body>
'''


def retag_hand_collision_bits(xml: str) -> str:
    old = 'contype="1" conaffinity="0"'
    new = 'contype="4" conaffinity="8"'
    n_retagged = 0
    for body in HAND_COL_BODIES:
        pattern = re.compile(rf'(<geom name="{re.escape(body)}_col"[^>]*?){re.escape(old)}')
        matches = pattern.findall(xml)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {body}_col geom with {old}, found {len(matches)}")
        xml, count = pattern.subn(rf"\1{new}", xml, count=1)
        n_retagged += count
    if n_retagged != len(HAND_COL_BODIES):
        raise ValueError(f"retagged {n_retagged} geoms, expected {len(HAND_COL_BODIES)}")
    return xml


def insert_joystick_block(xml: str) -> str:
    marker = '  <worldbody>\n'
    if xml.count(marker) != 1:
        raise ValueError(f"expected exactly one {marker!r}, found {xml.count(marker)}")
    return xml.replace(marker, marker + JOYSTICK_BLOCK, 1)


def main():
    xml = SRC.read_text()
    xml = xml.replace(
        'mujoco model="mouse_forelimb_right_janelia_arm_hand_v24_stac"',
        'mujoco model="mouse_forelimb_right_janelia_arm_hand_v26_stac"',
        1,
    )
    xml = retag_hand_collision_bits(xml)
    xml = insert_joystick_block(xml)

    DST.write_text(xml)
    print(f"Wrote {DST}")

    import xml.etree.ElementTree as ET
    ET.parse(DST)
    print("XML well-formed.")

    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(DST))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert not np.isnan(d.xpos).any(), "NaN in xpos after mj_forward"
    print(f"MuJoCo load OK: nbody={m.nbody}, njnt={m.njnt}, ngeom={m.ngeom}, nu={m.nu} (expect 0), ntendon={m.ntendon} (expect 0)")

    jsb_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "joystick_base")
    js_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "joystick")
    print(f"joystick_base xpos at rest: {d.xpos[jsb_bid]}")
    print(f"joystick xpos at rest: {d.xpos[js_bid]}")

    from mujoco import mjx
    mjx.put_model(m)
    print("mjx.put_model OK (no unsupported collision pairs).")

    kp_names_to_check = [
        "humerus_right", "ulna_right", "radius_right", "N_L_C_right",
        "Metacarpal_hand_1_right", "Phalanx_hand_1_1_right", "Phalanx_hand_2_1_right",
        "Phalanx_hand_1_2_right", "Phalanx_hand_2_2_right", "Phalanx_hand_3_2_right",
        "Phalanx_hand_1_5_right", "Phalanx_hand_2_5_right", "Phalanx_hand_3_5_right",
        "joystick_base", "joystick",
    ]
    for name in kp_names_to_check:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"body {name} not found in compiled model"
    print(f"All {len(kp_names_to_check)} arm/hand/joystick keypoint target bodies present.")

    from dm_control import mjcf
    root = mjcf.from_path(str(DST))
    mjcf.Physics.from_mjcf_model(root)
    print("dm_control mjcf.from_path + Physics.from_mjcf_model OK.")


if __name__ == "__main__":
    main()
