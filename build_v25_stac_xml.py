#!/usr/bin/env python
"""Build a STAC-safe copy of v25: v24's arm/hand/neck/head rig (build_v24_stac_xml.py's
output) plus a joystick body added the same way v22 added one, so v25 can get
a real STAC IK fit tracking both the arm/hand AND the joystick together
(unlike v24 alone, which has no joystick body at all).

Joystick geometry/placement is NOT guessed -- it comes from
compute_joystick_geometry_v25.py, which re-runs v22's static-calibration
recipe (static_joystick_labels_withbottom.v001.slp, the same physical rig)
through v24's OWN averaged registration rotation (register_v25_mocap.py,
12 gauge-reliable trials). shaft_length/ball_offset are physical constants
of the rig and match v21/v22 exactly (14.363mm / 1.822mm); joystick_base's
position (bottom_m) is new to v24's frame.

Orientation is deliberately UNTILTED (joystick child body has no quat), even
though the raw calibration output implies a ~36.7-degree tilt in v24's frame.
This mirrors vnl-playground's own documented lesson from v22
(docs/2026-07-17-v22-joystick-camera-fix-and-first-successful-eval.md): every
v22 physics run that ever trained cleanly used an untilted joystick, and the
one time a calibration-derived tilt was carried into the model it reintroduced
instability that was never fully resolved. STAC doesn't need the tilt either:
x_slide/y_slide (+-6mm) absorb any residual between this calibration-grounded
anchor and each frame's actual registered js_base target, same mechanism
v22's own build script relies on ("essentially zero mean residual... with
tight per-trial scatter").

Shaft is a CAPSULE, not a cylinder (v22's STAC-safe copy still uses
`type="cylinder"`, inherited from the original janelia_model v22 XML) --
per Eric's explicit request. Two independent reasons this matters, not just
one: (1) capsule-vs-mesh collision IS supported by mjx's narrowphase where
cylinder-vs-mesh throws NotImplementedError (see build_v22_stac_xml.py's own
comment on exactly this); (2) even where cylinder collision does compile, it
lacks capsule's cheap analytic contact solution and is a materially heavier
per-contact cost in the broadphase -- with v24's ~150 collision proxies
already in the scene, that cost difference is the "collisions slowing things
down" risk Eric flagged directly for the physics side. Kept consistent here
even though STAC itself never evaluates a single contact, since this STAC
copy is meant to validate the same collision-pair combination the eventual
physics copy will use.

The ball is a very short CAPSULE too, not a sphere: mjx's own collision-pair
table (collision_driver.py's _COLLISION_FUNC) has no (SPHERE, ELLIPSOID)
entry -- confirmed directly, mjx.put_model() throws NotImplementedError the
moment the ball's default sphere geom meets any of the hand's ellipsoid
proxies -- but DOES have (CAPSULE, ELLIPSOID). A capsule with a 0.1mm
half-length and the same 2mm radius is visually and physically indistinguishable
from the sphere it replaces.

Collision bits: v24's ~150 "_col" proxies are UNIFORMLY contype="1"
conaffinity="0" (checked directly -- these are the only two combos anywhere
in the file, the other being 0/0 for inert visual bone meshes). Naively
giving the joystick a conaffinity bit that overlaps contype="1" would make it
collide with EVERY one of those ~150 proxies (spine, ribs, skull, ...), not
just the hand -- exactly the uncontrolled broadphase blowup Eric warned about.
Fixed the same way vnl-playground's own placeholder v25 xml already fixed it
(docs/2026-07-17-overnight-handoff.md's "Bitmask overlap" bug): retag ONLY
the 32 hand-region proxies (Phalanx_hand_*/Metacarpal_hand_*/Sesamoid_hand_*/
Multangular_right) to contype="4" conaffinity="8", and give the joystick
contype="8" conaffinity="4" -- disjoint from the untouched rest-of-skeleton's
1/0, so the joystick collides with the hand only.

Usage: .venv/bin/python3 build_v25_stac_xml.py
"""
import re
from pathlib import Path

SRC = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v24_stac.xml")
DST = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v25_stac.xml")

# From compute_joystick_geometry_v25.py (12-reliable-trial average, v24 frame).
JOYSTICK_BASE_POS = "-0.01114158 -0.02948650 0.04407920"
SHAFT_LEN = 0.014363   # bottom -> base, physical rig constant (matches v21/v22 exactly)
BALL_OFFSET = 0.001822  # base -> ball
SHAFT_HALF_LEN = 0.0071815  # = SHAFT_LEN / 2, capsule half-length convention (matches v21/v22 shaft geom)

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
        'mujoco model="mouse_forelimb_right_janelia_arm_hand_v25_stac"',
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
