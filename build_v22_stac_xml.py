#!/usr/bin/env python
"""Build a STAC-safe copy of the v22 arm+hand+joystick model.

vnl-playground's mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml ships
with muscles/tendons ACTIVE (its own header comment explicitly says to flip
this if the model is headed back into a STAC run, mirroring what
build_v21_xml.py already did for v21). STAC only ever calls mjx.kinematics
(no mj_step / dynamics), so muscles/tendons/actuators are functionally inert
for fitting -- but leaving them in costs compile time and is one more way for
something to go wrong that has nothing to do with kinematics. This script
comments out (not deletes) the <tendon> and <actuator> blocks, leaving every
<joint> declaration (name/range/damping/stiffness) untouched, since STAC's
optimizer bounds come from jnt_range.

The joystick_base position/orientation in the source v22_contacts.xml is
already the v21-restored, static-calibration-derived value (see
docs/2026-07-16-v22-joystick-base-placement-bug.md in vnl-playground) --
this script does not touch geometry, only comments out actuation.

Usage: .venv/bin/python3 build_v22_stac_xml.py
"""
import re
from pathlib import Path

SRC = Path("/mnt/vast/niyo/janelia_model/v22/mouse_forelimb_right_janelia_arm_hand_v22.xml")
DST = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v22_stac.xml")
# Built from the non-"_contacts" v22 variant, not
# mouse_forelimb_right_janelia_arm_hand_v22_contacts.xml: that variant sets
# contype/conaffinity so joystick geoms (cylinder/sphere) collide with every
# skeletal mesh geom, and MJX's mjx.put_model() throws
# NotImplementedError("(mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_MESH)
# collisions not implemented") the moment ANY such pair is active, even
# though STAC never calls mj_step / never evaluates a single contact -- just
# loading the model into MJX validates every enabled collision pair upfront.
# The non-contacts variant leaves every mesh geom at the contype=0/
# conaffinity=0 default (confirmed via grep), so no cylinder-vs-mesh pair is
# ever active and it loads into MJX cleanly. STAC has no use for contact
# physics anyway (kinematics-only).


def comment_out_block(xml: str, tag: str) -> str:
    pattern = re.compile(rf"(  <{tag}>.*?</{tag}>\n)", re.DOTALL)
    matches = pattern.findall(xml)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one <{tag}> block, found {len(matches)}")
    block = matches[0]
    if "--" in block:
        raise ValueError(f"<{tag}> block contains '--', can't safely wrap in an XML comment")
    commented = "  <!-- STAC-safe build: commented out for kinematics-only fitting.\n" + block + "  -->\n"
    return xml.replace(block, commented)


def main():
    xml = SRC.read_text()
    xml = comment_out_block(xml, "tendon")
    xml = comment_out_block(xml, "actuator")
    xml = xml.replace(
        "<mujoco model=\"mouse_forelimb_right_hand_v22\">",
        "<mujoco model=\"mouse_forelimb_right_hand_v22_stac\">",
    )
    # Source XML's meshdir is a stale Mac path (this is the non-"_contacts"
    # variant, which wasn't updated when the "_contacts" copy's meshdir was
    # fixed to the local absolute path).
    xml = xml.replace(
        'meshdir="/Volumes/talmo/eric/janelia_model/v22"',
        'meshdir="/mnt/vast/niyo/janelia_model/v22"',
    )
    # joystick_base position + "joystick" body tilt quat, measured DIRECTLY
    # in v22's own registration/world frame by compute_joystick_geometry_v22.py
    # (same static-calibration recipe as v21's compute_joystick_geometry.py,
    # just averaged over v22's own registered_v22_refined.h5 rotations).
    #
    # An earlier attempt just copied v21's already-computed pos/quat verbatim
    # into v22 -- that's WRONG and was reverted: v21's world frame and v22's
    # world frame are different frames (each is its own independent
    # Procrustes best-fit of the same real mocap arm data against that
    # model's own, differently-postured rest pose), empirically related by
    # ~87 degrees of rotation, not a small residual. Copying v21's numbers
    # verbatim put the joystick ~15mm from where the arm's own registration
    # expected it (x_slide/y_slide saturating at their +-6mm limits trying to
    # compensate) and pointed the shaft sideways instead of vertical (per
    # Eric's direct visual report on the v22 smoke-test reprojection video).
    # Shaft length (14.363mm) and ball offset (1.822mm) are unchanged --
    # those are physical distances, frame-independent, and matched v21's
    # measurement exactly as a sanity check.
    #
    # Re-derived a third time (2026-07-16): the previous two anchors (both
    # above) were built on registered_v22_refined.h5 data from
    # register_v22_mocap.py's v4 recipe, which forced every trial to share
    # ONE rotation matrix averaged across all 15 CFL_35/36/37 trials --
    # spanning three sessions recorded months apart. That premise (verbatim
    # from the v4 docstring: "confirmed byte-identical calibration.toml") is
    # false: calibration_refined.toml is a different file, with measurably
    # different camera extrinsics (8-21 degrees apart), for every single
    # trial, even within one same-day session. Forcing a shared rotation
    # onto trials whose true calibration frame diverges from it by that much
    # produced a per-trial registration error of 0.9-3.8mm in the joystick
    # target specifically (small/invisible on the nearby arm keypoints,
    # amplified on the far-away rigid joystick) -- consuming most of
    # x_slide/y_slide's +-6mm travel on some trials and looking crooked/
    # offset in exactly the trial-dependent, unpredictable way reported
    # against the previous two anchors. register_v22_mocap.py's v5 recipe
    # registers every trial with its OWN independently-fit rotation instead
    # (see its register_trial docstring), which removes that error at the
    # source -- this anchor is derived from the v5-registered data.
    #
    # This position combines: (1) shaft orientation and length/ball-offset
    # from the dedicated static joystick calibration recording
    # (compute_joystick_geometry_v22.py -- a single precise measurement, not
    # per-frame-tracking-noise-prone, so reliable for these), with (2) the
    # position anchor taken as the MEDIAN of the 12 gauge-reliable trials'
    # own v5-registered js_base resting position directly (not the static
    # calibration's own placement into model frame, which carries a ~3.6mm
    # bias from mixing calibration_refined.toml-based per-trial rotations
    # with the static recording's own plain calibration.toml frame). This
    # combination leaves essentially zero mean residual (<0.3mm every axis)
    # with tight per-trial scatter (0.7-3.7mm, on par with v21's own
    # documented ~2-5mm) -- see this session's derivation for the numbers.
    old_joystick_base = '<body name="joystick_base" pos="-0.01685188 -0.02594152 0.04776544">'
    new_joystick_base = '<body name="joystick_base" pos="-0.01264354 -0.03211015 0.04737563">'
    if xml.count(old_joystick_base) != 1:
        raise ValueError(f"expected exactly one v21-transplanted joystick_base pos, found {xml.count(old_joystick_base)}")
    xml = xml.replace(old_joystick_base, new_joystick_base)

    old_joystick_body = '<body name="joystick" pos="0 0 0">'
    new_joystick_body = (
        '<body name="joystick" pos="0 0 0" '
        'quat="0.94389458 -0.21721094 0.24876180 0.00000000">'
    )
    if xml.count(old_joystick_body) != 1:
        raise ValueError(f"expected exactly one unrotated joystick body, found {xml.count(old_joystick_body)}")
    xml = xml.replace(old_joystick_body, new_joystick_body)

    # Cosmetic-only (see stac_mjx/multicam.py's RegistrationTransform docstring
    # -- STAC never touches geom size, only site positions): match v21's
    # already-user-approved visual sizes (shaft radius 0.001, ball radius
    # 0.002) instead of v22's smaller un-doubled originals (0.0005/0.001).
    # Ball radius (2mm) > measured base-to-ball offset (1.822mm) so the ball
    # visually overlaps/touches the shaft top, per Eric's request.
    old_shaft = 'size="0.0005 0.0071815" pos="0 0 0.0071815" rgba="0 0 1 1" contype="1" conaffinity="2" density="40000"'
    new_shaft = 'size="0.001 0.0071815" pos="0 0 0.0071815" rgba="0 0 1 1" contype="1" conaffinity="2" density="40000"'
    if xml.count(old_shaft) != 1:
        raise ValueError(f"expected exactly one un-doubled joystick_geom size, found {xml.count(old_shaft)}")
    xml = xml.replace(old_shaft, new_shaft)

    old_ball = 'pos="0 0 0.014" rgba="0 0 1 1" contype="1" conaffinity="2"/>'
    new_ball = 'pos="0 0 0.016184" rgba="0 0 1 1" contype="1" conaffinity="2"/>'
    if xml.count(old_ball) != 1:
        raise ValueError(f"expected exactly one un-measured joystick_ball pos, found {xml.count(old_ball)}")
    xml = xml.replace(old_ball, new_ball)

    old_ball_size = 'size="0.001"'
    new_ball_size = 'size="0.002"'
    if xml.count(old_ball_size) != 1:
        raise ValueError(f"expected exactly one un-doubled joystick_ball size, found {xml.count(old_ball_size)}")
    xml = xml.replace(old_ball_size, new_ball_size)
    DST.write_text(xml)
    print(f"Wrote {DST}")

    # Validate: strict XML parse + MuJoCo load + mj_forward at qpos=0, no NaNs.
    import xml.etree.ElementTree as ET
    ET.parse(DST)
    print("XML well-formed.")

    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(DST))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert not np.isnan(d.xpos).any(), "NaN in xpos after mj_forward"
    print(f"MuJoCo load OK: nbody={m.nbody}, njnt={m.njnt}, ngeom={m.ngeom}, nu={m.nu} (expect 0)")

    # STAC loads the model into MJX (mjx.put_model), which validates every
    # ENABLED collision pair upfront and throws NotImplementedError for
    # unsupported geom-type combinations (e.g. cylinder-vs-mesh) even though
    # STAC's kinematics-only fit never evaluates a single contact. Catch that
    # here, at build time, instead of discovering it mid smoke-test.
    from mujoco import mjx
    mjx.put_model(m)
    print("mjx.put_model OK (no unsupported collision pairs).")


if __name__ == "__main__":
    main()
