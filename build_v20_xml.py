"""Build v20 XML on top of v19: graft the 1cm spring-loaded XY joystick from
janelia_model/new_joystick_model/arm_model_and_joystick.xml onto the v19 arm+hand
model, unchanged otherwise.

v20 = v19 (unchanged: proximal arm, articulated hand, all joints/names) + joystick:
  - `joystick_base` body, fixed in world (not attached to the arm chain), same
    geometry/spring params as the reference joystick model:
      - x_slide, y_slide: slide joints, range +-0.006 m, spring-centered like a
        real joystick -- passive, no actuators (the hand drives it via contact
        only). Stiffness/damping use the real-joystick-measured coefficients
        (per Eric): x_slide stiffness=14.8 N/m damping=0.0099 Ns/m, y_slide
        stiffness=11.2 N/m damping=0.0082 Ns/m (anisotropic x/y, replacing the
        original reference's shared isotropic 10 N/m / 0.0099 Ns/m guess).
      - circular_base geom (cylinder) + nested `joystick` body with the
        shaft (cylinder) + ball (sphere) geoms
    Position shifted +3mm from the reference's (0.024, 0.012, 0.055) along the
    contact-normal direction: at the reference position, the resting forearm
    (`ulna` body) penetrates the joystick shaft by up to 1.14mm in v19's geometry
    (the reference's position was tuned against an older, pre-"scapula tilt fix"
    arm -- clavicle pos/quat match v19 exactly, but scapula quat differs by a few
    degrees, enough to shift the resting forearm into the joystick). Verified via
    contact query (ncon=0 at the shifted position vs ncon=2 at the original).
  - contact enabled between hand/arm geoms and the joystick: default geom
    contype/conaffinity flipped from "0 0" (v19: contacts fully disabled, only
    needed for IK) to "2 0" (matches the reference model's convention), so arm
    geoms collide with the joystick's contype=1/conaffinity=2 geoms but not with
    each other (2&0 both ways for any two arm geoms == 0, no self-collision)
  - jointpos sensors for x_slide/y_slide (matching the reference model)
  - all <actuator> position servos REMOVED (per Eric): v20's arm joints are
    passive, undriven DOFs for now -- no position control fighting the dynamics

Known issue found + resolved during this build: v18/v19 (before any joystick
work) are numerically unstable under real mj_step time-integration -- verified
this predates the joystick entirely (v18 alone blows up the same way). The
pristine reference model is stable under the same test. Diagnosed two
contributing factors: (1) v17/v18/v19's joint defaults use much weaker
armature/damping (4e-10/1e-6) than the reference (4e-8/1e-5), and (2) v18/v19
drive the arm with position-servo actuators instead of the reference's real
`<muscle>` tendons. Per Eric, actuators are simply removed here (option 3 was
not requested). This model remains stable in testing (500-step passive rollout,
final max|qvel| ~0.66) ONLY when the simulating script overrides the integrator
and solver exactly as the reference script (joystick_force_displacement.py)
does -- this is NOT baked into the XML's <option> (matching the reference's own
convention of overriding at the Python level, not in XML):
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 50
    model.opt.ls_iterations = 50
Any script that steps this model must set these, or expect instability.

See scratch/plans/PLAN_v19_hand_model.md for the v19 design (unchanged here), and
janelia_model/new_joystick_model/{arm_model_and_joystick.xml,joystick_only.xml,
joystick_force_displacement.py} for the joystick source and the proven solver recipe.

Output: models/v20_joystick/mouse_forelimb_right_janelia_arm_hand_v20_joystick.xml
  (self-contained, and mirrored to /mnt/vast/niyo/janelia_model/v20_joystick/)
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

V19_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v19.xml")
V19_MESHDIR = Path("/mnt/vast/niyo/stac-mjx/models/v19")

V20_NAME = "mouse_forelimb_right_janelia_arm_hand_v20_joystick.xml"
V20_DIRS = [
    Path("/mnt/vast/niyo/stac-mjx/models/v20_joystick"),
    Path("/mnt/vast/niyo/janelia_model/v20_joystick"),
]

ARM_STLS = ["clavicle_right.stl", "scapula_right.stl", "humerus_right.stl",
            "radius_right.stl", "ulna_right.stl"]

JOYSTICK_BODY = '''    <!-- ========================= -->
    <!-- 1 cm total-height joystick (from janelia_model/new_joystick_model/
         arm_model_and_joystick.xml): fixed in world, not attached to the arm -->

    <!-- ========================= -->
    <body name="joystick_base" pos="0.021645 0.014677 0.054980">
      <!-- Real-joystick-measured spring/damping coefficients (per Eric):
           Kx=0.0148 N/mm, Ky=0.0112 N/mm -> x1000 for MuJoCo's N/m slide-joint
           stiffness; Cx=0.0099 Ns/m, Cy=0.0082 Ns/m already match MuJoCo's
           native Ns/m damping convention directly (no scaling needed). -->
      <joint name="x_slide" type="slide" axis="1 0 0" limited="true" range="-0.006 0.006" damping="0.0099" stiffness="14.8"/>
      <joint name="y_slide" type="slide" axis="0 1 0" limited="true" range="-0.006 0.006" damping="0.0082" stiffness="11.2"/>

      <!-- small circular base -->
      <geom name="circular_base" type="cylinder" size="0.0015 0.00025" pos="0 0 0"
            rgba="0.6 0.6 0.6 1" contype="1" conaffinity="2"/>

      <!-- shaft + ball, total ~ 1 cm -->
      <body name="joystick" pos="0 0 0.0">
        <!-- shaft: radius 0.0005 m, half-length 0.0043 m (total 0.0086 m) -->
        <geom name="joystick_geom" type="cylinder" size="0.0005 0.0043" pos="0 0 0.004"
              rgba="0 0 1 1" contype="1" conaffinity="2"/>
        <!-- ball radius 0.001 m sitting on top (adds 0.002 m) -->
        <geom name="joystick_ball" type="sphere" size="0.001" pos="0 0 0.009"
              rgba="0 0 1 1" contype="1" conaffinity="2"/>
        <inertial mass="0.001" pos="0 0 0.004" diaginertia="1e-7 1e-7 1e-7"/>
      </body>
    </body>

'''

JOYSTICK_SENSOR = '''  <sensor>
    <jointpos name="x_slide_pos" joint="x_slide"/>
    <jointpos name="y_slide_pos" joint="y_slide"/>
  </sensor>

'''


def main():
    xml = V19_XML.read_text()

    # ---- 1. model name / header ----
    xml = xml.replace(
        '<mujoco model="mouse_forelimb_right_hand_v19">',
        '<mujoco model="mouse_forelimb_right_hand_joystick_v20">',
    )
    old_first_line = "    Mouse right forelimb musculoskeletal model with an articulated hand.\n"
    new_first_line = (
        "    Mouse right forelimb musculoskeletal model with an articulated hand\n"
        "    AND a 1cm spring-loaded XY joystick (from janelia_model/new_joystick_model/\n"
        "    arm_model_and_joystick.xml), fixed in world, contacted by the hand.\n"
    )
    if old_first_line not in xml:
        raise RuntimeError("v19 header first line not found verbatim")
    xml = xml.replace(old_first_line, new_first_line)

    # ---- 2. enable contact: default geom + hand_visual class contype/conaffinity
    #         0/0 (v19: contacts fully disabled) -> 2/0 (v20: collides with the
    #         joystick's contype=1/conaffinity=2 geoms, but not with itself) ----
    old_default = (
        '    <geom contype="0" conaffinity="0"/>\n'
        '    <default class="hand_visual">\n'
        '      <geom contype="0" conaffinity="0"/>\n'
        '    </default>\n'
    )
    new_default = (
        '    <geom contype="2" conaffinity="0"/>\n'
        '    <default class="hand_visual">\n'
        '      <geom contype="2" conaffinity="0"/>\n'
        '    </default>\n'
    )
    if old_default not in xml:
        raise RuntimeError("v19 default geom block not found verbatim")
    xml = xml.replace(old_default, new_default)

    # ---- 3. graft the joystick body as a top-level worldbody child, before "ground" ----
    old_worldbody_open = (
        "  <worldbody>\n"
        '    <light directional="true"/>\n'
        '    <body name="ground" euler="0 0 0">\n'
    )
    new_worldbody_open = (
        "  <worldbody>\n"
        '    <light directional="true"/>\n'
        + JOYSTICK_BODY
        + '    <body name="ground" euler="0 0 0">\n'
    )
    if old_worldbody_open not in xml:
        raise RuntimeError("worldbody opening block not found verbatim")
    xml = xml.replace(old_worldbody_open, new_worldbody_open)

    # ---- 4. remove the whole <actuator> block: v20's joints are passive, no
    #         position servos fighting the dynamics (per Eric) ----
    n_before = len(re.findall(r'<position ', xml))
    xml = re.sub(r'\n  <actuator>.*?</actuator>\n', '\n', xml, flags=re.DOTALL)
    if len(re.findall(r'<position ', xml)) != 0:
        raise RuntimeError("actuator block removal left position servos behind")
    print(f"  Removed <actuator> block ({n_before} position servos stripped).")

    # ---- 5. add sensor block (v19 has none) before </mujoco> ----
    old_tail = "\n</mujoco>\n"
    new_tail = "\n" + JOYSTICK_SENSOR + "</mujoco>\n"
    if old_tail not in xml:
        raise RuntimeError("mujoco tail not found verbatim")
    xml = xml.replace(old_tail, new_tail)

    # ---- Write self-contained copies ----
    import xml.etree.ElementTree as ET
    import os

    for out_dir in V20_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        # copy 41 hand OBJs + 5 arm STLs
        for f in V19_MESHDIR.iterdir():
            if f.is_file():
                dst = out_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
        out_path = out_dir / V20_NAME
        local_xml = xml.replace(f'meshdir="{V19_MESHDIR}"', f'meshdir="{out_dir}"')
        out_path.write_text(local_xml)

        # strict well-formedness check
        try:
            ET.parse(str(out_path))
        except ET.ParseError as e:
            raise RuntimeError(f"{out_path} is not well-formed XML: {e}") from e

        os.environ["MUJOCO_GL"] = "disable"
        import mujoco
        import numpy as np
        m = mujoco.MjModel.from_xml_path(str(out_path))
        print(f"Wrote + validated: {out_path}")
        print(f"  nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} nmesh={m.nmesh} "
              f"nu={m.nu} nsensor={m.nsensor}")
        for jname in ("x_slide", "y_slide"):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"joint {jname} missing!")
        assert m.nu == 0, "expected zero actuators in v20"
        print("  x_slide/y_slide present, nu=0 (no actuators) confirmed.")

        # neutral-pose contact check + real mj_step stability check, using the
        # documented required solver override (see module docstring)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        print(f"  ncon at neutral pose: {d.ncon} (expect 0 -- joystick clear of forearm)")
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        m.opt.iterations = 50
        m.opt.ls_iterations = 50
        d2 = mujoco.MjData(m)
        mujoco.mj_forward(m, d2)
        blew_up = False
        for step in range(500):
            mujoco.mj_step(m, d2)
            if np.isnan(d2.qvel).any() or np.abs(d2.qvel).max() > 1e3:
                blew_up = True
                print(f"  UNSTABLE at step {step}: max|qvel|={np.abs(d2.qvel).max():.4e}")
                break
        if not blew_up:
            print(f"  500-step mj_step rollout stable (implicitfast+Newton). "
                  f"final max|qvel|={np.abs(d2.qvel).max():.4f}")
        else:
            raise RuntimeError("v20 model is dynamically unstable even with the documented solver override!")


if __name__ == "__main__":
    main()
