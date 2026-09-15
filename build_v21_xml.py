"""Build v21 XML: Eric's corrected arm+hand (reverted scapula/clavicle tilt,
restored muscle-driven shoulder/elbow, mesh-shape densities) + v20's joystick,
with muscles/tendons/motors commented out for STAC registration (STAC only
ever uses mj_forward/mjx.kinematics -- actuators are 100% inert for it; see
scratch/plans/PLAN_v19_hand_model.md's successor plan for v21 in the plan
file used this session, and STAC_EMG_DIAGNOSTIC_REPORT.md for why joint
range/order must NOT be touched here).

Source of truth for the arm+hand: scratch/v21_reference_from_eric.xml (pasted
verbatim from Eric's own locally-iterated XML, only the meshdir path differs
since his copy pointed at a local Mac path).

v21 = Eric's reference arm+hand (unchanged structurally) + v20's joystick
      body (graft, same pattern as build_v20_xml.py) + tendon/actuator/
      lengthrange commented out (not deleted -- trivially reversible for
      future forward-dynamics work).

The joystick_base position below is a PLACEHOLDER carried over from v20's
already-computed clearance fix -- it does NOT yet reflect the new v21 arm
geometry (scapula reverted) and does NOT yet reflect the real mocap-derived
joystick position. Phase C of the plan (register_v21_mocap.py) computes the
real position from registered CFL mocap data; this script will be re-run with
that value once available.

Output: models/mouse_forelimb_right_janelia_arm_hand_v21.xml
  (self-contained mirrors: models/v21/, /mnt/vast/niyo/janelia_model/v21/)
"""
from __future__ import annotations
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REFERENCE_XML = Path("/mnt/vast/niyo/stac-mjx/scratch/v21_reference_from_eric.xml")
V19_MESHDIR = Path("/mnt/vast/niyo/stac-mjx/models/v19")  # source of the 41 hand OBJs + 5 arm STLs

V21_NAME = "mouse_forelimb_right_janelia_arm_hand_v21.xml"
V21_CANONICAL = Path(f"/mnt/vast/niyo/stac-mjx/models/{V21_NAME}")
V21_DIRS = [
    Path("/mnt/vast/niyo/stac-mjx/models/v21"),
    Path("/mnt/vast/niyo/janelia_model/v21"),
]

# Geometry, position, and orientation derived from a DEDICATED static
# calibration (static_joystick_labels_withbottom.v001.slp, js_ball/js_base/
# js_bottom labeled once in all 3 cameras -- see compute_joystick_geometry.py),
# not from noisy per-trial js_base/js_ball tracking during mouse behavior
# videos (the earlier approach -- averaging or even single-point-anchoring
# those two isolated, unconstrained points was never going to recover true
# shaft length or orientation, only a noisy position estimate). js_base = the
# base of the ball at the TOP of the shaft; js_bottom = where the shaft
# connects to the rig's mount, roughly the BOTTOM -- these two anchor the
# joystick's true physical size and orientation.
#
# Triangulating the static calibration (all 15 CFL_35/36/37 trials share one
# identical calibration.toml) and transforming into model frame via a
# rotation-averaged (SVD/chordal mean, over the 12 gauge-reliable trials'
# own registration_rotation) registration transform gives:
#   shaft length (bottom->base) = 14.363mm (model previously used 8.6mm -- too
#     short)
#   ball offset (base->ball)    = 1.822mm (model previously used 0.9mm)
#   shaft direction (bottom->base) = 39.4 degrees off model-frame +Z -- NOT
#     vertical. The earlier assumption of a rigidly-vertical joystick (based
#     on the mechanical x_slide/y_slide-only, no-rotation-DOF design) was an
#     incorrect inference: the DOF structure only tells us the joystick
#     doesn't change its tilt during a trial, not that its fixed tilt is
#     zero. This static calibration is the actual measurement of that fixed
#     tilt.
# joystick_base is placed at the BOTTOM (mount) point and stays unrotated, so
# x_slide/y_slide keep translating in world X/Y (matching the physical rig:
# "moving it should just move it along the x,y plane"). The child "joystick"
# body carries a fixed quat rotating its local +Z (the shaft axis in local
# geometry) to the true, measured shaft direction.
JOYSTICK_BODY = '''    <!-- ========================= -->
    <!-- Joystick geometry from compute_joystick_geometry.py (static
         base/ball/bottom calibration, not per-trial mocap tracking): shaft
         14.363mm (bottom to base), ball 1.822mm above base, shaft tilted
         39.4 degrees off vertical (not upright, per direct measurement).
         Fixed in world, not attached to the arm. joystick_base sits at the
         bottom/mount point (translates in world x/y only via
         x_slide/y_slide); the child "joystick" body carries the fixed tilt
         rotation so the shaft+ball geometry (unchanged local frame) point
         the right physical direction. -->
    <body name="joystick_base" pos="0.02754416 0.00213547 0.04439829">
      <!-- Real-joystick-measured spring/damping coefficients (per Eric):
           Kx=0.0148 N/mm, Ky=0.0112 N/mm -> x1000 for MuJoCo's N/m slide-joint
           stiffness; Cx=0.0099 Ns/m, Cy=0.0082 Ns/m already match MuJoCo's
           native Ns/m damping convention directly (no scaling needed). These
           don't matter for STAC (no dynamics) but keep the model consistent/
           reusable for future dynamics work. -->
      <joint name="x_slide" type="slide" axis="1 0 0" limited="true" range="-0.006 0.006" damping="0.0099" stiffness="14.8"/>
      <joint name="y_slide" type="slide" axis="0 1 0" limited="true" range="-0.006 0.006" damping="0.0082" stiffness="11.2"/>
      <!-- z_slide: NOT a real physical DOF of the joystick rig (which only
           moves in x/y); added to absorb per-trial registration noise.
           joystick_base's position is fixed from ONE static calibration
           measurement transformed via an averaged (across many trials)
           rotation; each individual trial's own Procrustes rotation fit
           (computed from that trial's own noisy resting arm pose) differs
           slightly from that average, and since the joystick sits far from
           the arm anchor point used for the fit, that small rotational
           difference is amplified into a position error, observed as a
           consistent per-trial Z-only bias (~1-4mm, no consistent X/Y bias)
           between the mocap target and the model's fitted site, since
           there was no DOF to correct it. A small z_slide lets STAC absorb
           this the same way x_slide/y_slide already absorb the X/Y version
           of the same noise, rather than showing up as irreducible
           misalignment. -->
      <joint name="z_slide" type="slide" axis="0 0 1" limited="true" range="-0.006 0.006" damping="0.01" stiffness="15"/>

      <!-- small circular base (mount plate) -->
      <geom name="circular_base" type="cylinder" size="0.0015 0.00025" pos="0 0 0"
            rgba="0.6 0.6 0.6 1" contype="1" conaffinity="2"/>
      <!-- cosmetic mount post extending down toward the rig floor, below
           js_bottom (this body's own origin): purely visual, not a tracked
           keypoint, doesn't affect fitting. Added because the real rig's
           post visibly extends further down than js_bottom's own measured
           point; length is an approximate visual match, not a calibrated
           measurement. -->
      <geom name="mount_post" type="cylinder" size="0.001 0.01" pos="0 0 -0.01"
            rgba="0.5 0.5 0.5 1" contype="1" conaffinity="2"/>

      <!-- shaft + ball, tilted to the measured shaft direction -->
      <body name="joystick" pos="0 0 0" quat="0.94147265 -0.29173666 -0.16887560 0.00000000">
        <!-- shaft: radius 0.001 m (2x visual, per Eric, not re-measured,
             cosmetic only), half-length 0.0071815 m (total 0.014363 m =
             measured bottom->base, unchanged) -->
        <geom name="joystick_geom" type="cylinder" size="0.001 0.0071815" pos="0 0 0.0071815"
              rgba="0 0 1 1" contype="1" conaffinity="2"/>
        <!-- ball radius 0.002 m (2x visual, per Eric, not re-measured,
             cosmetic only), center 0.001822 m above the base/top-of-shaft
             point (measured), i.e. at 0.014363+0.001822, unchanged -->
        <geom name="joystick_ball" type="sphere" size="0.002" pos="0 0 0.016184"
              rgba="0 0 1 1" contype="1" conaffinity="2"/>
        <inertial mass="0.001" pos="0 0 0.0071815" diaginertia="1e-7 1e-7 1e-7"/>
      </body>
    </body>

'''


def main():
    xml = REFERENCE_XML.read_text()

    # ---- 0. swap <option> integrator/solver: MJX (the JAX-based MuJoCo STAC
    #         uses internally) does not implement "implicitfast" -- mjx.put_model
    #         raises NotImplementedError on it. STAC only ever calls
    #         mjx.kinematics (pure forward kinematics, confirmed elsewhere in
    #         this session's research), never actually integrates dynamics, so
    #         the integrator/solver choice is functionally irrelevant to STAC.
    #         Use the same RK4/CG combination v18/v19/v20 already use
    #         successfully in this exact pipeline. (Real forward-dynamics work,
    #         if resumed later alongside re-enabling the muscles/actuators
    #         below, should restore implicitfast+Newton at the Python level,
    #         same as build_v20_xml.py's joystick analysis scripts do.) ----
    old_option = '  <option integrator="implicitfast" timestep="0.001" solver="Newton" iterations="50" ls_iterations="50"/>'
    new_option = '  <option integrator="RK4" timestep="0.001" solver="CG" iterations="6" ls_iterations="6"/>'
    if old_option not in xml:
        raise RuntimeError("reference <option> line not found verbatim")
    xml = xml.replace(old_option, new_option)

    # ---- 0b. sh_tx/ty/tz range: kept at Eric's original +-10mm (matching
    #          v18) -- NOT tightened. Earlier this session, tightening this
    #          to +-0.5mm was used to work around STAC's pose optimizer
    #          (stac_mjx/stac_core.py's q_solver, a jaxopt ProjectedGradient
    #          with a FIXED stepsize=0.1) overshooting and saturating this
    #          DOF at its limit on every frame with the full +-10mm range
    #          (v18 never hit this with only 5 keypoints; v21's 19-keypoint
    #          set sums a much larger gradient through this shared proximal
    #          DOF). Root-caused and fixed properly instead: q_solver now
    #          uses stepsize=0.0 (jaxopt backtracking line search + FISTA
    #          acceleration, see stac_core.py's comment) rather than one
    #          fixed step for every DOF regardless of gradient magnitude --
    #          verified this lets sh_tx/ty/tz settle naturally within a few
    #          mm even with the full +-10mm range available, at the same or
    #          better marker fit error than the tightened-range workaround.
    #          The XML range itself never needed to change; only the solver
    #          did. Left here as a no-op block so the history is visible if
    #          this needs revisiting. ----

    # ---- 1. fix meshdir from Eric's local Mac path to our self-contained models/v21 ----
    old_compiler = '  <compiler angle="degree" meshdir="/Users/ericleonardis/Downloads/v19">\n    <lengthrange mode="muscle" timestep="0"/>\n  </compiler>'
    new_compiler = (
        '  <compiler angle="degree" meshdir="/mnt/vast/niyo/stac-mjx/models/v21">\n'
        '    <!-- v21 STAC variant: muscles commented out below, so no muscle length\n'
        '         range to compute, lengthrange directive dropped (see actuator/\n'
        '         tendon comment-out below for why). -->\n'
        '  </compiler>'
    )
    if old_compiler not in xml:
        raise RuntimeError("reference compiler block not found verbatim")
    xml = xml.replace(old_compiler, new_compiler)

    # ---- 2. graft the joystick body as a top-level worldbody child, before "ground" ----
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

    # ---- 3. comment out <tendon>...</tendon> (STAC never uses it: only needed if/when
    #         real forward-dynamics work resumes) ----
    # XML comments cannot be nested or contain a literal "--", and the tendon block
    # already has its own inner <!-- elbow_wrap --> comments plus dashes in prose
    # -- strip those before wrapping the whole block in one outer comment.
    def strip_inner_comments_and_dashes(text):
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = text.replace("--", ": ")
        return text

    tendon_match = re.search(r"  <tendon>.*?</tendon>\n", xml, flags=re.DOTALL)
    if not tendon_match:
        raise RuntimeError("tendon block not found")
    tendon_block = tendon_match.group(0)
    commented_tendon = (
        "  <!-- v21 STAC variant: tendons commented out (inert for STAC, which only\n"
        "       uses mj_forward/mjx.kinematics, never mj_step, so no muscle forces\n"
        "       ever apply). Uncomment together with the muscle actuators below to\n"
        "       restore real forward-dynamics muscle-driven simulation.\n"
        + strip_inner_comments_and_dashes(tendon_block)
        + "  -->\n"
    )
    xml = xml.replace(tendon_block, commented_tendon)

    # ---- 4. comment out every <muscle>/<motor> line inside <actuator> (keep the
    #         <actuator></actuator> wrapper itself, just empty it: nu=0) ----
    actuator_match = re.search(r"  <actuator>\n(.*?)  </actuator>\n", xml, flags=re.DOTALL)
    if not actuator_match:
        raise RuntimeError("actuator block not found")
    actuator_inner = actuator_match.group(1)
    n_muscle = len(re.findall(r"<muscle ", actuator_inner))
    n_motor = len(re.findall(r"<motor ", actuator_inner))
    commented_inner = (
        "    <!-- v21 STAC variant: all motor/muscle actuators commented out (per\n"
        "         Eric), inert for STAC (mj_forward/mjx.kinematics only, never\n"
        "         mj_step). Uncomment for future forward-dynamics work.\n"
        + strip_inner_comments_and_dashes(actuator_inner)
        + "    -->\n"
    )
    xml = xml[:actuator_match.start(1)] + commented_inner + xml[actuator_match.end(1):]
    print(f"Commented out: {n_muscle} muscles, {n_motor} motors, 1 tendon block.")

    # ---- Write canonical + self-contained mirrors ----
    V21_CANONICAL.write_text(xml)
    print(f"Wrote {V21_CANONICAL}")

    import os
    os.environ["MUJOCO_GL"] = "disable"
    import mujoco

    for out_dir in V21_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in V19_MESHDIR.iterdir():
            if f.is_file():
                dst = out_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
        out_path = out_dir / V21_NAME
        local_xml = xml.replace(
            'meshdir="/mnt/vast/niyo/stac-mjx/models/v21"', f'meshdir="{out_dir}"'
        )
        out_path.write_text(local_xml)

        try:
            ET.parse(str(out_path))
        except ET.ParseError as e:
            raise RuntimeError(f"{out_path} is not well-formed XML: {e}") from e

        m = mujoco.MjModel.from_xml_path(str(out_path))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        import numpy as np
        assert not np.isnan(d.xpos).any(), "NaN in xpos at qpos=0!"
        print(f"Wrote + validated: {out_path}")
        print(f"  nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} nmesh={m.nmesh} "
              f"nu={m.nu} ntendon={m.ntendon}")


if __name__ == "__main__":
    main()
