#!/usr/bin/env python
"""Build a STAC-safe copy of the v24 full-skeleton forearm/hand model.

v24 (/mnt/vast/niyo/janelia_model/v24/forearm_v24.xml) is a Blender-exported
musculoskeletal model (postprocess_xml.py) -- unlike v17-v22's simplified
arm-only rigs, it carries the full spine/skull/mandible/clavicle/scapula
chain (49 bodies) plus 52 muscle tendons / 127 actuators (position+velocity+
torque+muscle, per postprocess_xml.py's header). Same rationale as
build_v22_stac_xml.py: STAC only ever calls mjx.kinematics (no mj_step /
dynamics), so muscles/tendons/actuators are functionally inert for
fitting -- but leaving them in costs compile time and is one more way for
something to go wrong that has nothing to do with kinematics. This script
comments out (not deletes) the <tendon> and <actuator> blocks, leaving every
<joint> declaration (name/range/damping) untouched, since STAC's optimizer
bounds come from jnt_range.

Body names differ from v21/v22 (humerus -> humerus_right, ulna -> ulna_right,
wrist -> N_L_C_right -- the carpal complex), and there is no joystick body at
all in this model. See configs/model/mouse_arm_hand_v24.yaml for the updated
KEYPOINT_MODEL_PAIRS built against these names.

The source XML has no explicit meshdir (MuJoCo defaults to the XML's own
directory, /mnt/vast/niyo/janelia_model/v24, where all the referenced .obj
files live) -- since this script writes its copy into stac-mjx/models/, an
explicit meshdir is added so the copy still resolves every mesh.

Also bakes explicit <inertial> tags into EVERY body, using mass/ipos/iquat/
inertia computed once by /root/miniforge3/bin/python3 (system MuJoCo 3.10.0,
cached in models/v24_stac_body_inertial.json). This project's actual .venv
(mujoco 3.2.2, the one with jax/mjx installed) hard-errors compiling the
untouched SRC file -- confirmed by loading it directly with .venv's mujoco --
in (at least) two distinct ways: (1) "body mass is too small, cannot compute
center of mass" for spine/neck bodies (T1-T13, C1-C6, "Armature") whose only
geoms are density="0" visual bone/cartilage meshes with no mass-bearing
collision proxy, so computed mass is exactly zero and 3.2.2 errors before its
own boundmass floor applies (3.10.0 floors it silently); (2) "inertia must
have positive eigenvalues" for at least one hand body with a valid-looking
but numerically-degenerate inertia tensor from a tiny sliver collision
ellipsoid (e.g. Phalanx_hand_3_5_right), where 3.2.2's own eigendecomposition
of the near-singular matrix comes out slightly non-positive-definite even
though 3.10.0 computes and floors (boundinertia="1e-14") a valid one. Rather
than chase every individual body a stricter compiler might reject next,
this bakes in the already-validated-positive values for all 48 bodies
uniformly, which sidesteps the strict compiler's own inertia computation
entirely. This is value-neutral for kinematics-only fitting -- it only
changes physical mass properties MuJoCo needs to compile the model, not any
body position/orientation.

Usage: python3 build_v24_stac_xml.py
"""
import json
import re
from pathlib import Path

SRC = Path("/mnt/vast/niyo/janelia_model/v24/forearm_v24.xml")
DST = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v24_stac.xml")
MESH_DIR = "/mnt/vast/niyo/janelia_model/v24"
BODY_INERTIAL_JSON = Path("/mnt/vast/niyo/stac-mjx/models/v24_stac_body_inertial.json")


def inject_inertials(xml: str) -> str:
    body_inertial = json.loads(BODY_INERTIAL_JSON.read_text())
    for name, v in body_inertial.items():
        pattern = re.compile(rf'(<body name="{re.escape(name)}"[^>]*>)')
        matches = pattern.findall(xml)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one <body name=\"{name}\"> opening tag, found {len(matches)}")
        tag = matches[0]
        inertial = (
            f'\n          <inertial pos="{v["ipos"][0]} {v["ipos"][1]} {v["ipos"][2]}" '
            f'quat="{v["iquat"][0]} {v["iquat"][1]} {v["iquat"][2]} {v["iquat"][3]}" '
            f'mass="{v["mass"]}" diaginertia="{v["inertia"][0]} {v["inertia"][1]} {v["inertia"][2]}"/>'
        )
        xml = xml.replace(tag, tag + inertial, 1)
    return xml


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

    xml = xml.replace('<mujoco>', '<mujoco model="mouse_forelimb_right_janelia_arm_hand_v24_stac">', 1)

    # Relocate <compiler> to be the first child of <mujoco>, before <asset>.
    # Native MuJoCo's own compiler is order-independent (two-pass parse), but
    # dm_control's mjcf.from_path (what stac_mjx/stac.py actually uses to
    # load the model) parses elements sequentially and applies meshdir at
    # the point it's encountered -- in SRC, <compiler> sits ~1000 lines in,
    # AFTER every <mesh> asset declaration (an artifact of postprocess_xml.py
    # appending it late), so every mesh got resolved with NO meshdir at all
    # under dm_control (confirmed: FileNotFoundError on stac-mjx/models/<obj>
    # instead of janelia_model/v24/<obj> before this fix).
    old_compiler = '<compiler boundmass="1e-8" boundinertia="1e-14" angle="degree"/>\n'
    if xml.count(old_compiler) != 1:
        raise ValueError(f"expected exactly one <compiler> tag to relocate, found {xml.count(old_compiler)}")
    xml = xml.replace(old_compiler, "", 1)
    new_compiler = f'  <compiler boundmass="1e-8" boundinertia="1e-14" angle="degree" meshdir="{MESH_DIR}"/>\n'
    xml = xml.replace(
        '<mujoco model="mouse_forelimb_right_janelia_arm_hand_v24_stac">\n',
        '<mujoco model="mouse_forelimb_right_janelia_arm_hand_v24_stac">\n' + new_compiler,
        1,
    )

    xml = inject_inertials(xml)

    # mjx.put_model only supports Euler/RK4 (NotImplementedError otherwise);
    # STAC never calls mj_step/integrates dynamics, so the integrator choice
    # is inert for kinematics-only fitting -- same rationale as commenting
    # out tendon/actuator above.
    old_option = '<option integrator="implicitfast" timestep="0.001"/>'
    new_option = '<option integrator="Euler" timestep="0.001"/>'
    if xml.count(old_option) != 1:
        raise ValueError(f"expected exactly one <option> tag to patch, found {xml.count(old_option)}")
    xml = xml.replace(old_option, new_option)

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

    from mujoco import mjx
    mjx.put_model(m)
    print("mjx.put_model OK (no unsupported collision pairs).")

    kp_names_to_check = [
        "humerus_right", "ulna_right", "radius_right", "N_L_C_right",
        "Metacarpal_hand_1_right", "Phalanx_hand_1_1_right", "Phalanx_hand_2_1_right",
        "Phalanx_hand_1_2_right", "Phalanx_hand_2_2_right", "Phalanx_hand_3_2_right",
        "Phalanx_hand_1_5_right", "Phalanx_hand_2_5_right", "Phalanx_hand_3_5_right",
    ]
    for name in kp_names_to_check:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"body {name} not found in compiled model"
    print(f"All {len(kp_names_to_check)} arm/hand keypoint target bodies present.")

    # stac_mjx/stac.py actually loads models via dm_control's mjcf.from_path,
    # not mujoco.MjModel.from_xml_path directly -- its asset-path resolution
    # is a distinct code path (see this script's meshdir-relocation comment
    # above), so it needs its own check.
    from dm_control import mjcf
    root = mjcf.from_path(str(DST))
    mjcf.Physics.from_mjcf_model(root)
    print("dm_control mjcf.from_path + Physics.from_mjcf_model OK.")


if __name__ == "__main__":
    main()
