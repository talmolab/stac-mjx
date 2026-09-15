"""Build v19 XML on top of v18: replace the flat, unjointed 104-piece Retopo hand
mesh with Shravan's real articulated hand (carpals -> metacarpals -> phalanx chains),
grafted onto the existing `wrist` body, plus a new wrist-deviation joint.

v19 = v18's proximal arm (unchanged: shoulder/elbow/forearm_supination/wrist_flexion,
      same STL meshes, same muscle sites) + Shravan's hand bones (OBJ meshes, from
      models/shravan_forearm/Forearm_hand.xml) + new digit joints:
  - rz_N_L_C_right   : wrist radial/ulnar deviation (new, on `wrist` body)
  - rx_Phalanx_hand_1_x_right, rz_Phalanx_hand_1_x_right : MCP flex + spread (x=1..5)
      (no ry -- axial twist is fully rigid, not modeled, per Eric)
  - rx_Phalanx_hand_2_x_right : PIP flex (x=1..5)
  - Phalanx_hand_3_x_right (DIP, x=2..5) is frozen: no joint at all, just marks where
      the dx_t keypoint sits -- per Eric, not meant to articulate.
  - Sesamoid_elbow_right_bone : rigid addition on the `ulna` body (no old-model
      counterpart, but Eric asked to include it anyway)

See scratch/plans/PLAN_v19_hand_model.md for the full design rationale, and the
scratchpad scripts that produced the alignment + subtree (fk_common.py, icp.py,
align_anchors.py, compute_graft_transforms.py, build_hand_subtree.py) for how the
geometry was derived.

Output: models/mouse_forelimb_right_janelia_arm_hand_v19.xml
"""
from __future__ import annotations
from pathlib import Path

V18_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v18.xml")
V19_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v19.xml")

# Self-contained mirrors: same XML, meshdir rewritten to each copy's own local folder,
# so both are fully standalone (no cross-repo path dependency).
V19_MIRRORS = [
    Path("/mnt/vast/niyo/stac-mjx/models/v19/mouse_forelimb_right_janelia_arm_hand_v19.xml"),
    Path("/mnt/vast/niyo/janelia_model/v19/mouse_forelimb_right_janelia_arm_hand_v19.xml"),
]
ARM_STLS = ["clavicle_right.stl", "scapula_right.stl", "humerus_right.stl",
            "radius_right.stl", "ulna_right.stl"]

SCRATCH = Path("/tmp/claude-0/-root-vast-eric-stac-mjx/f2fcc897-b1bb-4a87-90a3-6d8da622db8c/scratchpad")
HAND_ASSETS = (SCRATCH / "hand_assets.xml").read_text().rstrip("\n")
HAND_SUBTREE = (SCRATCH / "hand_subtree.xml").read_text().rstrip("\n")
ELBOW_SESAMOID_GEOM = (SCRATCH / "elbow_sesamoid_geom.xml").read_text().rstrip("\n")

NEW_JOINTS = [
    # (name, axis, range, spec-comment)
    ("rz_N_L_C_right", "0 0 1", "-15 15"),
] + [
    j
    for x in range(1, 6)
    for j in (
        [(f"rx_Phalanx_hand_1_{x}_right", "1 0 0", "-20 90"),
         (f"rz_Phalanx_hand_1_{x}_right", "0 0 1", "-20 20"),
         (f"rx_Phalanx_hand_2_{x}_right", "1 0 0", "0 100")]
        # rx_Phalanx_hand_3_x_right (DIP) is frozen -- no joint, no actuator.
    )
]


def main():
    xml = V18_XML.read_text()

    # ---- 1. header / model name ----
    xml = xml.replace(
        '<mujoco model="mouse_forelimb_right_hand_v18">',
        '<mujoco model="mouse_forelimb_right_hand_v19">',
    )
    old_header_comment = (
        "    Mouse right forelimb musculoskeletal model with hand/paw meshes.\n"
        "    12 muscles: 9 from reference model + 3 rotator cuff muscles.\n"
        "    104 hand/paw mesh pieces attached to ulna (frozen, no joints).\n"
        "    All contacts disabled. Hand meshes decimated to max 20K faces each.\n"
        "\n"
        "    Excluded meshes:\n"
        "      Retopo_3_005.stl  - degenerate (near-zero volume)\n"
        "      Retopo_3_058.stl  - degenerate (zero surface area)\n"
        "      Retopo_38_002.stl - removed by user (geom 33)\n"
        "      Retopo_55_003.stl - removed by user (geom 100)\n"
        "\n"
        "    Joint summary (4 DOF arm):\n"
        "      sh_elv:       shoulder elevation,  pitch about Y, [-70, 0] deg\n"
        "      sh_extension: shoulder extension,  yaw about Z,   [-70, 40] deg\n"
        "      sh_rotation:  shoulder rotation,   roll about X,  [-25, 89] deg\n"
        "      elbow:        elbow flex/extend,   about X,       [-70, 80] deg\n"
    )
    new_header_comment = (
        "    Mouse right forelimb musculoskeletal model with an articulated hand.\n"
        "    Proximal arm (shoulder/elbow/forearm/wrist) is unchanged from v18:\n"
        "    same STL meshes, same muscle sites, same joints/ranges.\n"
        "\n"
        "    v19: the old 104-piece flat, unjointed Retopo_*.stl hand mesh is replaced\n"
        "    with Shravan's articulated hand (carpals -> metacarpals -> phalanx\n"
        "    chains, OBJ meshes, models/v19/), matched to the old hand via\n"
        "    models/shravan_forearm/old_to_new_mesh_mapping.csv. See\n"
        "    scratch/plans/PLAN_v19_hand_model.md for the full design rationale.\n"
        "\n"
        "    Joint summary (4 DOF arm, unchanged from v18):\n"
        "      sh_elv:       shoulder elevation,  pitch about Y, [-70, 0] deg\n"
        "      sh_extension: shoulder extension,  yaw about Z,   [-70, 40] deg\n"
        "      sh_rotation:  shoulder rotation,   roll about X,  [-25, 89] deg\n"
        "      elbow:        elbow flex/extend,   about X,       [-70, 80] deg\n"
        "\n"
        "    Wrist/hand joints (new in v19 except forearm_supination/wrist_flexion):\n"
        "      forearm_supination: forearm twist,      about Y, [-90, 90] deg (v17)\n"
        "      wrist_flexion:      wrist flex/extend,   about X, [-60, 60] deg (v18)\n"
        "      rz_N_L_C_right:     wrist radial/ulnar deviation, about Z, [-15, 15] deg\n"
        "      rx_Phalanx_hand_1_x_right: MCP flex,    about X, [-20, 90] deg (x=1..5)\n"
        "      rz_Phalanx_hand_1_x_right: MCP spread,  about Z, [-20, 20] deg (x=1..5)\n"
        "        (no MCP twist axis: fully rigid, per Eric)\n"
        "      rx_Phalanx_hand_2_x_right: PIP flex,    about X, [0, 100] deg (x=1..5)\n"
        "      Phalanx_hand_3_x_right (DIP, x=2..5) is frozen: no joint, just marks\n"
        "        where the dx_t keypoint sits (digit 1/thumb has no 3rd segment).\n"
        "      Metacarpals, carpals, and all sesamoids are rigid (no joints), matching\n"
        "      Shravan's own model.\n"
    )
    if old_header_comment not in xml:
        raise RuntimeError("v18 header comment not found verbatim")
    xml = xml.replace(old_header_comment, new_header_comment)

    # ---- 2. compiler meshdir ----
    old_compiler = '  <compiler angle="degree" meshdir="/mnt/vast/niyo/janelia_model/skeleton_stls_object_coordinates_hand"/>'
    new_compiler = '  <compiler angle="degree" meshdir="/mnt/vast/niyo/stac-mjx/models/v19"/>'
    if old_compiler not in xml:
        raise RuntimeError("compiler meshdir line not found")
    xml = xml.replace(old_compiler, new_compiler)

    # ---- 3. asset: strip old Retopo mesh block, splice in the 41 new hand meshes ----
    start_marker = '    <!-- Hand/paw meshes (103 pieces, decimated, in radius_ulna_right local coords) -->\n'
    end_marker_line = '    <mesh name="Retopo_67_mesh" file="Retopo_67.stl"/>\n'
    start_idx = xml.index(start_marker)
    end_idx = xml.index(end_marker_line) + len(end_marker_line)
    old_asset_block = xml[start_idx:end_idx]
    new_asset_block = (
        "    <!-- Hand meshes: Shravan's articulated hand (carpals/metacarpals/phalanges/\n"
        "         sesamoids), OBJ format, matched to the old Retopo hand via\n"
        "         models/shravan_forearm/old_to_new_mesh_mapping.csv. -->\n"
        + HAND_ASSETS + "\n"
    )
    xml = xml[:start_idx] + new_asset_block + xml[end_idx:]

    # ---- 4. wrist body: add rz_N_L_C_right joint, replace flat Retopo geoms with the
    #         new articulated hand subtree ----
    old_wrist_open = (
        '            <body name="wrist" pos="0 0.01092 0">\n'
        '              <joint name="forearm_supination" type="hinge" pos="0 0 0" '
        'axis="0 1 0" range="-90 90" damping="0.01" limited="true"/>\n'
        '              <joint name="wrist_flexion" type="hinge" pos="0 0 0" '
        'axis="1 0 0" range="-60 60" damping="0.01" limited="true"/>\n'
        '              <inertial pos="0 0 0" mass="1e-6" '
        'diaginertia="1e-12 1e-12 1e-12"/>\n'
    )
    new_wrist_open = (
        '            <!-- "wrist" is anatomically Shravan\'s N_L_C_right (carpal '
        'complex); kept as "wrist" for STAC config/script compatibility. -->\n'
        '            <body name="wrist" pos="0 0.01092 0">\n'
        '              <joint name="forearm_supination" type="hinge" pos="0 0 0" '
        'axis="0 1 0" range="-90 90" damping="0.01" limited="true"/>\n'
        '              <joint name="wrist_flexion" type="hinge" pos="0 0 0" '
        'axis="1 0 0" range="-60 60" damping="0.01" limited="true"/>\n'
        '              <joint name="rz_N_L_C_right" type="hinge" pos="0 0 0" '
        'axis="0 0 1" range="-15 15" damping="0.01" limited="true"/>\n'
        '              <inertial pos="0 0 0" mass="1e-6" '
        'diaginertia="1e-12 1e-12 1e-12"/>\n'
    )
    if old_wrist_open not in xml:
        raise RuntimeError("wrist body opening block not found verbatim")
    xml = xml.replace(old_wrist_open, new_wrist_open)

    first_geom = '              <geom type="mesh" mesh="Retopo_102_001_mesh" class="hand_visual" pos="0 -0.01092 0"/>\n'
    last_geom = '              <geom type="mesh" mesh="Retopo_67_mesh" class="hand_visual" pos="0 -0.01092 0"/>\n'
    start_idx = xml.index(first_geom)
    end_idx = xml.index(last_geom) + len(last_geom)
    indented_subtree = "\n".join("              " + line if line.strip() else line
                                  for line in HAND_SUBTREE.splitlines()) + "\n"
    xml = xml[:start_idx] + indented_subtree + xml[end_idx:]

    old_trailer = (
        "            </body>\n"
        "\n"
        "            <!-- Hand/paw meshes (frozen, no joints, no collision) -->\n"
        "          </body>\n"
    )
    new_trailer = (
        "            </body>\n"
        "          </body>\n"
    )
    if old_trailer not in xml:
        raise RuntimeError("wrist-body trailer / stale comment not found verbatim")
    xml = xml.replace(old_trailer, new_trailer)

    # ---- 5. ulna body: add the Sesamoid_elbow rigid geom ----
    old_ulna_geoms = (
        '            <geom type="mesh" mesh="ulna_right_mesh"/>\n'
        '            <geom type="mesh" mesh="radius_right_mesh"/>\n'
    )
    new_ulna_geoms = old_ulna_geoms + ELBOW_SESAMOID_GEOM + "\n"
    if old_ulna_geoms not in xml:
        raise RuntimeError("ulna body geoms not found verbatim")
    xml = xml.replace(old_ulna_geoms, new_ulna_geoms)

    # ---- 6. actuators for all new joints ----
    old_act_tail = (
        '    <position name="forearm_supination_act" joint="forearm_supination" kp="0.5" ctrlrange="-90 90"/>\n'
        '    <position name="wrist_flexion_act" joint="wrist_flexion" kp="0.5" ctrlrange="-60 60"/>\n'
        "  </actuator>"
    )
    new_actuator_lines = "\n".join(
        f'    <position name="{name}_act" joint="{name}" kp="0.5" ctrlrange="{rng}"/>'
        for name, axis, rng in NEW_JOINTS
    )
    new_act_tail = (
        '    <position name="forearm_supination_act" joint="forearm_supination" kp="0.5" ctrlrange="-90 90"/>\n'
        '    <position name="wrist_flexion_act" joint="wrist_flexion" kp="0.5" ctrlrange="-60 60"/>\n'
        + new_actuator_lines + "\n"
        "  </actuator>"
    )
    if old_act_tail not in xml:
        raise RuntimeError("actuator tail not found verbatim")
    xml = xml.replace(old_act_tail, new_act_tail)

    V19_XML.write_text(xml)
    print(f"Wrote v19 XML: {V19_XML}")
    print(f"  size: {V19_XML.stat().st_size // 1024} KB")

    # ---- Validate ----
    # Strict well-formedness check first: MuJoCo's own XML parser is lenient about
    # things like "--" inside a comment (illegal per the XML spec), which silently
    # breaks stricter parsers/editors/viewers downstream. Catch that class of bug here.
    import xml.etree.ElementTree as ET
    try:
        ET.parse(str(V19_XML))
    except ET.ParseError as e:
        raise RuntimeError(f"v19 XML is not well-formed XML: {e}") from e
    print("  Strict XML well-formedness check passed.")

    import os
    os.environ["MUJOCO_GL"] = "disable"
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(V19_XML))
    print(f"  Loaded OK: nbody={m.nbody}, njnt={m.njnt}, ngeom={m.ngeom}, nmesh={m.nmesh}")

    import numpy as np
    for name, axis, rng in NEW_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint {name} missing from compiled model!")
    print(f"  All {len(NEW_JOINTS)} new joints present.")

    # ---- Self-contained mirrors: same XML, meshdir pointed at each mirror's own
    #      folder, plus the 5 arm STLs so each mirror is fully standalone ----
    import shutil
    stac_mjx_meshdir = "/mnt/vast/niyo/stac-mjx/models/v19"
    for mirror_path in V19_MIRRORS:
        mirror_dir = mirror_path.parent
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for stl in ARM_STLS:
            src = Path(stac_mjx_meshdir) / stl
            dst = mirror_dir / stl
            if not dst.exists():
                shutil.copy2(src, dst)
        mirror_xml = xml.replace(
            f'meshdir="{stac_mjx_meshdir}"',
            f'meshdir="{mirror_dir}"',
        )
        mirror_path.write_text(mirror_xml)
        m_check = mujoco.MjModel.from_xml_path(str(mirror_path))
        print(f"  Mirror OK: {mirror_path} (nbody={m_check.nbody}, njnt={m_check.njnt})")


if __name__ == "__main__":
    main()
