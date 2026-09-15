"""Build the v17 model XML by patching the v16 base model with:
  1. Scapula tilt (small upward rotation to clear elbow trajectory) — anatomy fix
  2. forearm_supination hinge joint on the wrist body (axis = +Y wrist-local)
  3. KnuckleC + KnuckleL sites on the wrist body at Eric's saved offsets
  4. Hand mesh geoms reparented from ulna body onto wrist body so they rotate with supination

Reads:  v17_knuckle_offsets_LATEST.json  (Eric's saved offsets)
Output: models/mouse_forelimb_right_janelia_arm_hand_v17.xml
"""
from __future__ import annotations
import json
import re
from pathlib import Path

V16_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_moving_shoulder_v2.xml")
V17_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v17.xml")
OFFSETS_JSON = Path("/mnt/vast/niyo/stac-mjx/v17_knuckle_offsets_LATEST.json")

SCAPULA_TILT_DEG = 8.0   # tilt scapula upward; quat axis = X (rotation about lateral axis)


def quat_mul(q1, q2):
    """Hamilton product of quats (w, x, y, z)."""
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def axis_angle_quat(axis, angle_rad):
    import math
    s = math.sin(angle_rad / 2)
    return (math.cos(angle_rad / 2), axis[0]*s, axis[1]*s, axis[2]*s)


def main():
    with open(OFFSETS_JSON) as f:
        off = json.load(f)
    knC_xml = off["KnuckleC"]["site_pos_xml_attr"]   # e.g. 'pos="-0.001800 0.003200 -0.000900"'
    knL_xml = off["KnuckleL"]["site_pos_xml_attr"]
    print(f"Using saved offsets:")
    print(f"  KnuckleC: {knC_xml}  (from {off['KnuckleC']['source_landmark']})")
    print(f"  KnuckleL: {knL_xml}  (from {off['KnuckleL']['source_landmark']})")

    xml = V16_XML.read_text()

    # ─── 1. Scapula tilt ──────────────────────────────────────────────────────
    # v16 scapula quat="0.639001 0.064200 0.435113 -0.631057"
    # Compose with an additional rotation about local X (medial-lateral axis) by SCAPULA_TILT_DEG.
    # This is a small upward tilt to clear elbow trajectory.
    import math
    tilt = axis_angle_quat((1.0, 0.0, 0.0), math.radians(SCAPULA_TILT_DEG))
    # Original scapula quat
    orig_q = (0.639001, 0.064200, 0.435113, -0.631057)
    new_q = quat_mul(orig_q, tilt)
    new_q_str = f'{new_q[0]:.6f} {new_q[1]:.6f} {new_q[2]:.6f} {new_q[3]:.6f}'
    old_scap = '<body name="scapula" pos="-0.008751 0.004141 0.006510" quat="0.639001 0.064200 0.435113 -0.631057">'
    new_scap = f'<body name="scapula" pos="-0.008751 0.004141 0.006510" quat="{new_q_str}">'
    if old_scap not in xml:
        raise RuntimeError("Scapula body line not found in v16 XML")
    xml = xml.replace(old_scap, new_scap)
    print(f"\nScapula tilt: +{SCAPULA_TILT_DEG}° about local X axis (medial-lateral)")
    print(f"  old quat: {orig_q}")
    print(f"  new quat: {new_q}")

    # ─── 2. Reparent hand mesh geoms from ulna to wrist ──────────────────────
    retopo_pat = re.compile(
        r'^[ \t]*<geom type="mesh" mesh="(Retopo_[^"]+)" class="hand_visual"/>\s*\n',
        re.MULTILINE,
    )
    matches = retopo_pat.findall(xml)
    print(f"\nReparenting {len(matches)} Retopo hand-mesh geoms from ulna → wrist body")
    xml = retopo_pat.sub("", xml)
    new_geom_block = "\n".join(
        f'              <geom type="mesh" mesh="{m}" class="hand_visual" pos="0 -0.01092 0"/>'
        for m in matches
    )

    # ─── 3 + 4. Patch wrist body: add supination joint + reparented hand ────
    # NOTE: do not hardcode Wrist/KnuckleC/KnuckleL sites here — stac_mjx adds them
    # automatically based on KEYPOINT_MODEL_PAIRS + KEYPOINT_INITIAL_OFFSETS in the config.
    # The saved knuckle offsets (Eric's dash app values) live in the YAML config.
    pat = re.compile(
        r'<body name="wrist" pos="0 0\.01092 0">\s*<inertial[^/]+/>\s*</body>',
        re.MULTILINE | re.DOTALL,
    )
    new_wrist = (
        '<body name="wrist" pos="0 0.01092 0">\n'
        '              <joint name="forearm_supination" type="hinge" pos="0 0 0" '
        'axis="0 1 0" range="-90 90" damping="0.01" limited="true"/>\n'
        '              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>\n'
        + new_geom_block + "\n"
        + '            </body>'
    )
    if not pat.search(xml):
        raise RuntimeError("Wrist body block not found in v16 XML")
    xml = pat.sub(new_wrist, xml)

    # ─── 5. Update top-of-file comment / model name ──────────────────────────
    xml = xml.replace(
        '<mujoco model="mouse_forelimb_right_hand">',
        '<mujoco model="mouse_forelimb_right_hand_v17">',
    )

    # ─── 6. Add forearm_supination position actuator ──────────────────────────
    actuator_block = (
        '    <!-- v17: supination actuator -->\n'
        '    <position name="forearm_supination_act" joint="forearm_supination" kp="0.5" '
        'ctrlrange="-90 90"/>\n'
        '  </actuator>'
    )
    xml = xml.replace('  </actuator>', actuator_block)

    V17_XML.write_text(xml)
    print(f"\nWrote v17 XML: {V17_XML}")
    print(f"  size: {V17_XML.stat().st_size // 1024} KB")

    # Validate by loading in MuJoCo
    print("\nValidating with MuJoCo...")
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(V17_XML))
    print(f"  Loaded OK: nbody={m.nbody}, njnt={m.njnt}, ngeom={m.ngeom}")
    sup_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "forearm_supination")
    if sup_jid < 0:
        raise RuntimeError("forearm_supination joint missing!")
    print(f"  forearm_supination joint id={sup_jid}, qpos_idx={m.jnt_qposadr[sup_jid]}")
    # Note: KP sites Wrist/KnuckleC/KnuckleL are added by stac_mjx at load time, not in XML.

    # Confirm hand meshes are now on wrist body
    ulna_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ulna")
    wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    n_retopo_on_ulna = 0; n_retopo_on_wrist = 0
    for gi in range(m.ngeom):
        mesh_id = m.geom_dataid[gi]
        if mesh_id < 0: continue
        mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if not mname or "Retopo" not in mname: continue
        if m.geom_bodyid[gi] == ulna_id: n_retopo_on_ulna += 1
        if m.geom_bodyid[gi] == wrist_id: n_retopo_on_wrist += 1
    print(f"  Retopo hand meshes: {n_retopo_on_wrist} on wrist body, {n_retopo_on_ulna} on ulna body")
    print(f"\nv17 XML ready.")


if __name__ == "__main__":
    main()
