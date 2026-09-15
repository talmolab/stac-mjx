"""Build v18 XML on top of v17: add wrist_flexion hinge to the wrist body.

v18 = v17 + wrist_flexion (axis = X = medial-lateral, perpendicular to supination)
  - joint at wrist body origin (same as supination, same as Wrist marker)
  - range -60 to +60 deg, damping 0.01

Output: models/mouse_forelimb_right_janelia_arm_hand_v18.xml
"""
from __future__ import annotations
import re
from pathlib import Path

V17_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v17.xml")
V18_XML = Path("/mnt/vast/niyo/stac-mjx/models/mouse_forelimb_right_janelia_arm_hand_v18.xml")


def main():
    xml = V17_XML.read_text()

    # Insert wrist_flexion joint right after forearm_supination on the wrist body.
    # Both joints share the wrist body origin; MuJoCo composes them.
    old = (
        '              <joint name="forearm_supination" type="hinge" pos="0 0 0" '
        'axis="0 1 0" range="-90 90" damping="0.01" limited="true"/>\n'
    )
    new = (
        old
        + '              <joint name="wrist_flexion" type="hinge" pos="0 0 0" '
        'axis="1 0 0" range="-60 60" damping="0.01" limited="true"/>\n'
    )
    if old not in xml:
        raise RuntimeError("forearm_supination joint not found in v17 XML")
    xml = xml.replace(old, new)

    # Add wrist_flexion actuator
    old_act = (
        '    <position name="forearm_supination_act" joint="forearm_supination" kp="0.5" '
        'ctrlrange="-90 90"/>\n'
        '  </actuator>'
    )
    new_act = (
        '    <position name="forearm_supination_act" joint="forearm_supination" kp="0.5" ctrlrange="-90 90"/>\n'
        '    <position name="wrist_flexion_act" joint="wrist_flexion" kp="0.5" ctrlrange="-60 60"/>\n'
        '  </actuator>'
    )
    if old_act not in xml:
        raise RuntimeError("forearm_supination actuator not found")
    xml = xml.replace(old_act, new_act)

    # Update model name
    xml = xml.replace(
        '<mujoco model="mouse_forelimb_right_hand_v17">',
        '<mujoco model="mouse_forelimb_right_hand_v18">',
    )

    V18_XML.write_text(xml)
    print(f"Wrote v18 XML: {V18_XML}")
    print(f"  size: {V18_XML.stat().st_size // 1024} KB")

    # Validate
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(V18_XML))
    print(f"  Loaded OK: nbody={m.nbody}, njnt={m.njnt}, ngeom={m.ngeom}")
    sup_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "forearm_supination")
    flex_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist_flexion")
    print(f"  forearm_supination: jid={sup_jid}, qpos_idx={m.jnt_qposadr[sup_jid]}, axis={m.jnt_axis[sup_jid]}")
    print(f"  wrist_flexion:      jid={flex_jid}, qpos_idx={m.jnt_qposadr[flex_jid]}, axis={m.jnt_axis[flex_jid]}")
    print(f"\nv18 XML ready (njoints went from {m.njnt-1} → {m.njnt}).")


if __name__ == "__main__":
    main()
