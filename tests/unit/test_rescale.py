from pathlib import Path

import numpy as np
import pytest

from stac_mjx import rescale


def _iter_body_pairs(orig_parent, scaled_parent):
    orig_body = orig_parent.first_body()
    scaled_body = scaled_parent.first_body()
    while orig_body:
        yield orig_body, scaled_body
        yield from _iter_body_pairs(orig_body, scaled_body)
        orig_body = orig_parent.next_body(orig_body)
        scaled_body = scaled_parent.next_body(scaled_body)


def _collect_scale_targets(orig_spec, scaled_spec):
    targets = []
    for orig_body, scaled_body in _iter_body_pairs(
        orig_spec.worldbody, scaled_spec.worldbody
    ):
        if orig_body.pos is not None:
            targets.append(
                ("body.pos", np.array(orig_body.pos), np.array(scaled_body.pos))
            )
        for idx, geom in enumerate(orig_body.geoms):
            scaled_geom = scaled_body.geoms[idx]
            if geom.fromto is not None:
                targets.append(
                    (
                        "geom.fromto",
                        np.array(geom.fromto),
                        np.array(scaled_geom.fromto),
                    )
                )
            if geom.size is not None:
                targets.append(
                    ("geom.size", np.array(geom.size), np.array(scaled_geom.size))
                )
            if geom.pos is not None:
                targets.append(
                    ("geom.pos", np.array(geom.pos), np.array(scaled_geom.pos))
                )
    return targets


def test_dm_scale_spec_scales_geometry():
    mujoco = pytest.importorskip("mujoco")
    model_path = Path.cwd() / "models" / "mouse" / "mouse_with_meshes.xml"
    spec = mujoco.MjSpec.from_file(str(model_path))

    scale = 2.0
    scaled = rescale.dm_scale_spec(spec, scale)

    targets = _collect_scale_targets(spec, scaled)
    non_zero = [t for t in targets if np.any(np.abs(t[1]) > 0)]
    assert non_zero, "No non-zero geometry or body positions found to validate scaling."

    for label, orig, scaled_val in non_zero:
        assert np.allclose(scaled_val, orig * scale), f"{label} was not scaled"
