"""Rescale utilities for MuJoCo specs."""

from mujoco import MjSpec


def dm_scale_spec(spec: MjSpec, scale: float) -> MjSpec:
    """Scale a MuJoCo spec uniformly by a scalar.

    Scales body positions, geom sizes/positions, mesh scales, actuator
    gears (by scale^2 for muscle cross-section), and keypoint z-positions.

    Args:
        spec: The MjSpec to scale.
        scale: Uniform scale factor.

    Returns:
        Scaled copy of the spec.
    """
    scaled_spec = spec.copy()

    def scale_bodies(parent: MjSpec, scale: float = 1.0) -> None:
        body = parent.first_body()
        while body:
            if body.pos is not None:
                body.pos = body.pos * scale
            for geom in body.geoms:
                geom.fromto = geom.fromto * scale
                geom.size = geom.size * scale
                if geom.pos is not None:
                    geom.pos = geom.pos * scale
            scale_bodies(body, scale)
            body = parent.next_body(body)

    for mesh in scaled_spec.meshes:
        mesh.scale = mesh.scale * scale

    for actuator in scaled_spec.actuators:
        actuator.gear = actuator.gear * scale * scale

    for keypoint in scaled_spec.keys:
        qpos = keypoint.qpos
        qpos[2] = qpos[2] * scale
        keypoint.qpos = qpos

    scale_bodies(scaled_spec.worldbody.first_body(), scale)
    return scaled_spec
