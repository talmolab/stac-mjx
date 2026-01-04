"""Rescale utils."""

from mujoco import MjSpec


def dm_scale_spec(spec: MjSpec, scale: float) -> MjSpec:
    """Scale a spec by a scalar.

    Args:
        spec (MjSpec): The spec to scale.
        scale (float): The scalar multiplier.

    Returns:
        MjSpec: The scaled spec.
    """
    scaled_spec = spec.copy()

    # Traverse the kinematic tree, scaling all geoms
    def scale_bodies(parent, scale=1.0):
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

    # if scale_actuators:
    # # scale gear
    for mesh in scaled_spec.meshes:
        mesh.scale = mesh.scale * scale

    for actuator in scaled_spec.actuators:
        # scale the actuator gear by (scale ** 2),
        # this is because muscle force-generating capacity
        # scales with the cross-sectional area of the muscle
        actuator.gear = actuator.gear * scale * scale

    # scale the z-position for all keypoints
    for keypoint in scaled_spec.keys:
        qpos = keypoint.qpos
        qpos[2] = qpos[2] * scale
        keypoint.qpos = qpos
        keypoint.qpos[2] = keypoint.qpos[2] * scale

    scale_bodies(scaled_spec.worldbody.first_body(), scale)
    return scaled_spec
