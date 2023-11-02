"""Environment for rodent modeling with dm_control and motion capture."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.walkers import base, legacy_base
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers.initializers import WalkerInitializer
import numpy as np

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the rat is considered standing.
_TORQUE_THRESHOLD = 60


class ZerosInitializer(WalkerInitializer):
    """An initializer that uses the zeros pose."""

    def initialize_pose(self, physics, walker, random_state):
        """Initialize the pose to all zeros."""
        qpos, xpos, xquat = walker.upright_pose
        physics.bind(walker.mjcf_model.find_all("joint")).qpos = 0.0 * qpos
        physics.bind(walker.mjcf_model.find_all("joint")).qvel = 0.0
        walker.set_velocity(physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))


class Rat(legacy_base.Walker):
    """A position-controlled rat with control range scaled to [-1, 1]."""

    def _build(self, params=None, name="walker", marker_rgba=None, initializer=None):
        self.params = params
        self._mjcf_root = mjcf.from_path(self._xml_path)
        if name:
            self._mjcf_root.model = name

        # Set corresponding marker color if specified.
        if marker_rgba is not None:
            for geom in self.marker_geoms:
                geom.set_attributes(rgba=marker_rgba)

        # Add keypoint sites to the mjcf model, and a reference to the sites as
        # an attribute for easier access
        self.body_sites = []
        for key, v in self.params["_KEYPOINT_MODEL_PAIRS"].items():
            parent = self._mjcf_root.find("body", v)
            pos = self.params["_KEYPOINT_INITIAL_OFFSETS"][key]
            site = parent.add(
                "site",
                name=key,
                type="sphere",
                size=[0.005],
                rgba="0 0 0 1",
                pos=pos,
                group=3,
            )
            self.body_sites.append(site)
        super(Rat, self)._build(initializer=initializer)

    # def reinitialize_pose(self, physics, random_state):
    #     for initializer in self._initializers:
    #         initializer.initialize_pose(physics, self, random_state)

    @property
    def upright_pose(self):
        """Reset pose to upright position."""
        return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

    @property
    def mjcf_model(self):
        """Return the model root."""
        return self._mjcf_root

    @composer.cached_property
    def actuators(self):
        """Return all actuators."""
        return tuple(self._mjcf_root.find_all("actuator"))

    @composer.cached_property
    def root_body(self):
        """Return the body."""
        return self._mjcf_root.find("body", "torso")

    @composer.cached_property
    def head(self):
        """Return the head."""
        return self._mjcf_root.find("body", "skull")

    @composer.cached_property
    def left_arm_root(self):
        """Return the left arm."""
        return self._mjcf_root.find("body", "scapula_L")

    @composer.cached_property
    def right_arm_root(self):
        """Return the right arm."""
        return self._mjcf_root.find("body", "scapula_R")

    @composer.cached_property
    def ground_contact_geoms(self):
        """Return ground contact geoms."""
        return tuple(
            self._mjcf_root.find("body", "foot_L").find_all("geom")
            + self._mjcf_root.find("body", "foot_R").find_all("geom")
        )

    @composer.cached_property
    def standing_height(self):
        """Return standing height."""
        return self.params["_STAND_HEIGHT"]

    @composer.cached_property
    def end_effectors(self):
        """Return end effectors."""
        return (
            self._mjcf_root.find("body", "lower_arm_R"),
            self._mjcf_root.find("body", "lower_arm_L"),
            self._mjcf_root.find("body", "foot_R"),
            self._mjcf_root.find("body", "foot_L"),
        )

    @composer.cached_property
    def observable_joints(self):
        """Return observable joints."""
        return tuple(
            actuator.joint for actuator in self.actuators if actuator.joint is not None
        )

    @composer.cached_property
    def bodies(self):
        """Return all bodies."""
        return tuple(self._mjcf_root.find_all("body"))

    @composer.cached_property
    def egocentric_camera(self):
        """Return the egocentric camera."""
        return self._mjcf_root.find("camera", "egocentric")

    @property
    def marker_geoms(self):
        """Return the lower arm geoms."""
        return (
            self._mjcf_root.find("geom", "lower_arm_R"),
            self._mjcf_root.find("geom", "lower_arm_L"),
        )

    @property
    def _xml_path(self):
        """Return the path to th model .xml file."""
        return self.params["_XML_PATH"]

    def _build_observables(self):
        return RodentObservables(self)


class RodentObservables(legacy_base.WalkerObservables):
    """Observables for the Rat."""

    @composer.observable
    def head_height(self):
        """Observe the head height."""
        return observable.Generic(
            lambda physics: physics.bind(self._entity.head).xpos[2]
        )

    @composer.observable
    def sensors_torque(self):
        """Observe the torque sensors."""
        return observable.MJCFFeature(
            "sensordata",
            self._entity.mjcf_model.sensor.torque,
            corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_THRESHOLD),
        )

    @composer.observable
    def actuator_activation(self):
        """Observe the actuator activation."""
        model = self._entity.mjcf_model
        return observable.MJCFFeature("act", model.find_all("actuator"))

    @composer.observable
    def appendages_pos(self):
        """Equivalent to `end_effectors_pos` with head's position appended."""

        def relative_pos_in_egocentric_frame(physics):
            end_effectors_with_head = self._entity.end_effectors + (self._entity.head,)
            end_effector = physics.bind(end_effectors_with_head).xpos
            torso = physics.bind(self._entity.root_body).xpos
            xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(end_effector - torso, xmat), -1)

        return observable.Generic(relative_pos_in_egocentric_frame)

    @property
    def proprioception(self):
        """Return proprioceptive information."""
        return [
            self.joints_pos,
            self.joints_vel,
            self.actuator_activation,
            self.body_height,
            self.end_effectors_pos,
            self.appendages_pos,
            self.world_zaxis,
        ] + self._collect_from_attachments("proprioception")
