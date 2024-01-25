"""Task for rat mocap."""
from dm_control import composer
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.mujoco.wrapper.mjbindings import enums
import numpy as np
from matplotlib import cm, colors
from scipy.interpolate import interp1d

MM_TO_METER = 1000
PEDESTAL_WIDTH = 0.099
PEDESTAL_HEIGHT = 0.054

ACTUATOR_BODY_PAIRS = {
    "lumbar_extend": "vertebra_4",
    "lumbar_bend": "vertebra_5",
    "lumbar_twist": "vertebra_3",
    "cervical_extend": "vertebra_atlant",
    "cervical_bend": "vertebra_cervical_1",
    "cervical_twist": "vertebra_cervical_3",
    "caudal_extend": "vertebra_C1",
    "caudal_bend": "vertebra_C2",
    "hip_L_supinate": "upper_leg_L",
    "hip_L_abduct": "upper_leg_L",
    "hip_L_extend": "upper_leg_L",
    "knee_L": "lower_leg_L",
    "ankle_L": "foot_L",
    "toe_L": "toe_L",
    "hip_R_supinate": "upper_leg_R",
    "hip_R_abduct": "upper_leg_R",
    "hip_R_extend": "upper_leg_R",
    "knee_R": "lower_leg_R",
    "ankle_R": "foot_R",
    "toe_R": "toe_R",
    "atlas": "skull",
    "mandible": "jaw",
    "scapula_L_supinate": "scapula_L",
    "scapula_L_abduct": "scapula_L",
    "scapula_L_extend": "scapula_L",
    "shoulder_L": "upper_arm_L",
    "shoulder_sup_L": "upper_arm_L",
    "elbow_L": "lower_arm_L",
    "wrist_L": "hand_L",
    "finger_L": "finger_L",
    "scapula_R_supinate": "scapula_R",
    "scapula_R_abduct": "scapula_R",
    "scapula_R_extend": "scapula_R",
    "shoulder_R": "upper_arm_R",
    "shoulder_sup_R": "upper_arm_R",
    "elbow_R": "lower_arm_R",
    "wrist_R": "hand_R",
    "finger_R": "finger_R",
}


class ViewMocap(composer.Task):
    """A ViewMocap task."""

    def __init__(
        self,
        walker,
        arena,
        kp_data,
        walker_spawn_position=(0, 0, 0),
        walker_spawn_rotation=None,
        physics_timestep=0.001,
        control_timestep=0.025,
        qpos=None,
        params=None,
    ):
        """Initialize ViewMocap environment.

        :param walker: Rodent walker
        :param arena: Arena defining floor
        :param kp_data: Keypoint data (t x (n_marker*ndims))
        :param walker_spawn_position: Initial spawn position.
        :param walker_spawn_rotation: Initial spawn rotation.
        :param physics_timestep: Timestep for physics simulation
        :param control_timestep: Timestep for controller
        :param qpos: Precomputed list of qposes.
        """
        self._arena = arena
        self._walker = walker

        self._walker.create_root_joints(self._arena.attach(self._walker))
        self._walker_spawn_position = walker_spawn_position
        self._walker_spawn_rotation = walker_spawn_rotation
        self.kp_data = kp_data
        self.sites = []
        self.qpos = qpos
        self.V = None
        self.params = params

        for id, name in enumerate(self.params["KEYPOINT_MODEL_PAIRS"]):
            start = (np.random.rand(3) - 0.5) * 0.001
            rgba = self.params["KEYPOINT_COLOR_PAIRS"][name]
            site = self._arena.mjcf_model.worldbody.add(
                "site",
                name=name,
                type="sphere",
                size=[0.005],
                rgba=rgba,
                pos=start,
                group=2,
            )
            self.sites.append(site)
        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        enabled_observables.append(self._walker.observables.egocentric_camera)
        for obs in enabled_observables:
            obs.enabled = True

            self.set_timesteps(
                physics_timestep=physics_timestep, control_timestep=control_timestep
            )

    @property
    def root_entity(self):
        """Return arena root."""
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Initialize an arena episode."""
        # self._arena.regenerate(random_state)
        self._arena.mjcf_model.visual.map.znear = 0.0002
        # self._arena.mjcf_model.visual.map.zfar = 4.

    def initialize_episode(self, physics, random_state):
        """Reinitialize the pose of the walker."""
        self._walker.reinitialize_pose(physics, random_state)

    def get_reward(self, physics):
        """Get reward."""
        return 0.0

    def get_discount(self, physics):
        """Get discount."""
        return 1.0

    def after_step(self, physics, random_state):
        """Update the mujoco markers on each step."""
        # Get the frame
        self.frame = physics.time()
        self.frame = np.floor(self.frame / self.params["TIME_BINS"]).astype("int32")
        # Set the mocap marker positions
        physics.bind(self.sites).pos[:] = np.reshape(
            self.kp_data[self.frame, :].T, (-1, 3)
        )

        # Set qpos if it has been precomputed.
        if self.qpos is not None:
            physics.named.data.qpos[:] = self.qpos[self.frame]
            physics.named.data.qpos["walker/mandible"] = self.params["MANDIBLE_POS"]
            physics.named.data.qvel[:] = 0.0
            physics.named.data.qacc[:] = 0.0
            # Forward kinematics for rendering
            mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)


class ViewVariability(ViewMocap):
    def __init__(self, variability, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rgba = [1, 1, 1, 1]

        self.actuator_names = [actuator.name for actuator in self._walker.actuators]
        self.actuator_sites = []
        for name in self.actuator_names:
            start = (np.random.rand(3) - 0.5) * 0.001
            site = self._arena.mjcf_model.worldbody.add(
                "site",
                name=name,
                type="sphere",
                size=[0.01],
                rgba=rgba,
                pos=start,
                group=5,
            )
            self.actuator_sites.append(site)
        self.variability = variability

        # Set up the colors.
        max_val = np.percentile(variability.flatten(), 95)
        min_val = np.percentile(variability.flatten(), 5)
        max_val = np.max([np.abs(min_val), np.abs(max_val)])
        min_val = -max_val
        self.variability = np.clip(self.variability, min_val, max_val)
        norm = colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        self.size_map = interp1d([min_val, max_val], [0.002, 0.01])

    def after_step(self, physics, random_state):
        """Update the mujoco markers on each step."""
        # Get the frame
        self.frame = physics.time()
        self.frame = np.floor(self.frame / self.params["TIME_BINS"]).astype("int32")

        # Set qpose if it has been precomputed.
        if self.qpos is not None:
            physics.named.data.qpos[:] = self.qpos[self.frame]
            physics.named.data.qpos["walker/mandible"] = self.params["MANDIBLE_POS"]
            physics.named.data.qvel[:] = 0.0
            physics.named.data.qacc[:] = 0.0
            # Forward kinematics for rendering
            mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

        site_names = ["walker/" + part for part in list(ACTUATOR_BODY_PAIRS.values())]
        physics.bind(self.actuator_sites).pos[:] = physics.named.data.xpos[site_names]

        # Set the variability marker colors
        physics.bind(self.actuator_sites).rgba[:] = self.mapper.to_rgba(
            self.variability[self.frame, :]
        )
        # Set the variability marker colors
        physics.bind(self.actuator_sites).size[:] = self.size_map(
            np.repeat(
                self.variability[self.frame, :].squeeze()[:, np.newaxis], 3, axis=1
            )
        )
        physics.bind(self.actuator_sites).rgba[:, 3] = 0.5
        mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)
