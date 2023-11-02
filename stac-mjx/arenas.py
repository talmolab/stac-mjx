"""Arena model for rat mocap."""
from dm_control import composer
import numpy as np
import h5py
import cv2
import scipy.ndimage as ndimage

_HEIGHTFIELD_ID = 0
_ARENA_DIAMETER: 0.5842
PEDESTAL_WIDTH = 0.099
PEDESTAL_HEIGHT = 0.054
_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1
_NUM_CYLINDER_SEGMENTS = 20
MM_TO_METER = 1000

FLOOR_PCTILE_CLAMP = 95
FLOOR_CLAMP_VALUE = 0.01
GROUND_GEOM_POS = "0 0 -0.005"
_GROUNDPLANE_QUAD_SIZE = 0.025


def _load_hfield(data_path, scale):
    """Load the floor_map from the snippet file."""
    with h5py.File(data_path, "r") as f:
        floormap = f["floormap"][:]
        floormap = floormap * scale / MM_TO_METER
    return floormap


class DannceArena(composer.Arena):
    def _build(
        self,
        params,
        arena_diameter=0.6985,
        arena_center=[0.1123, 0.1750],
        size=(2, 2),
        name="DannceArena",
        alpha=1.0,
    ):
        super(DannceArena, self)._build(name=name)

        self._size = size
        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

        self._ground_texture = self._mjcf_root.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=100,
            height=100,
            mark="edge",
            markrgb=[0.8, 0.8, 0.8],
        )
        if alpha == 0.0:
            self._ground_material = self._mjcf_root.asset.add(
                "material",
                name="groundplane",
                texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
                texuniform=True,
                reflectance=0.2,
                rgba=[0.0, 0.0, 0.0, alpha],
            )
        else:
            self._ground_material = self._mjcf_root.asset.add(
                "material",
                name="groundplane",
                texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
                texuniform=True,
                reflectance=0.2,
                texture=self._ground_texture,
            )

        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="groundplane",
            material=self._ground_material,
            pos=GROUND_GEOM_POS,
            size=list(size) + [_GROUNDPLANE_QUAD_SIZE],
        )

        # Get the dimensions of arena objects and floormap
        self.params = params
        self.arena_diameter = arena_diameter
        self.arena_center = arena_center

        # Make the cylinder
        if arena_diameter is not None:
            cylinder_segments = []
            radius = self.arena_diameter / 2
            height = 0.5
            chord = 2 * radius * np.tan(np.pi / _NUM_CYLINDER_SEGMENTS)
            for ii in range(_NUM_CYLINDER_SEGMENTS):
                ang = ii * 2 * np.pi / _NUM_CYLINDER_SEGMENTS
                cylinder_segments.append(
                    self._mjcf_root.worldbody.add(
                        "geom",
                        name="plane_{}".format(ii),
                        type="plane",
                        pos=[
                            radius * np.cos(ang) + arena_center[0],
                            radius * np.sin(ang) + arena_center[1],
                            height,
                        ],
                        size=[chord / 2, height, 0.1],
                        euler=[np.pi / 2, -np.pi / 2 + ang, 0],
                        rgba=[0.5, 0.5, 0.5, 0.2],
                    )
                )

        # Choose the FOV so that the floor always fits nicely within the frame
        # irrespective of actual floor size.
        fovy_radians = 2 * np.arctan2(
            _TOP_CAMERA_Y_PADDING_FACTOR * size[1], _TOP_CAMERA_DISTANCE
        )
        self._top_camera = self._mjcf_root.worldbody.add(
            "camera",
            name="top_camera",
            pos=[0, 0, _TOP_CAMERA_DISTANCE],
            zaxis=[0, 0, 1],
            fovy=np.rad2deg(fovy_radians),
        )

    @property
    def ground_geoms(self):
        """Return the ground geoms."""
        return (self._ground_geom,)

    def regenerate(self, random_state):
        """."""
        pass


class RatArena(composer.Arena):
    """A floor arena supporting heightfields."""

    def _build(
        self,
        hfield,
        params,
        process_objects=False,
        pedestal_center=None,
        pedestal_radius=None,
        pedestal_height=None,
        arena_diameter=None,
        arena_center=None,
        size=(8, 8),
        name="terrain",
    ):
        super(RatArena, self)._build(name=name)

        self._size = size
        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

        # Build heightfield.
        self._ground_asset = self._mjcf_root.asset.add(
            "hfield", name=name, nrow="101", ncol="101", size=".5 .5 1 0.1"
        )
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            name=name,
            type="hfield",
            rgba="0.2 0.3 0.4 0",
            pos=GROUND_GEOM_POS,
            hfield=name,
        )

        # Get the dimensions of arena objects and floormap
        self.params = params
        self.hfield = hfield
        if process_objects:
            self._process_objects()
        else:
            self.pedestal_center = pedestal_center
            self.pedestal_radius = pedestal_radius
            self.pedestal_height = pedestal_height
            self.arena_diameter = arena_diameter

        if self.pedestal_center is not None:
            self._pedestal = self._mjcf_root.worldbody.add(
                "geom",
                name="pedestal",
                type="cylinder",
                pos=self.pedestal_center,
                size=[self.pedestal_radius, self.pedestal_height / 2],
            )
        if arena_diameter is not None:
            cylinder_segments = []
            radius = self.arena_diameter / 2
            height = 0.5
            chord = 2 * radius * np.tan(np.pi / _NUM_CYLINDER_SEGMENTS)
            for ii in range(_NUM_CYLINDER_SEGMENTS):
                ang = ii * 2 * np.pi / _NUM_CYLINDER_SEGMENTS
                cylinder_segments.append(
                    self._mjcf_root.worldbody.add(
                        "geom",
                        name="plane_{}".format(ii),
                        type="plane",
                        pos=[
                            radius * np.cos(ang) + arena_center[0],
                            radius * np.sin(ang) + arena_center[1],
                            height + self._ground_geom.pos[2],
                        ],
                        size=[chord / 2, height, 1],
                        euler=[np.pi / 2, -np.pi / 2 + ang, 0],
                        rgba=[0.5, 0.5, 0.5, 0.2],
                    )
                )

        # Choose the FOV so that the floor always fits nicely within the frame
        # irrespective of actual floor size.
        fovy_radians = 2 * np.arctan2(
            _TOP_CAMERA_Y_PADDING_FACTOR * size[1], _TOP_CAMERA_DISTANCE
        )
        self._top_camera = self._mjcf_root.worldbody.add(
            "camera",
            name="top_camera",
            pos=[0, 0, _TOP_CAMERA_DISTANCE],
            zaxis=[0, 0, 1],
            fovy=np.rad2deg(fovy_radians),
        )

    def _process_objects(self):
        self.arena_diameter, self.arena_px_size = self._get_arena_params()
        self.hfield = self._preprocess_hfield(self.hfield)
        (
            self.pedestal_center,
            self.pedestal_radius,
            self.pedestal_height,
        ) = self._get_pedestal_params()

        # Find the bounds of the arena in the hfield.
        res = self._ground_asset.nrow
        ar_start = int(np.floor((res - self.arena_px_size) / 2))
        ar_end = ar_start + self.arena_px_size

        total_hfield = np.zeros((res, res))
        total_hfield[ar_start:ar_end, ar_start:ar_end] = self.hfield
        self.hfield = total_hfield

    def _smooth_hfield(self, image, sigma=1):
        image = ndimage.gaussian_filter(image, sigma, mode="nearest")
        return image

    def argmax2d(self, X):
        """Two-dimensional arg max."""
        n, m = X.shape
        x_ = np.ravel(X)
        k = np.argmax(x_)
        i, j = k // m, k % m
        return i, j

    def _get_arena_params(self):
        arena_diameter = self.params["_ARENA_DIAMETER"] * self.params["scale_factor"]
        res = self._ground_asset.nrow
        hfield_size = self._ground_asset.size[0]
        arena_px_size = int(np.floor(res * (arena_diameter / (hfield_size * 2))))
        return arena_diameter, arena_px_size

    def _preprocess_hfield(self, hfield, sigma=2.5):
        hfield = cv2.resize(
            hfield,
            (self.arena_px_size, self.arena_px_size),
            interpolation=cv2.INTER_LINEAR,
        )
        # Smooth the hfield
        hfield = self._smooth_hfield(hfield, sigma=sigma)

        # Rescale the z dim to fix xy scaling and account for global scaling
        resized_min = np.min(np.min(hfield))
        resized_max = np.max(np.max(hfield))
        hfield = (hfield - resized_min) / (resized_max - resized_min)

        # Clamp such that the 90th percentile of heights is 20 mm
        hfield -= hfield[0, 0]
        hfield /= np.percentile(hfield[:], FLOOR_PCTILE_CLAMP)
        hfield *= FLOOR_CLAMP_VALUE
        # hfield = ((hfield * (max_val - min_val)) + min_val) * scale
        return hfield

    def _get_pedestal_params(self):
        px_to_m = self.arena_diameter / self.arena_px_size
        arena_radius = self.arena_diameter / 2
        scale = self.params["scale_factor"]
        pedestal_radius = (PEDESTAL_WIDTH / 2) * scale
        pedestal_height = PEDESTAL_HEIGHT * scale

        cropped_hfield = self.hfield.copy()
        cropped_hfield[:, : int(cropped_hfield.shape[1] / 2)] = 0.0
        # import pdb; pdb.set_trace()
        pedestal_i, pedestal_j = self.argmax2d(cropped_hfield)
        pedestal_y = (pedestal_i * px_to_m) - arena_radius
        pedestal_x = (pedestal_j * px_to_m) - arena_radius
        pedestal_z = pedestal_height / 2 + self._ground_geom.pos[2]
        pedestal_center = [pedestal_x, pedestal_y, pedestal_z]
        return pedestal_center, pedestal_radius, pedestal_height

    @property
    def ground_geoms(self):
        """Return the ground geoms."""
        return (self._ground_geom,)

    def regenerate(self, random_state):
        """."""
        pass
