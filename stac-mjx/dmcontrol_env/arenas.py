"""Arena model for rat mocap."""
from dm_control import composer
import numpy as np
import h5py
import cv2
import scipy.ndimage as ndimage
import collections
import os

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
SKYBOX_PATH = "../assets/WhiteSkybox.png"
SkyBox = collections.namedtuple("SkyBox", ("file", "gridsize", "gridlayout"))
ASSETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)


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
        self._mjcf_root.compiler.texturedir = ASSETS_PATH
        sky_info = SkyBox(file=SKYBOX_PATH, gridsize="3 4", gridlayout=".U..LFRB.D..")
        self._skybox = self._mjcf_root.asset.add(
            "texture",
            name="wht_skybox",
            file=sky_info.file,
            type="skybox",
            gridsize=sky_info.gridsize,
            gridlayout=sky_info.gridlayout,
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
