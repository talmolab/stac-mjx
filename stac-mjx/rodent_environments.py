"""Environment for rodent modeling with dm_control and motion capture.

Attributes:
    MM_TO_METER (int): Description
    PEDESTAL_HEIGHT (float): Description
    PEDESTAL_WIDTH (float): Description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dm_control import composer
from dm_control.locomotion.arenas import floors
import stac.walkers as walkers
import stac.tasks as tasks
import stac.arenas as arenas
from typing import List, Dict, Text, Union, Tuple
import numpy as np

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

MM_TO_METER = 1000
# Height of head above which the rat is considered standing.
_TORQUE_THRESHOLD = 60
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 0.4  # Spatial scale of terrain bumps (in meters).
_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1
PEDESTAL_WIDTH = 0.099
PEDESTAL_HEIGHT = 0.054


def rodent_mocap(
    kp_data,
    params: Dict,
    random_state: int = None,
    hfield_image: np.ndarray = None,
    pedestal_center: List = None,
    pedestal_height: float = None,
    pedestal_radius: float = None,
    arena_diameter: float = None,
    arena_center: List = None,
    alpha=1.0,
):
    """View a rat with mocap sites.

    Args:
        kp_data (TYPE): Reference keypoint data
        params (Dict): Stac parameters dict
        random_state (int, optional): Random seed for arena initialization.
        hfield_image (np.ndarray, optional): Heightfield array for non-flat surfaces.
        pedestal_center (List, optional): Center of pedestal
        pedestal_height (float, optional): Height of pedestal
        pedestal_radius (float, optional): Radius of pedestal
        arena_diameter (float, optional): Diameter of circular arena
        arena_center (List, optional): Center of circular arena

    Deleted Parameters:
        arena_type (Text, optional): Description
    """
    # Build a position-controlled Rat
    walker = walkers.Rat(
        initializer=None,
        params=params,
        observable_options={"egocentric_camera": dict(enabled=True)},
    )
    if params["ARENA_TYPE"] == "HField":
        process_objects = False
        if hfield_image is None:
            hfield_image = arenas._load_hfield(
                params["data_path"], params["scale_factor"]
            )
            process_objects = True
        arena = arenas.RatArena(
            hfield_image,
            params,
            process_objects,
            pedestal_center=pedestal_center,
            pedestal_height=pedestal_height,
            pedestal_radius=pedestal_radius,
            arena_diameter=arena_diameter,
            arena_center=arena_center,
        )
        task = tasks.ViewMocap_Hfield(walker, arena, kp_data, params=params)
    elif params["ARENA_TYPE"] == "DannceArena":
        arena = arenas.DannceArena(
            params,
            arena_diameter=arena_diameter,
            arena_center=arena_center,
            alpha=alpha,
        )
        task = tasks.ViewMocap(walker, arena, kp_data, params=params)
    elif params["ARENA_TYPE"] == "Standard":
        # Build a Floor arena
        arena = floors.Floor(size=(1, 1))
        arena._ground_geom.pos = [0.0, 0.0, -0.01]
        # Build a mocap viewing task
        task = tasks.ViewMocap(walker, arena, kp_data, params=params)

    # time_limit = params["_TIME_BINS"] * (params["n_frames"] - 1)
    time_limit = params["_TIME_BINS"] * (params["n_frames"])
    return composer.Environment(
        task,
        time_limit=time_limit,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def rodent_variability(
    kp_data: np.ndarray,
    variability: np.ndarray,
    params: Dict,
    random_state: int = None,
    alpha=1.0,
) -> composer.Environment:
    """Environment to view model motor variability

    Args:
        kp_data (np.ndarray): Keypoint data
        variability (np.ndarray): Variability data
        params (Dict): Parameters dictionary
        random_state (int, optional): Environment random state. Defaults to None.
        alpha (float, optional): DannceArena alpha value. Defaults to 1.0.

    Returns:
        composer.Environment: Environment
    """
    walker = walkers.Rat(
        initializer=None,
        params=params,
        observable_options={"egocentric_camera": dict(enabled=True)},
    )
    arena = arenas.DannceArena(
        params, alpha=alpha, arena_diameter=None, arena_center=None
    )
    task = tasks.ViewVariability(variability, walker, arena, kp_data, params=params)
    time_limit = params["_TIME_BINS"] * (params["n_frames"])
    return composer.Environment(
        task,
        time_limit=time_limit,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )