"""Environment for rodent modeling with dm_control and motion capture."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dm_control import composer
from dm_control.locomotion.arenas import floors
import stac.walkers as walkers
import stac.tasks as tasks
import stac.arenas as arenas
from typing import List, Dict
import numpy as np
import collections

MM_TO_METER = 1000
SkyBox = collections.namedtuple("SkyBox", ("file", "gridsize", "gridlayout"))


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
    arena = arenas.DannceArena(
        params,
        arena_diameter=arena_diameter,
        arena_center=arena_center,
        alpha=alpha,
    )
    task = tasks.ViewMocap(walker, arena, kp_data, params=params)
    time_limit = params["TIME_BINS"] * (params["n_frames"])
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
    # arena._mjcf_root.compiler.texturedir = (
    #     "/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/stac/stac"
    # )
    # sky_info = SkyBox(
    #     file="WhiteSkybox2048x2048.png",
    #     gridsize="11",
    #     gridlayout="..",
    # )
    # arena._skybox = arena._mjcf_root.asset.add(
    #     "texture",
    #     name="white_skybox",
    #     file=sky_info.file,
    #     type="cube",
    #     # gridsize=sky_info.gridsize,
    #     # gridlayout=sky_info.gridlayout,
    # )

    task = tasks.ViewVariability(variability, walker, arena, kp_data, params=params)
    time_limit = params["TIME_BINS"] * (params["n_frames"])
    return composer.Environment(
        task,
        time_limit=time_limit,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
