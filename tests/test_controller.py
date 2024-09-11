from pathlib import Path

from stac_mjx import main
from stac_mjx import utils
from stac_mjx.controller import STAC, _align_joint_dims
from mujoco import _structs

_BASE_PATH = Path.cwd()


def test_init_stac(mocap_nwb, config):
    cfg = main.load_configs(config)
    xml_path = _BASE_PATH / cfg.model.MJCF_PATH
    kp_data, sorted_kp_names = utils.load_data(cfg)

    stac = STAC(xml_path, cfg, sorted_kp_names)

    assert stac.cfg == cfg
    assert stac._kp_names == sorted_kp_names
    assert isinstance(stac._mj_model, _structs.MjModel)


def test_align_joint_dims():
    from jax import numpy as jp
    import mujoco

    joint_types = [
        mujoco.mjtJoint.mjJNT_FREE,
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_BALL,
        mujoco.mjtJoint.mjJNT_SLIDE,
    ]
    ranges = [[0.0, 0.0], [-0.1, 0.1], [0.0, 1.0], [-0.5, 0.5]]
    names = ["root", "hingejoint", "balljoint", "slidejoint"]
    lb, ub, part_names = _align_joint_dims(joint_types, ranges, names)
    print(lb)

    true_lb = jp.array(
        [
            -jp.inf,
            -jp.inf,
            -jp.inf,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.5,
        ]
    )

    true_ub = jp.array(
        [
            jp.inf,
            jp.inf,
            jp.inf,
            1.0,
            1.0,
            1.0,
            1.0,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
        ]
    )
    assert jp.array_equal(lb, true_lb)
    assert jp.array_equal(ub, true_ub)
    assert part_names == [
        "root",
        "root",
        "root",
        "root",
        "root",
        "root",
        "root",
        "hingejoint",
        "balljoint",
        "balljoint",
        "balljoint",
        "balljoint",
        "slidejoint",
    ]
