from stac_mjx.stac import _align_joint_dims


def test_align_joint_dims():
    import mujoco
    from jax import numpy as jp

    joint_types = [
        mujoco.mjtJoint.mjJNT_FREE,
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_BALL,
        mujoco.mjtJoint.mjJNT_SLIDE,
    ]
    ranges = [[0.0, 0.0], [-0.1, 0.1], [0.0, 1.0], [-0.5, 0.5]]
    names = ["root", "hingejoint", "balljoint", "slidejoint"]
    lb, ub, part_names = _align_joint_dims(
        joint_types, ranges, names, [False, True, True, True]
    )
    print(lb)

    true_lb = jp.array(
        [
            -jp.inf,
            -jp.inf,
            -jp.inf,
            -jp.inf,
            -jp.inf,
            -jp.inf,
            -jp.inf,
            -0.1,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -0.5,
        ]
    )

    true_ub = jp.array(
        [
            jp.inf,
            jp.inf,
            jp.inf,
            jp.inf,
            jp.inf,
            jp.inf,
            jp.inf,
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
