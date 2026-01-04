import types

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stac_mjx.stac import Stac


def test_package_data_unbatched():
    dummy = types.SimpleNamespace(
        _offsets=np.zeros((2, 3)),
        _part_names=["root"],
        _body_names=["body"],
        _kp_names=["kp1", "kp2"],
        _body_site_idxs=np.array([0, 1]),
    )

    qposes = np.zeros((2, 1))
    xposes = np.zeros((2, 1, 3))
    xquats = np.zeros((2, 1, 4))
    marker_sites = np.zeros((2, 2, 3))
    kp_data = np.zeros((2, 6))

    out = Stac._package_data(
        dummy,
        None,
        qposes,
        xposes,
        xquats,
        marker_sites,
        kp_data,
        batched=False,
    )

    assert out.offsets.shape == (2, 3)
    assert out.kp_names == ["kp1", "kp2"]
