import numpy as np

from stac_mjx import io


def test_stac_data_as_dict_preserves_output_fields():
    data = io.StacData(
        qpos=np.zeros((2, 1)),
        xpos=np.zeros((2, 1, 3)),
        xquat=np.zeros((2, 1, 4)),
        marker_sites=np.zeros((2, 2, 3)),
        offsets=np.zeros((2, 3)),
        kp_data=np.zeros((2, 6)),
        names_qpos=["root"],
        names_xpos=["body"],
        kp_names=["kp1", "kp2"],
    )

    out = data.as_dict()

    assert out["qpos"].shape == (2, 1)
    assert out["xpos"].shape == (2, 1, 3)
    assert out["xquat"].shape == (2, 1, 4)
    assert out["marker_sites"].shape == (2, 2, 3)
    assert out["offsets"].shape == (2, 3)
    assert out["kp_names"] == ["kp1", "kp2"]
