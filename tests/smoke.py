"""Model compatibility and real-data pipeline checks, run explicitly by CI."""

from pathlib import Path

import mujoco
import numpy as np

import stac_mjx
from stac_mjx.io import load_stac_data
from stac_mjx.utils import mjx_load


def test_mouse_load():
    repo_root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(
        str(repo_root / "models/mouse/mouse_with_meshes.xml")
    )
    mjx_load(model)


def test_rodent_pipeline(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg = stac_mjx.load_stac_config(repo_root / "configs")
    cfg.stac.calibration_path = str(tmp_path / "calibration.h5")
    cfg.stac.ik_path = str(tmp_path / "ik.h5")
    cfg.stac.n_calibration_frames = 1
    cfg.stac.n_frames_per_clip = 2
    cfg.stac.q_opt.context_frames = 0
    cfg.stac.q_opt.coarse_init_stride = 0
    # Exercise each solve with a small iteration budget; this is not a fit-quality test.
    cfg.stac.q_opt.calibration_max_iterations = 1
    cfg.stac.q_opt.ik_max_iterations = 1
    cfg.model.N_ITERS = 1
    cfg.model.N_SAMPLE_FRAMES = 1

    keypoints, names = stac_mjx.load_keypoint_data(cfg, repo_root)
    assert keypoints.shape == (1000, 3 * len(names))
    calibration_path, ik_path = stac_mjx.run_stac(
        cfg, keypoints[:2], names, base_path=repo_root
    )
    _, calibration = load_stac_data(calibration_path)
    _, fitted = load_stac_data(ik_path)
    model = mujoco.MjModel.from_xml_path(str(repo_root / cfg.model.MJCF_PATH))
    assert calibration.qpos.shape == (1, model.nq)
    assert fitted.qpos.shape == (2, model.nq)
    assert fitted.qvel.shape == (2, model.nv)
    assert fitted.kp_names == calibration.kp_names == names
    assert np.isfinite(calibration.offsets).all()
    assert np.isfinite(fitted.qpos).all() and np.isfinite(fitted.qvel).all()
