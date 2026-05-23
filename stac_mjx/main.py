"""User-level API to run stac."""

from jax import Array
import numpy as np
import time
from omegaconf import DictConfig
from stac_mjx import io, utils
from stac_mjx.config import compose_config
from stac_mjx.stac import Stac
from pathlib import Path

from jaxtyping import Float


def load_stac_config(
    config_dir: Path | str,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load and validate a STAC config from a Hydra config directory.

    Args:
        config_dir: Absolute path to config directory.
        config_name: Name of the Hydra config to load.
        overrides: Optional Hydra override list.

    Returns:
        Validated STAC configuration.
    """
    cfg = compose_config(config_dir, config_name=config_name, overrides=overrides)
    print("Config loaded and validated.")
    return cfg


def run_stac(
    cfg: DictConfig,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    kp_names: list[str],
    base_path: Path | None = None,
) -> tuple[str, str | None]:
    """Run the full skeletal registration pipeline.

    Runs calibration (unless skipped), then IK using the production jaxls
    q optimization path (unless skipped), optionally infers velocities,
    and saves results to HDF5.

    Args:
        cfg: STAC configuration.
        kp_data: Flattened mocap keypoint data.
        kp_names: Ordered keypoint names matching kp_data columns.
        base_path: Base path for resolving relative file paths. Defaults to cwd.

    Returns:
        Tuple of (calibration output path, IK output path or None).

    Raises:
        ValueError: If kp_data columns don't match kp_names * 3.
    """
    if base_path is None:
        base_path = Path.cwd()

    expected_cols = len(kp_names) * 3
    if kp_data.shape[1] != expected_cols:
        raise ValueError(
            f"kp_data has {kp_data.shape[1]} columns but expected {expected_cols} "
            f"({len(kp_names)} keypoints × 3). "
            f"Ensure kp_data is shaped (n_frames, n_keypoints * 3) and that "
            f"kp_names length matches the number of keypoints in kp_data."
        )

    start_time = time.time()

    calibration_path = base_path / cfg.stac.calibration_path
    ik_path = base_path / cfg.stac.ik_path
    xml_path = base_path / cfg.model.MJCF_PATH
    stac = Stac(xml_path, cfg, kp_names)

    if not cfg.stac.skip_calibration:
        kps = kp_data[: cfg.stac.n_calibration_frames]
        print(f"Running calibration. Mocap data shape: {kps.shape}")
        calibration_data = stac.calibrate(kps)
        print(f"saving data to {calibration_path}", flush=True)
        io.save_data_to_h5(
            config=cfg, file_path=calibration_path, **calibration_data.as_dict()
        )
    else:
        print(
            "Skipping calibration. To change this behavior, set cfg.stac.skip_calibration to False."
        )

    if cfg.stac.skip_ik:
        print(
            "Skipping IK phase. To change this behavior, set cfg.stac.skip_ik to False."
        )
        return calibration_path, None

    print("Running IK")
    _, calibration_data = io.load_stac_data(calibration_path)

    offsets = calibration_data.offsets

    print(f"kp_data shape: {kp_data.shape}")
    ik_data = stac.run_ik(kp_data, offsets)

    print(f"Final qpos shape: {ik_data.qpos.shape}")
    if cfg.stac.infer_qvels:
        t_vel = time.time()
        qvels = utils.compute_velocity_from_kinematics(
            qpos_trajectory=ik_data.qpos,
            dt=stac._mj_model.opt.timestep,
            freejoint=stac._freejoint,
        )
        ik_data.qvel = np.array(qvels)
        print(f"Finished compute velocity in {time.time() - t_vel} seconds")

    print(
        f"Saving data to {ik_path}. Finished in {(time.time() - start_time)/60:.2f} minutes"
    )
    io.save_data_to_h5(config=cfg, file_path=ik_path, **ik_data.as_dict())
    return calibration_path, ik_path
