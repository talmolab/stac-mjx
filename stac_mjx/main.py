"""User-level API to run stac."""

import jax
from jax import Array
from jax import numpy as jp
import numpy as np
import time
from omegaconf import DictConfig
from stac_mjx import io, utils
from stac_mjx.config import compose_config
from stac_mjx.stac import Stac
from pathlib import Path
from functools import partial

from jaxtyping import Float
from jaxtyping import jaxtyped
from beartype import beartype


def load_configs(
    config_dir: Path | str, config_name: str = "config"
) -> DictConfig:
    """Load and validate configs from a Hydra config directory.

    Args:
        config_dir: Absolute path to config directory.
        config_name: Name of the Hydra config to load.

    Returns:
        Validated STAC configuration.
    """
    cfg = compose_config(config_dir, config_name=config_name)
    print("Config loaded and validated.")
    return cfg


@jaxtyped(typechecker=beartype)
def run_stac(
    cfg: DictConfig,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    kp_names: list[str],
    base_path: Path | None = None,
) -> tuple[str, str | None]:
    """Run the full skeletal registration pipeline.

    Runs fit_offsets (unless skipped), then ik_only (unless skipped),
    optionally infers velocities, and saves results to HDF5.

    Args:
        cfg: STAC configuration.
        kp_data: Flattened mocap keypoint data.
        kp_names: Ordered keypoint names matching kp_data columns.
        base_path: Base path for resolving relative file paths. Defaults to cwd.

    Returns:
        Tuple of (fit_offsets output path, ik_only output path or None).

    Raises:
        ValueError: If kp_data columns don't match kp_names * 3.
        ValueError: If n_frames_per_clip doesn't evenly divide total frames.
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

    utils.enable_xla_flags()

    start_time = time.time()

    fit_offsets_path = base_path / cfg.stac.fit_offsets_path
    ik_only_path = base_path / cfg.stac.ik_only_path

    xml_path = base_path / cfg.model.MJCF_PATH

    stac = Stac(xml_path, cfg, kp_names)

    compute_velocity_fn = partial(
        utils.compute_velocity_from_kinematics,
        dt=stac._mj_model.opt.timestep,
        freejoint=stac._freejoint,
    )
    vmap_compute_velocity_fn = jax.vmap(compute_velocity_fn)

    if not cfg.stac.skip_fit_offsets:
        kps = kp_data[: cfg.stac.n_fit_frames]
        print(f"Running fit. Mocap data shape: {kps.shape}")
        fit_offsets_data = stac.fit_offsets(kps)
        print(f"saving data to {fit_offsets_path}", flush=True)
        io.save_data_to_h5(
            config=cfg, file_path=fit_offsets_path, **fit_offsets_data.as_dict()
        )
    else:
        print(
            "Skipping fit_offsets. To change this behavior, set cfg.stac.skip_fit_offsets to False."
        )

    if cfg.stac.skip_ik_only:
        print(
            "Skipping IK-only phase. To change this behavior, set cfg.stac.skip_ik_only to False."
        )
        return fit_offsets_path, None
    elif kp_data.shape[0] % cfg.stac.n_frames_per_clip != 0:
        raise ValueError(
            f"n_frames_per_clip ({cfg.stac.n_frames_per_clip}) must divide evenly with the total number of mocap frames({kp_data.shape[0]})"
        )

    print("Running ik_only()")
    cfg, fit_offsets_data = io.load_stac_data(fit_offsets_path)

    offsets = fit_offsets_data.offsets

    print(f"kp_data shape: {kp_data.shape}")
    ik_only_data = stac.ik_only(kp_data, offsets)

    if cfg.stac.continuous:
        print("Handling edge effects...")
        ik_only_data = utils.handle_edge_effects(
            ik_only_data, cfg.stac.n_frames_per_clip
        )

    batched_qpos = ik_only_data.qpos.reshape(
        (-1, cfg.stac.n_frames_per_clip, ik_only_data.qpos.shape[-1])
    )

    print(f"Final qpos shape: {ik_only_data.qpos.shape}")
    if cfg.stac.infer_qvels:
        t_vel = time.time()
        qvels = vmap_compute_velocity_fn(qpos_trajectory=batched_qpos)
        ik_only_data.qvel = np.array(qvels).reshape(-1, *qvels.shape[2:])
        print(f"Finished compute velocity in {time.time() - t_vel} seconds")

    print(
        f"Saving data to {ik_only_path}. Finished in {(time.time() - start_time)/60:.2f} minutes"
    )
    io.save_data_to_h5(config=cfg, file_path=ik_only_path, **ik_only_data.as_dict())
    return fit_offsets_path, ik_only_path
