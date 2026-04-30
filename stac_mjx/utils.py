"""Utility functions for STAC."""

import os

import jax
from jax import Array
from jax import numpy as jp
from jaxtyping import Float, Int, Bool
from jaxtyping import jaxtyped
from beartype import beartype
from mujoco import mjx
from mujoco.mjx._src import smooth
import numpy as np
from stac_mjx import io
from scipy import ndimage
from jax.extend.backend import get_backend

CONTINUOUS_BATCH_OVERLAP = 10


def enable_xla_flags() -> None:
    """Enable XLA flags for faster runtime on Nvidia GPUs."""
    if get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

    # Enable persistent compilation cache to avoid recompilation across runs
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "stac-mjx", "jax")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


def mjx_load(mj_model) -> tuple[mjx.Model, mjx.Data]:
    """Load a MuJoCo model into MJX.

    Args:
        mj_model: MuJoCo model to convert.

    Returns:
        Tuple of (MJX model, MJX data).
    """
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    return mjx_model, mjx_data


@jax.jit
def kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
    """JIT-compiled forward kinematics.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.

    Returns:
        Updated MJX data after forward kinematics.
    """
    return smooth.kinematics(mjx_model, mjx_data)


@jax.jit
def com_pos(mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
    """JIT-compiled center-of-mass position calculation.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.

    Returns:
        Updated MJX data with center-of-mass positions.
    """
    return smooth.com_pos(mjx_model, mjx_data)


def get_site_xpos(
    mjx_data: mjx.Data,
    site_idxs: Int[Array, " n_sites"],
) -> Float[Array, "n_sites 3"]:
    """Get Cartesian coordinates of keypoint body sites.

    Args:
        mjx_data: MJX data.
        site_idxs: Indices of sites to query.

    Returns:
        Cartesian positions of the selected sites.
    """
    return mjx_data.site_xpos[site_idxs]


def get_site_pos(
    mjx_model: mjx.Model,
    site_idxs: Int[Array, " n_sites"],
) -> Float[Array, "n_sites 3"]:
    """Get local position offsets of keypoint body sites.

    Args:
        mjx_model: MJX model.
        site_idxs: Indices of sites to query.

    Returns:
        Local position offsets relative to parent body.
    """
    return mjx_model.site_pos[site_idxs]


def set_site_pos(
    mjx_model: mjx.Model,
    offsets: Float[Array, "n_sites 3"],
    site_idxs: Int[Array, " n_sites"],
) -> mjx.Model:
    """Set site positions to given offsets.

    Args:
        mjx_model: MJX model.
        offsets: New local position offsets for the sites.
        site_idxs: Indices of sites to update.

    Returns:
        Updated MJX model with new site positions.
    """
    new_site_pos = mjx_model.site_pos.at[site_idxs].set(offsets)
    mjx_model = mjx_model.replace(site_pos=new_site_pos)
    return mjx_model


def make_qs(
    q0: Float[Array, " n_qpos"],
    qs_to_opt: Bool[Array, " n_qpos"],
    q: Float[Array, " n_qpos"],
) -> Float[Array, " n_qpos"]:
    """Combine initial and optimized joint angles based on optimization mask.

    Args:
        q0: Initial joint angles.
        qs_to_opt: Boolean mask selecting which joints were optimized.
        q: New joint angles from optimization.

    Returns:
        Combined joint angles.
    """
    return jp.copy((1 - qs_to_opt) * q0 + qs_to_opt * jp.copy(q))


def replace_qs(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    q: Float[Array, " n_qpos"] | None,
) -> mjx.Data:
    """Replace joint angles in MJX data and run forward kinematics.

    Args:
        mjx_model: MJX model.
        mjx_data: MJX data.
        q: New joint angles, or None if optimization failed.

    Returns:
        Updated MJX data after forward kinematics.
    """
    if q is None:
        print("optimization failed, continuing")

    else:
        mjx_data = mjx_data.replace(qpos=q)
        mjx_data = kinematics(mjx_model, mjx_data)

    return mjx_data


# Constants used to determine when a rotation is close to a pole.
_POLE_LIMIT = 1.0 - 1e-6
_TOL = 1e-10


def _get_qmat_indices_and_signs() -> tuple[Int[Array, "4 4"], Int[Array, "4 4"]]:
    """Precompute index and sign arrays for quaternion multiplication matrix."""
    w, x, y, z = range(4)
    qmat_idx_and_sign = jp.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )
    indices = jp.abs(qmat_idx_and_sign)
    signs = 2 * (qmat_idx_and_sign >= 0) - 1
    return indices, signs


_qmat_idx, _qmat_sign = _get_qmat_indices_and_signs()


def quat_mul(
    quat1: Float[Array, "*batch 4"],
    quat2: Float[Array, "*batch 4"],
) -> Float[Array, "*batch 4"]:
    """Compute the Hamilton product of two quaternions.

    Supports any number of leading batch dimensions.

    Args:
        quat1: Quaternion [w, i, j, k].
        quat2: Quaternion [w, i, j, k].

    Returns:
        Quaternion product quat1 * quat2.
    """
    # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
    # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
    # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
    # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
    qmat = quat1[..., _qmat_idx] * _qmat_sign
    return (qmat @ quat2[..., None])[..., 0]


def _clip_within_precision(
    number: Float[Array, ""],
    low: float,
    high: float,
    precision: float = _TOL,
) -> Float[Array, ""]:
    """Clip input to range, checking precision.

    Args:
        number: Scalar to clip.
        low: Lower bound.
        high: Upper bound.
        precision: Tolerance for out-of-range values.

    Returns:
        Input clipped to [low, high].
    """
    return jp.clip(number, low, high)


def quat_conj(
    quat: Float[Array, "*batch 4"],
) -> Float[Array, "*batch 4"]:
    """Return conjugate of a quaternion.

    Supports inputs with or without leading batch dimensions.

    Args:
        quat: Quaternion [w, i, j, k].

    Returns:
        Conjugate quaternion [w, -i, -j, -k].
    """
    quat = jp.asarray(quat)
    return jp.stack(
        [quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]], axis=-1
    )


def quat_diff(
    source: Float[Array, "*batch 4"],
    target: Float[Array, "*batch 4"],
) -> Float[Array, "*batch 4"]:
    """Compute quaternion difference from source to target.

    Supports inputs with or without leading batch dimensions.

    Args:
        source: Source quaternion [w, i, j, k].
        target: Target quaternion [w, i, j, k].

    Returns:
        Quaternion representing rotation from source to target.
    """
    return quat_mul(quat_conj(source), target)


def quat_to_axisangle(
    quat: Float[Array, " 4"],
) -> Float[Array, " 3"]:
    """Convert a quaternion to axis-angle representation.

    Args:
        quat: Quaternion [w, i, j, k].

    Returns:
        Axis-angle vector with angle encoded by its length.
    """
    angle = 2 * jp.arccos(_clip_within_precision(quat[0], -1.0, 1.0))

    def true_fn(angle):
        return jp.zeros(3)

    def false_fn(angle):
        qn = jp.sin(angle / 2)
        angle = (angle + jp.pi) % (2 * jp.pi) - jp.pi
        axis = quat[1:4] / qn
        out = axis * angle
        return out

    return jax.lax.cond(angle < _TOL, true_fn, false_fn, angle)


@jaxtyped(typechecker=beartype)
def compute_velocity_from_kinematics(
    qpos_trajectory: Float[Array, "n_frames n_qpos"],
    dt: float,
    freejoint: bool = True,
    max_qvel: float = 20.0,
) -> Float[Array, "n_frames n_qvel"]:
    """Compute velocity trajectory from position trajectory for a continuous clip.

    Assumes freejoint as the first 7 dimensions of qpos when freejoint=True.

    Args:
        qpos_trajectory: Generalized coordinates over time.
        dt: Timestep between qpos entries.
        freejoint: Whether the model has a free joint (first 7 dims).
        max_qvel: Maximum velocity magnitude for clipping.

    Returns:
        Velocity trajectory.
    """
    qpos_trajectory = jp.concatenate(
        [qpos_trajectory, qpos_trajectory[-1, jp.newaxis, :]], axis=0
    )

    if not freejoint:
        qvel_joints = (qpos_trajectory[1:, :] - qpos_trajectory[:-1, :]) / dt
        return jp.clip(qvel_joints, -max_qvel, max_qvel)
    else:
        qvel_joints = (qpos_trajectory[1:, 7:] - qpos_trajectory[:-1, 7:]) / dt
        qvel_translation = (qpos_trajectory[1:, :3] - qpos_trajectory[:-1, :3]) / dt
        qvel_gyro = []
        for t in range(qpos_trajectory.shape[0] - 1):
            normed_diff = quat_diff(
                qpos_trajectory[t, 3:7], qpos_trajectory[t + 1, 3:7]
            )
            normed_diff /= jp.linalg.norm(normed_diff)
            angle = quat_to_axisangle(normed_diff)
            qvel_gyro.append(angle / dt)
        qvel_gyro = jp.stack(qvel_gyro)

        mocap_qvels = jp.concatenate([qvel_translation, qvel_gyro, qvel_joints], axis=1)

        vels = mocap_qvels[:, 6:]
        clipped_vels = jp.clip(vels, -max_qvel, max_qvel)

        return mocap_qvels.at[:, 6:].set(clipped_vels)


@jaxtyped(typechecker=beartype)
def batch_kp_data(
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    n_frames_per_clip: int,
    continuous: bool = False,
) -> Float[Array, "n_clips clip_frames n_keypoints_xyz"]:
    """Reshape keypoint data into batches for parallel processing.

    Args:
        kp_data: Flattened keypoint data.
        n_frames_per_clip: Number of frames per clip.
        continuous: If True, create overlapping batches for edge effect handling.

    Returns:
        Batched keypoint data.
    """
    n_frames = n_frames_per_clip
    total_frames = kp_data.shape[0]
    n_batches = int(total_frames // n_frames)
    if continuous:
        window = n_frames + CONTINUOUS_BATCH_OVERLAP
        if total_frames < window:
            batched_kp_data = kp_data.reshape((n_batches, window) + kp_data.shape[1:])
        else:
            step = n_frames
            starts = jp.arange(0, n_batches * step, step)
            batches = [kp_data[s : s + window] for s in starts]
            batches[-1] = jp.pad(
                batches[-1],
                ((0, CONTINUOUS_BATCH_OVERLAP), (0, 0)),
                mode="wrap",
            )
            batched_kp_data = jp.stack(batches, axis=0)
    else:
        batched_kp_data = kp_data[: int(n_batches) * n_frames]
        batched_kp_data = batched_kp_data.reshape(
            (n_batches, n_frames) + kp_data.shape[1:]
        )

    return batched_kp_data


# TODO: make this more efficient by parallelizing the crossfade operation
def handle_edge_effects(
    ik_only_data: io.StacData, n_frames_per_clip: int
) -> io.StacData:
    """Handle overlapping batch boundaries via sigmoid crossfade.

    Args:
        ik_only_data: IK output data with overlapping batches.
        n_frames_per_clip: Number of frames per clip.

    Returns:
        Data with crossfaded overlap regions and overlaps removed.
    """

    def crossfade_sigmoid(
        a: np.ndarray,
        b: np.ndarray,
        *,
        axis: int = 0,
        center: float = 0.5,
        steepness: float = 10.0,
    ) -> np.ndarray:

        n = a.shape[axis]
        x = np.linspace(0.0, 1.0, n)

        # Numerically stable sigmoid: 0.5 * (1 + tanh(z/2))
        z = steepness * (x - center)
        m = 0.5 * (1.0 + np.tanh(z / 2.0))

        shape = [1] * a.ndim
        shape[axis] = n
        m = m.reshape(shape)

        return (1.0 - m) * a + m * b

    def f(data: np.ndarray) -> np.ndarray:
        batched_data = data.reshape(
            (
                -1,
                n_frames_per_clip + CONTINUOUS_BATCH_OVERLAP,
            )
            + data.shape[1:]
        )

        num_clips = batched_data.shape[0]
        for i in range(num_clips - 1):
            a = batched_data[i, -CONTINUOUS_BATCH_OVERLAP:, :]
            b = batched_data[i + 1, :CONTINUOUS_BATCH_OVERLAP, :]
            cross = crossfade_sigmoid(a, b, axis=0)

            batched_data[i, -CONTINUOUS_BATCH_OVERLAP:, :] = cross

        first_data = batched_data[0, :, :]
        middle_data = batched_data[1:-1, CONTINUOUS_BATCH_OVERLAP:, :]
        last_data = batched_data[
            -1, CONTINUOUS_BATCH_OVERLAP:-CONTINUOUS_BATCH_OVERLAP, :
        ]

        flattened_middle_data = middle_data.reshape((-1,) + middle_data.shape[2:])
        res = np.concatenate([first_data, flattened_middle_data, last_data], axis=0)
        return res

    ik_only_data.qpos = f(ik_only_data.qpos)
    ik_only_data.kp_data = f(ik_only_data.kp_data)
    ik_only_data.xpos = f(ik_only_data.xpos)
    ik_only_data.xquat = f(ik_only_data.xquat)
    ik_only_data.marker_sites = f(ik_only_data.marker_sites)

    return ik_only_data
