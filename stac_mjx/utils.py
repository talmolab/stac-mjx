"""Utility functions for STAC."""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if (
    os.environ.get("MUJOCO_GL") == "osmesa"
    and os.environ.get("PYOPENGL_PLATFORM") != "osmesa"
):
    os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import Array
from jax import numpy as jp
import jaxlie
from jaxtyping import Float, Int
from jaxtyping import jaxtyped
from beartype import beartype
import mujoco

if not hasattr(mujoco, "_enums"):
    for name in list(sys.modules):
        if name == "mujoco" or name.startswith("mujoco."):
            del sys.modules[name]
    import mujoco

from mujoco import mjx
from mujoco.mjx._src import smooth
import numpy as np
from scipy import ndimage
from jax.extend.backend import get_backend


def _enable_jax_compilation_cache() -> None:
    """Enable persistent JAX compilation cache for all entrypoints."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "stac-mjx", "jax")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


_enable_jax_compilation_cache()


def enable_xla_flags() -> None:
    """Enable XLA flags for faster runtime on Nvidia GPUs."""
    if get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "


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


def make_context_window(
    data: Float[Array, "n_frames n_features"],
    start: int,
    n_center_frames: int,
    n_context_frames: int,
    n_solve_frames: int,
) -> tuple[Float[Array, "n_solve_frames n_features"], int]:
    """Create a fixed-shape context-padded window from a frame sequence."""
    total_frames = int(data.shape[0])
    n_frames_to_output = min(n_center_frames, total_frames - start)
    data_start = max(start - n_context_frames, 0)
    data_stop = min(start + n_center_frames + n_context_frames, total_frames)
    prefix = max(n_context_frames - start, 0)
    suffix = n_solve_frames - prefix - (data_stop - data_start)

    parts = []
    if prefix > 0:
        parts.append(jp.repeat(data[:1], prefix, axis=0))
    parts.append(data[data_start:data_stop])
    if suffix > 0:
        parts.append(jp.repeat(data[-1:], suffix, axis=0))
    return jp.concatenate(parts, axis=0), n_frames_to_output


def normalize_freejoint_quat(
    qpos: Float[Array, "n_frames n_qpos"],
    freejoint: bool,
) -> Float[Array, "n_frames n_qpos"]:
    """Normalize freejoint quaternions in a batched qpos array."""
    if not freejoint:
        return qpos
    quat = qpos[:, 3:7]
    quat_norm = jp.linalg.norm(quat, axis=-1, keepdims=True)
    return qpos.at[:, 3:7].set(quat / jp.where(quat_norm > 0, quat_norm, 1.0))


def interpolate_qpos_from_keyframes(
    keyframe_qpos: Float[Array, "n_keyframes n_qpos"],
    keyframe_indices: Int[Array, " n_keyframes"],
    n_frames: int,
    freejoint: bool,
) -> Float[Array, "n_frames n_qpos"]:
    """Interpolate sparse qpos keyframes onto every frame.

    For each frame, find the neighboring keyframes and interpolate between them.
    For free joints, interpolate root xyz and hinge coordinates linearly, but
    interpolate root rotation on SO(3) using log/exp instead of raw quaternion
    interpolation.
    """
    frame_ids = jp.arange(n_frames)

    # For each frame, find bracketing keyframes: left <= frame < right.
    right = jp.searchsorted(keyframe_indices, frame_ids, side="right")
    right = jp.clip(right, 1, keyframe_qpos.shape[0] - 1)
    left = right - 1

    # Fractional position of each frame between its two keyframes.
    left_t = keyframe_indices[left]
    right_t = keyframe_indices[right]
    alpha = (
        (frame_ids - left_t).astype(keyframe_qpos.dtype)
        / jp.maximum(right_t - left_t, 1).astype(keyframe_qpos.dtype)
    )[:, None]
    q_left = keyframe_qpos[left]
    q_right = keyframe_qpos[right]

    if freejoint:
        # MuJoCo free-joint qpos layout: [root xyz, root quat wxyz, hinges...].
        xyz = (1.0 - alpha) * q_left[:, :3] + alpha * q_right[:, :3]
        left_wxyz = q_left[:, 3:7]
        right_wxyz = q_right[:, 3:7]
        left_wxyz = left_wxyz / jp.maximum(
            jp.linalg.norm(left_wxyz, axis=-1, keepdims=True), 1e-12
        )
        right_wxyz = right_wxyz / jp.maximum(
            jp.linalg.norm(right_wxyz, axis=-1, keepdims=True), 1e-12
        )

        # Geodesic rotation interpolation:
        # R = R_left exp(alpha log(R_left^{-1} R_right)).
        left_rot = jaxlie.SO3(wxyz=left_wxyz)
        right_rot = jaxlie.SO3(wxyz=right_wxyz)
        rot = left_rot @ jaxlie.SO3.exp(alpha * (left_rot.inverse() @ right_rot).log())
        hinges = (1.0 - alpha) * q_left[:, 7:] + alpha * q_right[:, 7:]
        return normalize_freejoint_quat(
            jp.concatenate([xyz, rot.wxyz, hinges], axis=-1), freejoint=True
        )

    # Plain qpos: simple linear interpolation.
    return (1.0 - alpha) * q_left + alpha * q_right


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
