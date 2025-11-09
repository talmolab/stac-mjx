"""This module contains utility functions for Stac."""

import os

import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src import smooth
import numpy as np
from stac_mjx import io
from scipy import ndimage
from jax.extend.backend import get_backend

CONTINUOUS_BATCH_OVERLAP = 10


def enable_xla_flags():
    """Enables XLA Flags for faster runtime on Nvidia GPUs."""
    if get_backend().platform == "gpu":
        os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "


def mjx_load(mj_model):
    """Load mujoco model into mjx."""
    # Create mjx model and data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    return mjx_model, mjx_data


@jax.jit
def kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """Jit compiled forward kinematics.

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.kinematics(mjx_model, mjx_data)


@jax.jit
def com_pos(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """Jit compiled com_pos calculation.

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.com_pos(mjx_model, mjx_data)


def get_site_xpos(mjx_data: mjx.Data, site_idxs: jp.ndarray):
    """Get MjxData.site_xpos of keypoint body sites.

    Args:
        mjx_data (mjx.Data):

    Returns:
        jax.Array: MjxData.site_xpos of keypoint body sites, ie
        Cartesian coords of body sites.
    """
    return mjx_data.site_xpos[site_idxs]


def get_site_pos(mjx_model: mjx.Model, site_idxs: jp.ndarray):
    """Get MjxModel.site_pos of keypoint body sites.

    Args:
        mjx_data (mjx.Data):

    Returns:
        jax.Array: MjxModel.site_pos of keypoint body sites, ie
        local position offset rel. to body.
    """
    return mjx_model.site_pos[site_idxs]


def set_site_pos(mjx_model: mjx.Model, offsets, site_idxs: jp.ndarray):
    """Set MjxModel.sites_pos to offsets and returns the new mjx_model.

    Args:
        mjx_model (mjx.Model):
        offsets (jax.Array):

    Returns:
        mjx_model: Resulting mjx.Model
    """
    new_site_pos = mjx_model.site_pos.at[site_idxs].set(offsets)
    mjx_model = mjx_model.replace(site_pos=new_site_pos)
    return mjx_model


def make_qs(q0, qs_to_opt, q):
    """Create new set of qs combining initial and new qs for part optimization based on qs_to_opt.

    Args:
        q0 (jax.Array): initial joint angles
        qs_to_opt (jax.Array): joint angles that were optimized
        q (jax.Array): new joint angles

    Returns:
        jp.Array: resulting set of joint angles
    """
    return jp.copy((1 - qs_to_opt) * q0 + qs_to_opt * jp.copy(q))


def replace_qs(mjx_model: mjx.Model, mjx_data: mjx.Data, q):
    """Replace joint angles in mjx.Data with new ones and performs forward kinematics.

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):
        q (jax.Array): new joint angles

    Returns:
        mjx.Data: resulting mjx Data
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


def _get_qmat_indices_and_signs():
    """Precomputes index and sign arrays for constructing `qmat` in `quat_mul`."""
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


def quat_mul(quat1, quat2):
    """Computes the Hamilton product of two quaternions.

    Any number of leading batch dimensions is supported.

    Args:
      quat1: A quaternion [w, i, j, k].
      quat2: A quaternion [w, i, j, k].

    Returns:
      The quaternion product quat1 * quat2.
    """
    # Construct a (..., 4, 4) matrix to multiply with quat2 as shown below.
    qmat = quat1[..., _qmat_idx] * _qmat_sign

    # Compute the batched Hamilton product:
    # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
    # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
    # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
    # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
    return (qmat @ quat2[..., None])[..., 0]


def _clip_within_precision(number, low, high, precision=_TOL):
    """Clips input to provided range, checking precision.

    Args:
      number: (float) number to be clipped.
      low: (float) lower bound.
      high: (float) upper bound.
      precision: (float) tolerance.

    Returns:
      Input clipped to given range.

    Raises:
      ValueError: If number is outside given range by more than given precision.
    """
    # This is raising an error when jitted
    # def _raise_if_not_in_precision():
    #     if (number < low - precision).any() or (number > high + precision).any():
    #         raise ValueError(
    #             "Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}".format(
    #                 number, low, high, precision
    #             )
    #         )

    # jax.debug.callback(_raise_if_not_in_precision)

    return jp.clip(number, low, high)


def quat_conj(quat):
    """Return conjugate of quaternion.

    This function supports inputs with or without leading batch dimensions.

    Args:
      quat: A quaternion [w, i, j, k].

    Returns:
      A quaternion [w, -i, -j, -k] representing the inverse of the rotation
      defined by `quat` (not assuming normalization).
    """
    # Ensure quat is an np.array in case a tuple or a list is passed
    quat = jp.asarray(quat)
    return jp.stack(
        [quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]], axis=-1
    )


def quat_diff(source, target):
    """Computes quaternion difference between source and target quaternions.

    This function supports inputs with or without leading batch dimensions.

    Args:
      source: A quaternion [w, i, j, k].
      target: A quaternion [w, i, j, k].

    Returns:
      A quaternion representing the rotation from source to target.
    """
    return quat_mul(quat_conj(source), target)


def quat_to_axisangle(quat):
    """Returns the axis-angle corresponding to the provided quaternion.

    Args:
      quat: A quaternion [w, i, j, k].

    Returns:
      axisangle: A 3x1 numpy array describing the axis of rotation, with angle
          encoded by its length.
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


def compute_velocity_from_kinematics(
    qpos_trajectory: jp.ndarray,
    dt: float,
    freejoint: bool = True,
    max_qvel: float = 20.0,
) -> jp.ndarray:
    """Computes velocity trajectory from position trajectory for a continuous clip.

    Args:
        qpos_trajectory (jp.ndarray): trajectory of qpos values T x ?
          Note assumes has freejoint as the first 7 dimensions
        dt (float): timestep between qpos entries

    Returns:
        jp.ndarray: Trajectory of velocities.
    """
    # Padding for velocity corner case.
    qpos_trajectory = jp.concatenate(
        [qpos_trajectory, qpos_trajectory[-1, jp.newaxis, :]], axis=0
    )

    # If there's no freejoint, qpos only has the joint angles so no need for indexing.
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


def batch_kp_data(
    kp_data: jp.ndarray, n_frames_per_clip: int, continuous: bool = False
):
    """Reshape data for parallel processing."""
    n_frames = n_frames_per_clip
    total_frames = kp_data.shape[0]
    n_batches = int(total_frames // n_frames)  # Cut off the last batch if it's not full
    # For continuous data, create overlapping batches (10 frames) to allow for edge effects post-processing
    if continuous:
        window = n_frames + CONTINUOUS_BATCH_OVERLAP
        # If there's only one batch, just reshape to add batch dim
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
        # Reshape the array to create batches
        batched_kp_data = batched_kp_data.reshape(
            (n_batches, n_frames) + kp_data.shape[1:]
        )

    return batched_kp_data


# TODO: make this more efficient by parallelizing the crossfade operation
def handle_edge_effects(ik_only_data: io.StacData, n_frames_per_clip: int):
    """Naive handling: remove the final overlapping frames for each batch.

    Args:
        ik_only_data (io.StacData): ik_only data to be processed
        n_frames_per_clip (int): number of frames per clip

    Returns:
        io.StacData: processed data
    """

    def crossfade_sigmoid(
        a: jp.ndarray,
        b: jp.ndarray,
        *,
        axis: int = 0,
        center: float = 0.5,
        steepness: float = 10.0,
    ) -> jp.ndarray:

        n = a.shape[axis]
        x = jp.linspace(0.0, 1.0, n)

        # Numerically stable sigmoid: 0.5 * (1 + tanh(z/2))
        z = steepness * (x - center)
        m = 0.5 * (1.0 + jp.tanh(z / 2.0))  # shape: (n,)

        # Reshape for broadcasting along the chosen axis
        shape = [1] * a.ndim
        shape[axis] = n
        m = m.reshape(shape)

        return (1.0 - m) * a + m * b

    def f(data: jp.ndarray):
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

            # batched_data = batched_data.at[i, -CONTINUOUS_BATCH_OVERLAP:, :].set(cross)
            batched_data[i, -CONTINUOUS_BATCH_OVERLAP:, :] = cross

        first_data = batched_data[0, :, :]
        middle_data = batched_data[1:-1, CONTINUOUS_BATCH_OVERLAP:, :]
        last_data = batched_data[
            -1, CONTINUOUS_BATCH_OVERLAP:-CONTINUOUS_BATCH_OVERLAP, :
        ]

        flattened_middle_data = middle_data.reshape((-1,) + middle_data.shape[2:])
        res = jp.concatenate([first_data, flattened_middle_data, last_data], axis=0)
        print(f"res shape: {res.shape}")
        return res

    ik_only_data.qpos = f(ik_only_data.qpos)
    ik_only_data.kp_data = f(ik_only_data.kp_data)
    ik_only_data.xpos = f(ik_only_data.xpos)
    ik_only_data.xquat = f(ik_only_data.xquat)
    ik_only_data.marker_sites = f(ik_only_data.marker_sites)

    return ik_only_data
