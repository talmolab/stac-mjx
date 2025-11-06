"""This module contains utility functions for Stac."""

import os

import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src import smooth
import numpy as np
from stac_mjx import io
from scipy import ndimage


def enable_xla_flags():
    """Enables XLA Flags for faster runtime on Nvidia GPUs."""
    if jax.extend.backend.get_backend().platform == "gpu":
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
    if (
        continuous
    ):  # For continuous data, create overlapping batches (10 frames) to allow for edge effects post-processing
        overlap = 10
        window = n_frames + overlap
        if (
            total_frames < window
        ):  # If there's no need to overlap just reshape to add batch dim
            batched_kp_data = kp_data.reshape((n_batches, window) + kp_data.shape[1:])
        else:
            step = n_frames
            starts = jp.arange(0, n_batches * step, step)
            batches = [kp_data[s : s + window] for s in starts]
            batches[-1] = jp.pad(
                batches[-1], ((0, overlap), (0, 0), (0, 0)), mode="mean"
            )
            batched_kp_data = jp.stack(batches, axis=0)
    else:
        batched_kp_data = kp_data[: int(n_batches) * n_frames]
        # Reshape the array to create batches
        batched_kp_data = batched_kp_data.reshape(
            (n_batches, n_frames) + kp_data.shape[1:]
        )

    return batched_kp_data


# def smooth_edge_effects(qpos: jp.ndarray, segment_length: int):
#     """Smooth the edge effects of the qposes.

#     Args:
#         qpos (jp.ndarray): qpos to smooth
#         segment_length (int): length of each segment
#     Returns:
#         jp.ndarray: smoothed qposes
#     """
#     # Smooth the edge effects from segment_length-frame segments
#     n_frames, n_joints = qpos.shape

#     # Create a copy to modify
#     qpos_smoothed = qpos.copy()

#     # Define smoothing parameters with increased blending
#     smooth_window = (
#         200  # frames to smooth on each side of boundary (increased for more blending)
#     )
#     blend_window = (
#         60  # frames for blending the smoothed region (increased for more blending)
#     )

#     for segment_idx in range(1, n_frames // segment_length):
#         boundary_frame = segment_idx * segment_length

#         # Define the region to smooth (around the boundary)
#         start_frame = max(0, boundary_frame - smooth_window)
#         end_frame = min(n_frames, boundary_frame + smooth_window)

#         # Apply multi-pass smoothing for better results
#         for joint_idx in range(n_joints):
#             segment = qpos_smoothed[start_frame:end_frame, joint_idx].copy()

#             # First pass: Gaussian smoothing with much larger sigma for heavy smoothing
#             smoothed_segment = ndimage.gaussian_filter1d(segment, sigma=7.0)

#             # Second pass: Additional median filter to remove outliers
#             smoothed_segment = ndimage.median_filter(smoothed_segment, size=7)

#             # Third pass: Another round of heavy Gaussian smoothing
#             smoothed_segment = ndimage.gaussian_filter1d(smoothed_segment, sigma=5.0)

#             # Fourth pass: Final gentle Gaussian smoothing for extra smoothness
#             smoothed_segment = ndimage.gaussian_filter1d(smoothed_segment, sigma=2.0)
#             # Create smooth blending weights with extended transition zones
#             blend_start = max(0, boundary_frame - blend_window - start_frame)
#             blend_end = min(len(segment), boundary_frame + blend_window - start_frame)

#             # Use cosine-based blending weights for smoother transitions
#             weights = np.ones_like(segment)
#             if blend_start > 0:
#                 # Cosine transition from 0 to 1
#                 t = np.linspace(0, np.pi / 2, blend_start)
#                 weights[:blend_start] = np.sin(t)
#             if blend_end < len(segment):
#                 # Cosine transition from 1 to 0
#                 t = np.linspace(0, np.pi / 2, len(segment) - blend_end)
#                 weights[blend_end:] = np.cos(t)

#             # Apply weighted blending with additional feathering
#             blended_segment = weights * smoothed_segment + (1 - weights) * segment

#             # Apply additional feathering at the edges of the blend region
#             feather_size = min(5, blend_start // 3, (len(segment) - blend_end) // 3)
#             if feather_size > 0:
#                 # Feather the start
#                 if blend_start > feather_size:
#                     feather_weights = np.linspace(0, 1, feather_size)
#                     blended_segment[blend_start - feather_size : blend_start] = (
#                         feather_weights
#                         * blended_segment[blend_start - feather_size : blend_start]
#                         + (1 - feather_weights)
#                         * segment[blend_start - feather_size : blend_start]
#                     )
#                 # Feather the end
#                 if blend_end + feather_size < len(segment):
#                     feather_weights = np.linspace(1, 0, feather_size)
#                     blended_segment[blend_end : blend_end + feather_size] = (
#                         feather_weights
#                         * blended_segment[blend_end : blend_end + feather_size]
#                         + (1 - feather_weights)
#                         * segment[blend_end : blend_end + feather_size]
#                     )

#             qpos_smoothed[start_frame:end_frame, joint_idx] = blended_segment

#     # Final global smoothing pass with enhanced blending
#     for joint_idx in range(n_joints):
#         # Apply two-stage global smoothing for better blending
#         qpos_smoothed[:, joint_idx] = ndimage.gaussian_filter1d(
#             qpos_smoothed[:, joint_idx], sigma=1.2, mode="nearest"
#         )
#         # Second pass with lighter smoothing to maintain detail
#         qpos_smoothed[:, joint_idx] = ndimage.gaussian_filter1d(
#             qpos_smoothed[:, joint_idx], sigma=0.6, mode="nearest"
#         )

#     print(f"Smoothed {n_frames // segment_length - 1} segment boundaries")
#     print(f"Applied enhanced multi-pass smoothing with cosine blending and feathering")
#     print(f"Smoothed data shape: {qpos_smoothed.shape}")

#     return qpos_smoothed
