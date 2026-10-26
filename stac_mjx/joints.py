"""Joint-aware operations on MuJoCo position and velocity coordinates."""

import jaxlie
import mujoco
import numpy as np
from jax import numpy as jp


def normalize_quaternions(quat):
    """Normalize wxyz quaternions, using identity for a zero initial value."""
    squared_norm = jp.sum(quat**2, axis=-1, keepdims=True)
    normalized = quat / jp.sqrt(jp.maximum(squared_norm, 1e-24))
    return jp.where(squared_norm > 0, normalized, jp.array([1.0, 0.0, 0.0, 0.0]))


class JointLayout:
    """Static indices for scalar coordinates and SO(3) factors in qpos."""

    def __init__(self, joint_types):
        scalar_qpos, scalar_qvel, quaternion_qpos, angular_qvel = [], [], [], []
        nq = nv = 0
        for joint_type in joint_types:
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                scalar_qpos.extend(range(nq, nq + 3))
                scalar_qvel.extend(range(nv, nv + 3))
                nq += 3
                nv += 3
            if joint_type in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                quaternion_qpos.append(range(nq, nq + 4))
                angular_qvel.append(range(nv, nv + 3))
                nq += 4
                nv += 3
            else:
                scalar_qpos.append(nq)
                scalar_qvel.append(nv)
                nq += 1
                nv += 1
        self.nq, self.nv = nq, nv
        self.scalar_qpos = np.asarray(scalar_qpos, dtype=np.int32)
        self.scalar_qvel = np.asarray(scalar_qvel, dtype=np.int32)
        self.quaternion_qpos = np.asarray(quaternion_qpos, dtype=np.int32).reshape(
            -1, 4
        )
        self.angular_qvel = np.asarray(angular_qvel, dtype=np.int32).reshape(-1, 3)

    @classmethod
    def from_qpos(cls, nq, freejoint=True, joint_types=None):
        """Use model joint types, or the legacy free-root-plus-scalars layout."""
        if joint_types is None:
            joint_types = ([mujoco.mjtJoint.mjJNT_FREE] if freejoint else []) + [
                mujoco.mjtJoint.mjJNT_HINGE
            ] * (nq - (7 if freejoint else 0))
        layout = cls(joint_types)
        if layout.nq != nq:
            raise ValueError(
                f"Joint layout expects {layout.nq} qpos coordinates, got {nq}"
            )
        return layout

    def identity(self):
        """Return zero translation/hinge/slide coordinates and wxyz = [1, 0, 0, 0].

        Each quaternion represents zero rotation. This is independent of the model's
        configured qpos0.
        """
        qpos = jp.zeros(self.nq)
        return qpos.at[self.quaternion_qpos[:, 0]].set(1.0)

    def normalize(self, qpos):
        if self.quaternion_qpos.size:
            qpos = qpos.at[..., self.quaternion_qpos].set(
                normalize_quaternions(qpos[..., self.quaternion_qpos])
            )
        return qpos

    def difference(self, source, target):
        """Return nv displacement components in MuJoCo qvel order.

        Scalars: delta_s = s_target - s_source.
        Rotations: delta_theta = log(R_source.inverse() @ R_target), a local
        rotation vector whose direction is the axis and magnitude is radians.
        """
        source, target = jp.broadcast_arrays(source, target)
        result = jp.zeros((*source.shape[:-1], self.nv), dtype=source.dtype)
        result = result.at[..., self.scalar_qvel].set(
            target[..., self.scalar_qpos] - source[..., self.scalar_qpos]
        )
        if self.quaternion_qpos.size:
            before = jaxlie.SO3(
                normalize_quaternions(source[..., self.quaternion_qpos])
            )
            after = jaxlie.SO3(normalize_quaternions(target[..., self.quaternion_qpos]))
            result = result.at[..., self.angular_qvel].set(
                (before.inverse() @ after).log()
            )
        return result

    def retract(self, qpos, delta):
        """Apply an nv-dimensional displacement to qpos.

        Scalars: s_new = s + delta_s.
        Rotations: R_new = R @ exp(delta_theta), using local rotation vectors
        in radians and storing the result as normalized wxyz quaternions.
        """
        result = qpos.at[..., self.scalar_qpos].add(delta[..., self.scalar_qvel])
        if self.quaternion_qpos.size:
            rotation = jaxlie.SO3(
                normalize_quaternions(qpos[..., self.quaternion_qpos])
            )
            rotation = rotation @ jaxlie.SO3.exp(delta[..., self.angular_qvel])
            result = result.at[..., self.quaternion_qpos].set(
                normalize_quaternions(rotation.wxyz)
            )
        return result

    def validate_mask(self, mask):
        """Quaternion components cannot be independently frozen on SO(3)."""
        quaternion_mask = np.asarray(mask)[self.quaternion_qpos]
        if np.any(np.any(quaternion_mask, axis=-1) != np.all(quaternion_mask, axis=-1)):
            raise ValueError(
                "All four coordinates of a quaternion must share the joint mask"
            )

    def regularization(self, qpos, weights):
        """Return weighted residuals relative to the mathematical identity pose.

        Scalars: r_s = sqrt(w_s) * s.
        Rotations: r_R = sqrt(mean(w_quat)) * log(R), averaging the four
        quaternion weights. The squared rotation penalty is sign-invariant.
        """
        tangent_weights = (
            jp.zeros(self.nv).at[self.scalar_qvel].set(weights[self.scalar_qpos])
        )
        if self.quaternion_qpos.size:
            rotation_weights = jp.mean(
                weights[self.quaternion_qpos], axis=-1, keepdims=True
            )
            tangent_weights = tangent_weights.at[self.angular_qvel].set(
                jp.broadcast_to(rotation_weights, self.angular_qvel.shape)
            )
        return jp.sqrt(tangent_weights) * self.difference(self.identity(), qpos)
