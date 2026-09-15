"""Multi-camera calibration loading, coordinate transforms, and projection.

Loads calibration.toml (anipose/SLEAP format), handles the MuJoCo ↔ calibration
frame transform (using registration parameters), and projects 3D points to 2D.
"""

import numpy as np
import cv2
import tomllib
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from scipy.spatial.transform import Rotation as RotLib


@dataclass
class CameraParams:
    """Camera intrinsics and extrinsics from calibration."""

    name: str
    size: tuple  # (width, height)
    K: np.ndarray  # 3x3 intrinsic matrix
    dist: np.ndarray  # 5-element distortion coeffs [k1, k2, p1, p2, k3]
    rvec: np.ndarray  # 3-element Rodrigues rotation vector
    tvec: np.ndarray  # 3-element translation vector

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix (3x3) from Rodrigues vector."""
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 projection matrix P = K @ [R|t]."""
        Rt = np.hstack([self.R, self.tvec.reshape(3, 1)])
        return self.K @ Rt

    def world_position(self) -> np.ndarray:
        """Camera position in world coordinates: -R^T @ t."""
        return -self.R.T @ self.tvec

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D world points to 2D pixel coords (no distortion applied).

        Args:
            points_3d: (N, 3) array of 3D points in calibration world frame.

        Returns:
            (N, 2) array of 2D pixel coordinates.
        """
        pts, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            self.rvec,
            self.tvec,
            self.K,
            distCoeffs=None,
        )
        return pts.reshape(-1, 2)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort a video frame using camera intrinsics and distortion."""
        return cv2.undistort(frame, self.K, self.dist)


@dataclass
class RegistrationTransform:
    """Registration transform from calibration frame to MuJoCo model frame.

    Forward (applied during registration):
        p_mj = (p_centered_rescaled @ R_total) + model_shoulder
    where p_centered_rescaled = rescaled(p_calib - shoulder_calib)

    Inverse (for reprojection):
        p_calib_centered = (p_mj - model_shoulder) @ R_total.T
        p_calib = p_calib_centered + shoulder_calib  (per-frame)

    Note: bone rescaling makes this non-rigid, but for mesh reprojection
    we treat it as rigid (rotation + translation) which is a good approximation.
    The units conversion (mm→m) is embedded in the bone rescaling step.
    """

    R_total: np.ndarray  # (3, 3) compound rotation (Procrustes + nudge)
    model_shoulder: np.ndarray  # (3,) = registration_translation in meters
    shoulder_reference_mm: np.ndarray = None  # (3,) fixed shoulder ref in calibration frame (mm), or None for per-frame
    scale_factor: float = 1.0  # per-trial uniform calibration-scale correction
    # (register_v21_mocap.py's shared-rotation_v4 recipe -- see its docstring);
    # forward registration applies centered_m *= scale_factor before rotating,
    # so the inverse must divide by it. Defaults to 1.0 (no-op) for older
    # registrations / other model versions that don't have this field.

    def mujoco_to_calibration(
        self,
        points_mj: np.ndarray,
        shoulder_calib: np.ndarray,
        scale_m_to_mm: float = 1000.0,
    ) -> np.ndarray:
        """Transform points from MuJoCo frame to calibration frame.

        Args:
            points_mj: (..., 3) points in MuJoCo frame (meters).
            shoulder_calib: (3,) shoulder position in calibration frame (mm).
            scale_m_to_mm: scale factor from model units to calibration units.

        Returns:
            (..., 3) points in calibration frame (mm).
        """
        centered = points_mj - self.model_shoulder
        rotated = centered @ self.R_total.T
        unscaled = rotated / self.scale_factor  # undo the per-trial calibration-scale correction
        # Scale from meters to mm
        scaled = unscaled * scale_m_to_mm
        return scaled + shoulder_calib


def load_calibration(toml_path: str) -> List[CameraParams]:
    """Load camera calibration from a TOML file (anipose/SLEAP format).

    Args:
        toml_path: Path to calibration.toml.

    Returns:
        List of CameraParams sorted by camera index.
    """
    with open(toml_path, "rb") as f:
        cal = tomllib.load(f)

    cameras = []
    cam_idx = 0
    while f"cam_{cam_idx}" in cal:
        section = cal[f"cam_{cam_idx}"]
        name = section["name"]
        size = tuple(section["size"])
        K = np.array(section["matrix"], dtype=np.float64)
        dist = np.array(section["distortions"], dtype=np.float64)
        rvec = np.array(section["rotation"], dtype=np.float64)
        tvec = np.array(section["translation"], dtype=np.float64)
        cameras.append(CameraParams(name=name, size=size, K=K, dist=dist, rvec=rvec, tvec=tvec))
        cam_idx += 1

    return cameras


def load_registration_transform(registered_h5_path: str) -> RegistrationTransform:
    """Load registration transform parameters from a registered H5 file.

    The v4 registration applies: centered_rescaled @ R_proc @ R_nudge.T + T
    where R_nudge = Rot_down(10°) * Rot_right(-10°).

    Args:
        registered_h5_path: Path to *_mujoco_registered_forelimb_right_v4.h5.

    Returns:
        RegistrationTransform with compound rotation and translation.
    """
    with h5py.File(registered_h5_path, "r") as f:
        R_proc = np.array(f.attrs["registration_rotation"])
        model_shoulder = np.array(f.attrs["registration_translation"])
        # v6+ registrations store a fixed shoulder reference instead of per-frame
        shoulder_ref = None
        if "shoulder_reference_mm" in f.attrs:
            shoulder_ref = np.array(f.attrs["shoulder_reference_mm"])
        # register_v21_mocap.py's shared-rotation_v4 recipe stores a per-trial
        # uniform calibration-scale correction; absent in older registrations
        # (defaults to 1.0, a no-op).
        scale_factor = float(f.attrs["scale_factor"]) if "scale_factor" in f.attrs else 1.0

    R_total = R_proc

    return RegistrationTransform(
        R_total=R_total, model_shoulder=model_shoulder,
        shoulder_reference_mm=shoulder_ref, scale_factor=scale_factor,
    )


def load_raw_shoulder_positions(points3d_path: str) -> np.ndarray:
    """Load raw shoulder positions from the session's points3d.h5.

    These are the per-frame shoulder positions in calibration frame (mm),
    needed to invert the registration transform.

    Args:
        points3d_path: Path to session points3d.h5 (raw triangulated data).

    Returns:
        (n_frames, 3) shoulder positions in calibration frame (mm).
    """
    with h5py.File(points3d_path, "r") as f:
        tracks = f["tracks"][:]  # (frames, 1, 3_nodes, 3_xyz)
    return tracks[:, 0, 0, :]  # shoulder = first node


def load_env_points3d(h5_path: str) -> dict:
    """Load 3D environmental marker positions from env points3d.h5.

    Args:
        h5_path: Path to env points3d.h5 file.

    Returns:
        dict with 'points' (N, 3) array and 'names' list of node names.
    """
    with h5py.File(h5_path, "r") as f:
        points = f["points_3d"][()]  # (1, 1, N, 3)
        names = [n.decode() if isinstance(n, bytes) else n for n in f["node_names"][()]]

    points = points.squeeze()  # (N, 3)
    return {"points": points, "names": names}
