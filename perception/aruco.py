# Using ArUco markers for precise pose estimation of the tray, grippers,
# and arm bases.  There are 6 tags: one on each arm base, one on each
# gripper, and two on the tray (left and right).
#
# cv2.aruco.estimatePoseSingleMarkers was removed in OpenCV 4.8.
# We use cv2.solvePnP with the known marker geometry instead.

import cv2
import numpy as np
from dataclasses import dataclass


# Assign the markers
@dataclass
class MarkerConfig:
    """
    Maps physical ArUco marker IDs to semantic roles.

    Attributes
    ----------
    dictionary : int
        OpenCV dictionary constant.
    marker_size_m : float
        Physical side length of gripper/tray markers in meters.
    large_marker_size_m : float
        Physical side length of base + gripper markers in meters.
    left_base : int
        Marker ID on the left arm's base.
    left_gripper : int
        Marker ID on the left arm's gripper.
    right_base : int
        Marker ID on the right arm's base.
    tray_right : int
        Marker ID on the right side of the tray.
    right_gripper : int
        Marker ID on the right arm's gripper.
    tray_left : int
        Marker ID on the left side of the tray.
    tray_marker_spacing_m : float
        Horizontal distance between the two tray markers in meters.
    """
    dictionary: int = cv2.aruco.DICT_APRILTAG_16h5  # technically AprilTags
    marker_size_m: float = 0.015       # 15 mm tray tags
    large_marker_size_m: float = 0.030 # 30 mm base + gripper tags
    # IDs 0-5, left to right across the camera frame
    left_base: int = 0
    left_gripper: int = 1
    right_base: int = 2
    tray_right: int = 3
    right_gripper: int = 4
    tray_left: int = 5
    tray_marker_spacing_m: float = 0.14  # 140 mm between the 2 tray markers


# readable labels for visualization
MARKER_LABELS: dict[str, str] = {}  # populated by _build_labels()


def _build_labels(config: MarkerConfig) -> dict[int, str]:
    return {
        config.left_base: "left_base",
        config.left_gripper: "left_gripper",
        config.right_base: "right_base",
        config.tray_right: "tray_right",
        config.right_gripper: "right_gripper",
        config.tray_left: "tray_left",
    }


# Detector
class ArucoDetector:
    """Detects markers and computes their 3D poses in camera frame."""

    def __init__(self, config: MarkerConfig, camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray):
        """
        Parameters
        ----------
        config : MarkerConfig
            Marker ID assignments and physical dimensions.
        camera_matrix : ndarray, shape (3, 3)
            Camera intrinsic matrix from RealSenseCamera.
        dist_coeffs : ndarray, shape (5,) or (8,)
            Lens distortion coefficients (often near-zero for D435i).
        """
        self.config = config
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(config.dictionary)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.labels = _build_labels(config)
        self._large_ids = {
            config.left_base, config.right_base,
            config.left_gripper, config.right_gripper,
        }

    def _marker_size(self, marker_id: int) -> float:
        """Return the physical marker size for a given marker ID."""
        if marker_id in self._large_ids:
            return self.config.large_marker_size_m
        return self.config.marker_size_m

    def _solve_pnp_single(
        self, corner: np.ndarray, marker_size_m: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Run solvePnP for one marker corner set.

        Returns (rvec (3,), tvec (3,), success).
        """
        half = marker_size_m / 2.0
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)
        img_pts = corner.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if ok:
            return rvec.reshape(3), tvec.reshape(3), True
        return np.zeros(3), np.zeros(3), False

    def _estimate_pose(
        self, corners: tuple, ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate rotation and translation for each detected marker via solvePnP.

        Uses the correct physical marker size per ID (base tags may differ
        from gripper/tray tags).

        Parameters
        ----------
        corners : tuple of ndarray
            Corner pixel coordinates from detectMarkers().
        ids : ndarray, optional
            Marker IDs (used to look up per-marker size).  If None, the
            default ``marker_size_m`` is used for all markers.

        Returns
        -------
        rvecs : ndarray, shape (N, 1, 3)
        tvecs : ndarray, shape (N, 1, 3)
        """
        flat_ids = ids.flatten() if ids is not None else None

        rvecs = []
        tvecs = []
        for i, corner in enumerate(corners):
            mid = int(flat_ids[i]) if flat_ids is not None else -1
            size = self._marker_size(mid) if mid >= 0 else self.config.marker_size_m
            rv, tv, ok = self._solve_pnp_single(corner, size)
            rvecs.append(rv.reshape(1, 3))
            tvecs.append(tv.reshape(1, 3))

        return np.array(rvecs), np.array(tvecs)

    def detect(self, color_frame: np.ndarray) -> dict[int, np.ndarray]:
        """
        Detect all ArUco markers in an RGB frame.

        Parameters
        ----------
        color_frame : ndarray, shape (H, W, 3)
            BGR image.

        Returns
        -------
        poses : dict[int, ndarray]
            Mapping from marker ID to 4x4 homogeneous transform (camera frame).
            Only includes markers that were successfully detected.
        """
        corners, ids, _ = self.detector.detectMarkers(color_frame)

        if ids is None or len(ids) == 0:
            return {}

        rvecs, tvecs = self._estimate_pose(corners, ids)

        poses: dict[int, np.ndarray] = {}
        for i, marker_id in enumerate(ids.flatten()):
            R, _ = cv2.Rodrigues(rvecs[i])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvecs[i].flatten()
            poses[int(marker_id)] = T

        return poses

    def detect_raw(self, color_frame: np.ndarray):
        """
        Return raw detection data for visualization.

        Returns
        -------
        corners : tuple of ndarray
            Corner pixel coordinates for each detected marker.
        ids : ndarray or None
            Detected marker IDs.
        rvecs, tvecs : ndarray or None
            Rotation/translation vectors for each marker.
        """
        corners, ids, _ = self.detector.detectMarkers(color_frame)

        if ids is None or len(ids) == 0:
            return corners, ids, None, None

        rvecs, tvecs = self._estimate_pose(corners, ids)
        return corners, ids, rvecs, tvecs

    def get_tray_poses(self, poses: dict[int, np.ndarray]) -> dict[str, np.ndarray] | None:
        """
        Extract tray left/right 3D positions from the two side markers.

        Returns None if either tray marker is missing.

        Returns
        -------
        dict with keys:
            'tray_left'  — ndarray (3,) position of left tray marker
            'tray_right' — ndarray (3,) position of right tray marker
        """
        left_id = self.config.tray_left
        right_id = self.config.tray_right

        if left_id not in poses or right_id not in poses:
            return None

        return {
            "tray_left": poses[left_id][:3, 3].copy(),
            "tray_right": poses[right_id][:3, 3].copy(),
        }

    def get_gripper_pose(self, poses: dict[int, np.ndarray],
                         side: str) -> np.ndarray | None:
        """
        Extract a gripper's 3D position from its ArUco marker.

        Parameters
        ----------
        side : 'left' or 'right'

        Returns
        -------
        position : ndarray (3,) or None if marker not visible
        """
        marker_id = (self.config.left_gripper if side == "left"
                     else self.config.right_gripper)

        if marker_id not in poses:
            return None

        return poses[marker_id][:3, 3].copy()

    def get_base_pose(self, poses: dict[int, np.ndarray],
                      side: str) -> np.ndarray | None:
        """
        Extract an arm base's full 4x4 transform from its ArUco marker.

        Parameters
        ----------
        side : 'left' or 'right'

        Returns
        -------
        T : ndarray (4, 4) or None if marker not visible
            Homogeneous transform (base-tag frame in camera frame).
        """
        marker_id = (self.config.left_base if side == "left"
                     else self.config.right_base)

        if marker_id not in poses:
            return None

        return poses[marker_id].copy()

    def get_tray_marker_pose(self, poses: dict[int, np.ndarray],
                             side: str) -> np.ndarray | None:
        """
        Extract one tray marker's 3D position.

        Parameters
        ----------
        side : 'left' or 'right'
            Which tray marker (left arm tracks tray_left, right tracks tray_right).

        Returns
        -------
        position : ndarray (3,) or None if marker not visible
        """
        marker_id = (self.config.tray_left if side == "left"
                     else self.config.tray_right)

        if marker_id not in poses:
            return None

        return poses[marker_id][:3, 3].copy()
