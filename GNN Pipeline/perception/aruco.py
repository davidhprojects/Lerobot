# Using ArUco markers for precise pose estimation of the tray and grippers.
# There are 4 tags: one on each gripper, and two on the tray (left and right).

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
        OpenCV dictionary constant
    marker_size_m : float
        Physical side length of each printed marker in meters.
    tray_left : int
        Marker ID on the left side of the tray.
    tray_right : int
        Marker ID on the right side of the tray.
    left_gripper : int
        Marker ID on the left (black) arm's gripper.
    right_gripper : int
        Marker ID on the right (white) arm's gripper.
    tray_marker_spacing_m : float
        Horizontal distance between the two tray markers in meters.
        Used to compute tilt angle from their height difference.
    """
    dictionary: int = cv2.aruco.DICT_APRILTAG_16h5 # We're technically using AprilTags, not ArUco 
    marker_size_m: float = 0.015  # 15 mm tag size (inner pattern, not border): we learned this the hard way
    # 1-4: left to right across the camera frame
    left_gripper: int = 1
    tray_left: int = 2
    tray_right: int = 3
    right_gripper: int = 4
    tray_marker_spacing_m: float = 0.14 # 140 mm between the 2 tray markers


# readable labels for visualization
MARKER_LABELS: dict[str, str] = {}  # populated by _build_labels()


def _build_labels(config: MarkerConfig) -> dict[int, str]:
    return {
        config.tray_left: "tray_left",
        config.tray_right: "tray_right",
        config.left_gripper: "left_gripper",
        config.right_gripper: "right_gripper",
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

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.config.marker_size_m,
            self.camera_matrix, self.dist_coeffs,
        )

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

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.config.marker_size_m,
            self.camera_matrix, self.dist_coeffs,
        )
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
