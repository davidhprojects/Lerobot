"""
Intel RealSense D435i interface (RGB-only).

Handles camera connection, RGB frame capture, and provides the camera
intrinsics needed by ArUco pose estimation (estimatePoseSingleMarkers
derives full 3D position from the known marker size — no depth stream
needed).

The camera serial number is loaded from Setup/ports.json so the correct
device is opened if multiple RealSense cameras are attached.
"""

import json
import numpy as np
from pathlib import Path

import pyrealsense2 as rs

PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"

# Target capture resolution and frame rate
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 30


class RealSenseCamera:
    """Manages the D435i lifecycle: connect, capture RGB frames, disconnect."""

    def __init__(self, serial_number: str | None = None):
        """
        Parameters
        ----------
        serial_number : str, optional
            D435i serial number.  If None, loads from ports.json.
        """
        if serial_number is None:
            serial_number = self._load_serial()
        self.serial_number = serial_number
        self.pipeline = None
        self.profile = None
        self._camera_matrix = None
        self._dist_coeffs = None

    def connect(self):
        """Start the RealSense pipeline (color stream only)."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT,
                             rs.format.bgr8, FPS)

        self.profile = self.pipeline.start(config)

        # Read intrinsics from the color stream profile
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self._camera_matrix = np.array([
            [intrinsics.fx, 0.0,           intrinsics.ppx],
            [0.0,           intrinsics.fy,  intrinsics.ppy],
            [0.0,           0.0,            1.0],
        ])
        self._dist_coeffs = np.array(intrinsics.coeffs[:5])

    def get_frame(self) -> np.ndarray:
        """
        Capture one RGB frame.

        Returns
        -------
        color : ndarray, shape (H, W, 3), dtype uint8
            BGR image (OpenCV convention).
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def get_camera_matrix(self) -> np.ndarray:
        """
        Return the 3x3 camera intrinsic matrix.

        Must be called after connect().
        """
        if self._camera_matrix is None:
            raise RuntimeError("Call connect() before get_camera_matrix().")
        return self._camera_matrix

    def get_dist_coeffs(self) -> np.ndarray:
        """
        Return the lens distortion coefficients.

        The D435i has very low distortion; these are often near-zero.
        """
        if self._dist_coeffs is None:
            raise RuntimeError("Call connect() before get_dist_coeffs().")
        return self._dist_coeffs

    def disconnect(self):
        """Stop the pipeline and release the device."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    @staticmethod
    def _load_serial() -> str:
        if not PORTS_FILE.exists():
            raise FileNotFoundError(
                f"ports.json not found at {PORTS_FILE}. Run find_ports.py first."
            )
        with open(PORTS_FILE) as f:
            ports = json.load(f)
        if "camera" not in ports:
            raise KeyError("No camera entry in ports.json. Run find_ports.py first.")
        return ports["camera"]["serial_number"]
