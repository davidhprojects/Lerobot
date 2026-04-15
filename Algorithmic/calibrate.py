"""
Calibrate camera-to-base transforms by detecting AprilTags on the arm bases.

The arms should be in their home position (or moved aside) so the base
tags are unoccluded.  The script averages detections over N_FRAMES to
reduce noise, then saves a T_cam_base 4x4 transform per arm.

Usage:
    python algorithmic/calibrate.py

The saved transforms are used by all downstream scripts (recording,
runtime hover).  Re-run this whenever the camera moves.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig

# How many frames to average over (at 30 fps ≈ 1 second)
N_FRAMES = 30

# Output path
TRANSFORMS_FILE = Path(__file__).parent / "base_transforms.json"


def average_poses(
    rvecs: list[np.ndarray], tvecs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average a set of Rodrigues rotation vectors and translation vectors."""
    mean_rvec = np.mean(rvecs, axis=0)
    mean_tvec = np.mean(tvecs, axis=0)
    return mean_rvec, mean_tvec


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rvec + tvec to a 4x4 homogeneous transform."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def align_base_frame_vertical(T_cam_base: np.ndarray) -> np.ndarray:
    """
    Rotate the base-tag frame so that base-Y is exactly the vertical axis.

    Builds a corrected frame where:
      - Y axis = camera Y direction (vertical / gravity)
      - Z axis = the tag's most-depth-like axis, projected onto the
        horizontal plane (perpendicular to Y)
      - X axis = right-hand cross product of Y and Z

    This handles arbitrarily tilted tags — not just 90-degree rotations.

    Returns the corrected T_cam_base.
    """
    R = T_cam_base[:3, :3]

    # Camera Y is approximately vertical (camera is roughly level).
    cam_y = np.array([0.0, 1.0, 0.0])
    dots = R.T @ cam_y  # projection of camera-Y onto each tag axis
    vertical_idx = int(np.argmax(np.abs(dots)))

    axis_names = ["X", "Y", "Z"]
    print(f"    Vertical alignment (cam_Y dot tag_axis): "
          f"X={dots[0]:+.3f}  Y={dots[1]:+.3f}  Z={dots[2]:+.3f}")
    print(f"    Tag axis most aligned with vertical: {axis_names[vertical_idx]} "
          f"(|{abs(dots[vertical_idx]):.3f}|)")

    # New Y axis = negative camera Y (camera Y points down, we want Y up)
    new_y = -cam_y

    # Pick the tag axis most aligned with depth (camera Z) to anchor
    # the horizontal orientation.  Use whichever of X/Y/Z tag axes has
    # the largest camera-Z component (excluding the vertical one).
    cam_z = np.array([0.0, 0.0, 1.0])
    z_dots = R.T @ cam_z  # projection of camera-Z onto each tag axis
    # Zero out the vertical axis so we don't pick it again
    z_dots[vertical_idx] = 0.0
    depth_idx = int(np.argmax(np.abs(z_dots)))

    # The tag's depth-like axis in camera frame, projected onto the
    # horizontal plane (remove any vertical component)
    tag_depth_cam = R[:, depth_idx].copy()
    tag_depth_cam -= np.dot(tag_depth_cam, new_y) * new_y
    tag_depth_cam /= np.linalg.norm(tag_depth_cam)
    new_z = tag_depth_cam

    # X = Y cross Z (right-handed)
    new_x = np.cross(new_y, new_z)
    new_x /= np.linalg.norm(new_x)

    # Build the corrected rotation: columns are the new axes in camera frame
    R_corrected = np.column_stack([new_x, new_y, new_z])

    T_corrected = np.eye(4)
    T_corrected[:3, :3] = R_corrected
    T_corrected[:3, 3] = T_cam_base[:3, 3]
    return T_corrected


def main():
    print("Connecting camera...")
    camera = RealSenseCamera()
    camera.connect()
    print(f"  Camera connected (S/N: {camera.serial_number})")

    cfg = MarkerConfig()
    detector = ArucoDetector(
        config=cfg,
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )

    base_tags = {
        "left":  cfg.left_base,
        "right": cfg.right_base,
    }

    # Collect detections over N_FRAMES
    print(f"\nCollecting {N_FRAMES} frames...")
    print("  Ensure both base tags are visible and the arms are out of the way.\n")

    collected: dict[str, dict] = {
        side: {"rvecs": [], "tvecs": []}
        for side in base_tags
    }

    for frame_idx in range(N_FRAMES):
        frame = camera.get_frame()
        corners, ids, rvecs, tvecs = detector.detect_raw(frame)

        if ids is None:
            continue

        flat_ids = ids.flatten()
        for side, tag_id in base_tags.items():
            matches = np.where(flat_ids == tag_id)[0]
            if len(matches) == 1:
                idx = matches[0]
                collected[side]["rvecs"].append(rvecs[idx].flatten())
                collected[side]["tvecs"].append(tvecs[idx].flatten())

        # Progress
        status_parts = []
        for side in base_tags:
            n = len(collected[side]["rvecs"])
            status_parts.append(f"{side}={n}")
        print(f"\r  Frame {frame_idx+1}/{N_FRAMES}  ({', '.join(status_parts)})", end="", flush=True)

    print("\n")

    # Average and save
    transforms: dict[str, list] = {}
    for side, tag_id in base_tags.items():
        n = len(collected[side]["rvecs"])
        if n == 0:
            print(f"  WARNING: {side} base tag (ID {tag_id}) was never detected!")
            print(f"           Ensure the tag is printed, attached, and visible.")
            continue

        mean_rvec, mean_tvec = average_poses(
            collected[side]["rvecs"], collected[side]["tvecs"]
        )
        T_raw = rvec_tvec_to_T(mean_rvec, mean_tvec)

        pos = mean_tvec * 1000
        print(f"  {side} base (ID {tag_id}): {n}/{N_FRAMES} detections")
        print(f"    position = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm")

        # Correct axis alignment so base-Y is always vertical
        T = align_base_frame_vertical(T_raw)

        transforms[f"T_cam_base_{side}"] = T.tolist()

    if not transforms:
        print("\nERROR: No base tags detected. Cannot save calibration.")
        camera.disconnect()
        sys.exit(1)

    with open(TRANSFORMS_FILE, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"\n  Saved to {TRANSFORMS_FILE}")

    camera.disconnect()
    print("  Done.")


def load_base_transforms(
    path: Path = TRANSFORMS_FILE,
) -> dict[str, np.ndarray]:
    """
    Load saved camera-to-base transforms.

    Returns
    -------
    transforms : dict
        Keys are 'left' and/or 'right', values are 4x4 ndarray.

    Raises
    ------
    FileNotFoundError
        If calibration has not been run yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Base transforms not found at {path}. "
            f"Run `python algorithmic/calibrate.py` first."
        )
    with open(path) as f:
        data = json.load(f)

    result = {}
    for side in ("left", "right"):
        key = f"T_cam_base_{side}"
        if key in data:
            result[side] = np.array(data[key])
    return result


if __name__ == "__main__":
    main()
