"""
Allows testing that the markers are correctly detected and tracked.

Shows the camera feed with detected markers outlined, labeled with their
semantic role (tray_left, tray_right, left_gripper, right_gripper), and
annotated with their 3D position in camera frame.

Usage:
    python perception/test_perception.py

Press ctrl+C to end.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig

# Colors for each marker role (BGR)
COLORS = {
    "tray_left":     (0, 255, 255),   # yellow
    "tray_right":    (0, 165, 255),   # orange
    "left_gripper":  (0, 255, 0),     # green
    "right_gripper": (255, 0, 0),     # blue
}
UNKNOWN_COLOR = (128, 128, 128)       # grey for unrecognized IDs


def main():
    camera = RealSenseCamera()
    config = MarkerConfig()

    print("Connecting to camera...")
    camera.connect()
    print(f"Camera connected (S/N: {camera.serial_number})")
    print(f"Intrinsics:\n{camera.get_camera_matrix()}\n")

    detector = ArucoDetector(
        config=config,
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )

    print("Showing live feed. Press Q to quit.\n")
    print(f"Looking for markers:  tray_left={config.tray_left}  "
          f"tray_right={config.tray_right}  "
          f"left_gripper={config.left_gripper}  "
          f"right_gripper={config.right_gripper}")

    try:
        while True:
            frame = camera.get_frame()
            corners, ids, rvecs, tvecs = detector.detect_raw(frame)

            # Draw detected markers
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    marker_id = int(marker_id)
                    label = detector.labels.get(marker_id)
                    color = COLORS.get(label, UNKNOWN_COLOR) if label else UNKNOWN_COLOR

                    # Draw marker outline
                    pts = corners[i][0].astype(int)
                    cv2.polylines(frame, [pts], True, color, 2)

                    # Draw axis on the marker
                    if rvecs is not None:
                        cv2.drawFrameAxes(frame, detector.camera_matrix,
                                          detector.dist_coeffs,
                                          rvecs[i], tvecs[i],
                                          config.marker_size_m * 0.5)

                    # Label and 3D position
                    pos = tvecs[i].flatten() if tvecs is not None else None
                    tag = label if label else f"id={marker_id}"
                    top_left = pts[0]

                    cv2.putText(frame, tag, (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if pos is not None:
                        pos_str = f"({pos[0]*1000:.0f}, {pos[1]*1000:.0f}, {pos[2]*1000:.0f}) mm"
                        cv2.putText(frame, pos_str, (top_left[0], top_left[1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Tilt info if both tray markers visible
            poses = detector.detect(frame)
            tray = detector.get_tray_poses(poses)
            if tray is not None:
                dz = tray["tray_right"][2] - tray["tray_left"][2]
                spacing = config.tray_marker_spacing_m
                tilt_deg = np.degrees(np.arctan2(dz, spacing)) if spacing > 0 else 0.0
                cv2.putText(frame, f"Tray tilt: {tilt_deg:.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            # Count detected markers
            n = len(ids) if ids is not None else 0
            cv2.putText(frame, f"Markers: {n}/4", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("ArUco Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        camera.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
