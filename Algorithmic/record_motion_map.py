"""
Record motion map data for one arm.

With torque disabled, the user manually sweeps the arm across the workspace
while keeping the gripper marker roughly at the target hover height and
squared to the camera.  Two windows show real-time feedback:

  Window 1 — Camera feed with base-frame position overlay and Y guide.
  Window 2 — Coverage grid (red = uncovered, green = covered).

Usage:
    python algorithmic/record_motion_map.py left
    python algorithmic/record_motion_map.py right

Prerequisites:
    - Run calibrate.py first (base tags → base_transforms.json).
    - Place the tray in the workspace (used to compute target hover Y).
"""

import sys
import csv
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from algorithmic.calibrate import load_base_transforms

# ── Paths ───────────────────────────────────────────────────
PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"
WAYPOINTS_FILE = Path(__file__).parent.parent / "data_collection" / "waypoints.json"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"
OUTPUT_DIR = Path(__file__).parent

# ── Recording parameters ────────────────────────────────────
HOVER_HEIGHT_M = 0.12       # vertical offset above tray (base-frame Y)
Y_TOLERANCE_M = 0.005       # ±5 mm to count as "valid" for coverage grid
CELL_SIZE_M = 0.005         # 5 mm grid cells
GRID_WIDTH_M = 0.200        # 200 mm across (base X)
GRID_DEPTH_M = 0.300        # 300 mm depth (base Z)
APPROACH_DURATION_S = 4.0   # smooth-move duration

# ── Motor names ─────────────────────────────────────────────
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


# ============================================================
# Hardware helpers
# ============================================================

def load_ports() -> dict:
    with open(PORTS_FILE) as f:
        return json.load(f)


def connect_arm(name: str, port: str) -> SOFollower:
    config = SOFollowerRobotConfig(
        id=name, port=port, use_degrees=True,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SOFollower(config)
    robot.connect()
    return robot


def read_joints_raw(robot: SOFollower) -> dict[str, float]:
    return robot.bus.sync_read("Present_Position", normalize=False)


def write_joints_raw(robot: SOFollower, target: dict[str, float]):
    robot.bus.sync_write("Goal_Position", target, normalize=False)


def load_waypoints() -> dict:
    with open(WAYPOINTS_FILE) as f:
        return json.load(f)


# ============================================================
# Smooth single-arm motion
# ============================================================

def smooth_move_single(
    robot: SOFollower,
    start: dict[str, float],
    end: dict[str, float],
    duration: float,
    fps: int = 30,
):
    """Minimum-jerk interpolation for one arm."""
    interval = 1.0 / fps
    n_steps = max(1, int(duration * fps))
    for step in range(n_steps):
        t0 = time.perf_counter()
        tau = np.clip((step + 1) / n_steps, 0.0, 1.0)
        progress = float(10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        target = {
            name: start[name] + (end[name] - start[name]) * progress
            for name in start
        }
        write_joints_raw(robot, target)
        remaining = interval - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)


# ============================================================
# Coordinate transform helpers
# ============================================================

def cam_to_base(p_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Transform a 3D point from camera frame to base-tag frame."""
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    return (T_base_cam @ p_h)[:3]


# ============================================================
# Coverage grid
# ============================================================

N_CELLS_X = int(GRID_WIDTH_M / CELL_SIZE_M)   # 40
N_CELLS_Z = int(GRID_DEPTH_M / CELL_SIZE_M)   # 60


def pos_to_cell(
    bx: float, bz: float,
    x_min: float, z_min: float,
) -> tuple[int, int]:
    """Convert base-frame (x, z) to grid cell indices."""
    cx = int((bx - x_min) / CELL_SIZE_M)
    cz = int((bz - z_min) / CELL_SIZE_M)
    return cx, cz


def in_grid(cx: int, cz: int) -> bool:
    return 0 <= cx < N_CELLS_X and 0 <= cz < N_CELLS_Z


def render_grid(grid: np.ndarray, cell_px: int = 12) -> np.ndarray:
    """Render coverage grid as a BGR image.  Green = covered, red = empty."""
    n_z, n_x = grid.shape
    # Build a color image: green where covered, red where not
    green = np.array([0, 160, 0], dtype=np.uint8)
    red = np.array([0, 0, 160], dtype=np.uint8)
    colors = np.where(grid[..., None], green, red)  # (n_z, n_x, 3)
    # Scale up to pixel grid
    img = np.repeat(np.repeat(colors, cell_px, axis=0), cell_px, axis=1)
    # Draw thin grid lines
    for i in range(1, n_x):
        img[:, i * cell_px, :] = 50
    for i in range(1, n_z):
        img[i * cell_px, :, :] = 50
    return img


# ============================================================
# Main
# ============================================================

def main():
    # ── Parse args ──────────────────────────────────────────
    if len(sys.argv) != 2 or sys.argv[1] not in ("left", "right"):
        print("Usage: python record_motion_map.py [left | right]")
        sys.exit(1)
    side = sys.argv[1]
    print(f"Recording motion map for {side} arm.\n")

    # ── Connect hardware ────────────────────────────────────
    ports = load_ports()
    print("Connecting camera...")
    camera = RealSenseCamera()
    camera.connect()
    print(f"  Camera connected (S/N: {camera.serial_number})")

    print(f"Connecting {side} arm...")
    robot = connect_arm(side, ports[side])
    print(f"  Connected on {ports[side]}")

    cfg = MarkerConfig()
    detector = ArucoDetector(
        config=cfg,
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )

    # ── Load base transform ─────────────────────────────────
    print("\nLoading base transforms...")
    transforms = load_base_transforms()
    if side not in transforms:
        print(f"  ERROR: No transform for {side} arm. Run calibrate.py first.")
        robot.disconnect()
        camera.disconnect()
        sys.exit(1)
    T_cam_base = transforms[side]
    T_base_cam = np.linalg.inv(T_cam_base)
    print("  Loaded.")

    # ── Move to pre_grasp ───────────────────────────────────
    waypoints = load_waypoints()
    home_wp = waypoints["home"][side]
    pre_grasp_wp = waypoints["pre_grasp"][side]

    print("\nMoving to home...")
    current = read_joints_raw(robot)
    smooth_move_single(robot, current, home_wp, APPROACH_DURATION_S)
    time.sleep(0.5)

    print("Moving to pre_grasp...")
    smooth_move_single(robot, home_wp, pre_grasp_wp, APPROACH_DURATION_S)
    time.sleep(0.5)

    # ── Detect tray → compute target_base_y ─────────────────
    print("\nDetecting tray marker to set hover height...")
    print("  Ensure the tray is in the workspace.\n")
    tray_pos_cam = None
    for attempt in range(60):  # up to 2 seconds
        frame = camera.get_frame()
        poses = detector.detect(frame)
        tray_pos_cam = detector.get_tray_marker_pose(poses, side)
        if tray_pos_cam is not None:
            break
        time.sleep(1 / 30)

    if tray_pos_cam is None:
        tray_id = cfg.tray_left if side == "left" else cfg.tray_right
        print(f"  ERROR: Could not detect tray marker (ID {tray_id}).")
        print(f"         Place the tray so the marker faces the camera.")
        robot.disconnect()
        camera.disconnect()
        sys.exit(1)

    tray_base = cam_to_base(tray_pos_cam, T_base_cam)
    target_base_y = tray_base[1] + HOVER_HEIGHT_M
    print(f"  Tray base-frame Y = {tray_base[1]*1000:.1f} mm")
    print(f"  Hover height      = {HOVER_HEIGHT_M*1000:.0f} mm")
    print(f"  Target base Y     = {target_base_y*1000:.1f} mm")

    # ── Detect gripper → grid center ────────────────────────
    grip_cam = None
    for attempt in range(60):
        frame = camera.get_frame()
        poses = detector.detect(frame)
        grip_cam = detector.get_gripper_pose(poses, side)
        if grip_cam is not None:
            break
        time.sleep(1 / 30)

    if grip_cam is None:
        print("\n  ERROR: Could not detect gripper marker. Ensure it faces the camera.")
        robot.disconnect()
        camera.disconnect()
        sys.exit(1)

    grip_base = cam_to_base(grip_cam, T_base_cam)
    grid_x_min = grip_base[0] - GRID_WIDTH_M / 2
    grid_z_min = grip_base[2] - GRID_DEPTH_M / 2
    print(f"\n  Grid center (base): X={grip_base[0]*1000:.0f}, Z={grip_base[2]*1000:.0f} mm")
    print(f"  Grid X range: [{grid_x_min*1000:.0f}, {(grid_x_min + GRID_WIDTH_M)*1000:.0f}] mm")
    print(f"  Grid Z range: [{grid_z_min*1000:.0f}, {(grid_z_min + GRID_DEPTH_M)*1000:.0f}] mm")

    # ── Disable torque on positioning joints ─────────────────
    print("\nDisabling torque on positioning joints (gripper stays locked open)...")
    print("  You can now move the arm freely.")
    print("  The tray can be removed — it's no longer needed.\n")
    for name in MOTOR_NAMES:
        torque = 1 if name == "gripper" else 0
        robot.bus.sync_write("Torque_Enable", {name: torque}, normalize=False)

    # ── Open CSV ────────────────────────────────────────────
    csv_path = OUTPUT_DIR / f"motion_map_raw_{side}.csv"
    csv_file = open(csv_path, "w", newline="")
    # Metadata comment line
    meta = {
        "arm": side,
        "target_base_y": round(target_base_y, 6),
        "hover_height_m": HOVER_HEIGHT_M,
        "y_tolerance_m": Y_TOLERANCE_M,
        "grid_x_min": round(grid_x_min, 6),
        "grid_z_min": round(grid_z_min, 6),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    csv_file.write(f"# {json.dumps(meta)}\n")
    writer = csv.writer(csv_file)
    headers = [
        "cam_x", "cam_y", "cam_z",
        "base_x", "base_y", "base_z",
    ] + MOTOR_NAMES
    writer.writerow(headers)

    # ── Initialize coverage grid ────────────────────────────
    grid = np.zeros((N_CELLS_Z, N_CELLS_X), dtype=bool)
    total_cells = N_CELLS_X * N_CELLS_Z
    valid_cells = 0
    total_samples = 0

    # ── Recording loop ──────────────────────────────────────
    print("Recording... sweep the arm across the workspace.")
    print("  Keep the marker at the target Y height and facing the camera.")
    print("  Press 'q' or ESC in any window to stop.\n")

    try:
        while True:
            frame = camera.get_frame()
            poses = detector.detect(frame)
            grip_cam = detector.get_gripper_pose(poses, side)

            display = frame.copy()

            if grip_cam is not None:
                p_base = cam_to_base(grip_cam, T_base_cam)
                y_err = p_base[1] - target_base_y
                y_ok = abs(y_err) < Y_TOLERANCE_M

                # Only record points that pass the Y check
                if y_ok:
                    joints = read_joints_raw(robot)
                    row = [
                        f"{grip_cam[0]:.6f}", f"{grip_cam[1]:.6f}", f"{grip_cam[2]:.6f}",
                        f"{p_base[0]:.6f}", f"{p_base[1]:.6f}", f"{p_base[2]:.6f}",
                    ] + [str(int(joints[name])) for name in MOTOR_NAMES]
                    writer.writerow(row)
                    total_samples += 1

                    # Update coverage grid
                    cx, cz = pos_to_cell(p_base[0], p_base[2], grid_x_min, grid_z_min)
                    if in_grid(cx, cz) and not grid[cz, cx]:
                        grid[cz, cx] = True
                        valid_cells += 1

                # Overlay on camera feed
                color = (0, 255, 0) if y_ok else (0, 0, 255)
                cv2.putText(
                    display,
                    f"Base: ({p_base[0]*1000:.0f}, {p_base[1]*1000:.0f}, {p_base[2]*1000:.0f}) mm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
                cv2.putText(
                    display,
                    f"Y err: {y_err*1000:+.1f} mm  (tol: +/-{Y_TOLERANCE_M*1000:.0f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )
            else:
                cv2.putText(
                    display, "MARKER LOST",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            pct = valid_cells / total_cells * 100
            cv2.putText(
                display,
                f"Samples: {total_samples}  |  Coverage: {valid_cells}/{total_cells} ({pct:.1f}%)",
                (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )

            cv2.imshow("Camera", display)

            # Render and show coverage grid
            grid_img = render_grid(grid)
            # Add text header to grid window
            cv2.putText(
                grid_img,
                f"{pct:.1f}% covered",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.imshow("Coverage", grid_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    # ── Clean up ────────────────────────────────────────────
    print(f"\nStopping...")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid cells:   {valid_cells}/{total_cells} ({valid_cells/total_cells*100:.1f}%)")

    csv_file.close()
    print(f"  Saved to {csv_path}")

    cv2.destroyAllWindows()

    # Re-enable torque and disconnect
    robot.bus.sync_write("Torque_Enable", 0, normalize=False)
    robot.config.disable_torque_on_disconnect = False
    robot.disconnect()
    camera.disconnect()
    print("  Disconnected.")


if __name__ == "__main__":
    main()
