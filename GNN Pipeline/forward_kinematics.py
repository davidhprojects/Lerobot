"""
Forward kinematics for the SO-101 robot arm.

Computes end-effector (gripper frame) pose from joint angles using a chain of
homogeneous transforms derived from the official SO-101 URDF
(TheRobotStudio/SO-ARM100 — so101_new_calib.urdf).

Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
Gripper open/close does not affect EE pose and is excluded.

All measurements in meters and radians.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: fixed-axis X-Y-Z (equivalently intrinsic Z-Y-X)."""
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _tf(xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
        joint_angle: float = 0.0) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from a URDF joint origin (xyz, rpy)
    followed by a rotation of ``joint_angle`` around the local Z axis.
    """
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(*rpy)
    T[:3, 3] = xyz

    R = np.eye(4)
    R[:3, :3] = _rot_z(joint_angle)
    return T @ R


# measured parameters
@dataclass
class SO101Params:
    """
    We measured all of these parameters and referenced them against the URDF.
    """

    # Joint 1 — shoulder_pan  (base_link → shoulder_link)
    shoulder_pan_xyz: tuple = (0.03103, 0.0, 0.11935)
    shoulder_pan_rpy: tuple = (3.14159, 0.0, -3.14159)

    # Joint 2 — shoulder_lift  (shoulder_link → upper_arm_link)
    shoulder_lift_xyz: tuple = (-0.03052, 0.0, -0.0734)
    shoulder_lift_rpy: tuple = (-1.5708, -1.5708, 0.0)

    # Joint 3 — elbow_flex  (upper_arm_link → lower_arm_link)
    elbow_flex_xyz: tuple = (-0.11257, -0.028, 0.0)
    elbow_flex_rpy: tuple = (0.0, 0.0, 1.5708)

    # Joint 4 — wrist_flex  (lower_arm_link → wrist_link)
    wrist_flex_xyz: tuple = (-0.1349, 0.0052, 0.0)
    wrist_flex_rpy: tuple = (0.0, 0.0, -1.5708)

    # Joint 5 — wrist_roll  (wrist_link → gripper_link)
    wrist_roll_xyz: tuple = (0.0, -0.05745, 0.0)
    wrist_roll_rpy: tuple = (1.5708, 0.0487, 3.14159)

    # Fixed — gripper_frame  (gripper_link → gripper_frame_link / TCP)
    gripper_frame_xyz: tuple = (0.0, 0.0, -0.10795)
    gripper_frame_rpy: tuple = (0.0, 3.14159, 0.0)


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"

# STS3215 motor resolution (12-bit: 0–4095)
_MOTOR_RESOLUTION = 4096


def load_joint_limits(arm_name: str) -> dict[str, tuple[float, float]]:
    """
    Load joint limits (in degrees) from a calibration JSON file.

    The calibration stores raw encoder range_min/range_max.  The lerobot
    DEGREES normalization is::

        angle_deg = (raw - mid) * 360 / resolution

    where mid = (range_min + range_max) / 2.  Applying that formula to
    range_min and range_max themselves gives the symmetric limits.
    """
    cal_path = CALIBRATION_DIR / f"{arm_name}.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {cal_path}. Run calibrate.py first."
        )

    with open(cal_path) as f:
        cal = json.load(f)

    limits: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        entry = cal[name]
        rmin = entry["range_min"]
        rmax = entry["range_max"]
        mid = (rmin + rmax) / 2
        lo_deg = (rmin - mid) * 360 / _MOTOR_RESOLUTION
        hi_deg = (rmax - mid) * 360 / _MOTOR_RESOLUTION
        limits[name] = (lo_deg, hi_deg)

    return limits


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(
    joint_angles: np.ndarray,
    params: SO101Params | None = None,
) -> np.ndarray:
    """
    Compute the 4x4 end-effector (gripper frame) pose in the base frame.

    Parameters
    ----------
    joint_angles : array-like, shape (5,)
        Joint angles in **radians**:
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    params : SO101Params, optional
        Arm geometry.  Defaults to standard URDF values.

    Returns
    -------
    T : ndarray, shape (4, 4)
        Homogeneous transform from base_link to gripper_frame_link.
    """
    if params is None:
        params = SO101Params()

    q = np.asarray(joint_angles, dtype=float)
    assert q.shape == (5,), f"Expected 5 joint angles, got {q.shape}"

    T = np.eye(4)
    T = T @ _tf(params.shoulder_pan_xyz,  params.shoulder_pan_rpy,  q[0])
    T = T @ _tf(params.shoulder_lift_xyz, params.shoulder_lift_rpy, q[1])
    T = T @ _tf(params.elbow_flex_xyz,    params.elbow_flex_rpy,    q[2])
    T = T @ _tf(params.wrist_flex_xyz,    params.wrist_flex_rpy,    q[3])
    T = T @ _tf(params.wrist_roll_xyz,    params.wrist_roll_rpy,    q[4])
    # Fixed transform to the tool center point (gripper frame)
    T = T @ _tf(params.gripper_frame_xyz, params.gripper_frame_rpy)

    return T


def forward_kinematics_deg(
    joint_angles_deg: np.ndarray,
    params: SO101Params | None = None,
) -> np.ndarray:
    """Same as ``forward_kinematics`` but accepts angles in degrees."""
    return forward_kinematics(np.deg2rad(joint_angles_deg), params)


def ee_position(joint_angles: np.ndarray, params: SO101Params | None = None) -> np.ndarray:
    """Return just the (x, y, z) end-effector position in meters."""
    return forward_kinematics(joint_angles, params)[:3, 3]


def ee_pose_6d(joint_angles: np.ndarray, params: SO101Params | None = None) -> np.ndarray:
    """
    Return (x, y, z, roll, pitch, yaw) of the end-effector.

    Euler angles use the ZYX (yaw-pitch-roll) convention.
    """
    T = forward_kinematics(joint_angles, params)
    pos = T[:3, 3]
    R = T[:3, :3]

    # Extract ZYX Euler angles
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Live test — read joint angles from a physical arm
# ---------------------------------------------------------------------------

ARMS = ["right", "left"]
PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"


def _read_joint_angles(robot) -> np.ndarray:
    """Read the 5 FK joint angles (degrees) from a connected SOFollower."""
    raw = robot.bus.sync_read("Present_Position", normalize=True)
    return np.array([raw[name] for name in JOINT_NAMES])


def _connect_arm(arm_name, ports):
    from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

    config = SOFollowerRobotConfig(
        id=arm_name,
        port=ports[arm_name],
        use_degrees=True,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SOFollower(config)
    robot.bus.connect()
    if robot.calibration:
        robot.bus.write_calibration(robot.calibration)
    robot.configure()
    robot.bus.disable_torque()
    return robot


def _run_live(robot):
    """Live FK display: shows EE position while you move the arm."""
    import time
    try:
        while True:
            angles_deg = _read_joint_angles(robot)
            angles_rad = np.deg2rad(angles_deg)
            pose = ee_pose_6d(angles_rad)

            print(
                f"\rPos: x={pose[0]*1000:7.1f}  y={pose[1]*1000:7.1f}  z={pose[2]*1000:7.1f} mm  |  "
                f"Rot: r={np.degrees(pose[3]):6.1f}  p={np.degrees(pose[4]):6.1f}  y={np.degrees(pose[5]):6.1f} deg",
                end="", flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")


def main():
    import sys

    usage = f"Usage: python forward_kinematics.py [{' | '.join(ARMS)}]"

    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        print(usage)
        sys.exit(1)

    arm_name = sys.argv[1]

    if not PORTS_FILE.exists():
        print(f"ports.json not found at {PORTS_FILE}. Run find_ports.py first.")
        sys.exit(1)

    with open(PORTS_FILE) as f:
        ports = json.load(f)

    if arm_name not in ports:
        print(f"No port found for '{arm_name}' in ports.json. Run find_ports.py first.")
        sys.exit(1)

    # Load joint limits from this arm's calibration file
    limits = load_joint_limits(arm_name)
    print(f"Joint limits for {arm_name} (degrees):")
    for name, (lo, hi) in limits.items():
        print(f"  {name:16s}  {lo:+7.1f}  to  {hi:+7.1f}")
    print()

    robot = _connect_arm(arm_name, ports)
    print(f"Connected to {arm_name} on {ports[arm_name]}")
    print("Move the arm around — FK output updates live. Press Ctrl+C to stop.\n")
    _run_live(robot)

    robot.bus.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
