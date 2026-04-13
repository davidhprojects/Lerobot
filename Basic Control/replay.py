#!python
"""
Replay a previously recorded arm motion from recorded_motion.json.

Usage:
  python replay.py right
  python replay.py left
"""

import sys
import time
import json
from pathlib import Path

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

ARMS = ["right", "left"]
PORTS_FILE = Path(__file__).parent.parent / "Setup" / "ports.json"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"
MOTION_FILE = Path(__file__).parent / "recorded_motion.json"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        print(f"Usage: python replay.py [{' | '.join(ARMS)}]")
        sys.exit(1)

    arm_name = sys.argv[1]

    if not PORTS_FILE.exists():
        print(f"Ports file not found at {PORTS_FILE}. Run find_ports.py first.")
        sys.exit(1)

    with open(PORTS_FILE) as f:
        ports = json.load(f)

    if arm_name not in ports:
        print(f"No port found for '{arm_name}' in {PORTS_FILE}. Run find_ports.py first.")
        sys.exit(1)

    if not MOTION_FILE.exists():
        print(f"No recorded motion found at {MOTION_FILE}. Run record_and_replay.py first.")
        sys.exit(1)

    with open(MOTION_FILE) as f:
        data = json.load(f)

    frames = data["frames"]
    fps = data["fps"]
    interval = 1.0 / fps

    print(f"Loaded {len(frames)} frames ({len(frames)/fps:.1f}s) at {fps} FPS")

    port = ports[arm_name]

    config = SOFollowerRobotConfig(
        id=arm_name,
        port=port,
        use_degrees=True,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SOFollower(config)

    robot.bus.connect()
    if robot.calibration:
        robot.bus.write_calibration(robot.calibration)
    robot.configure()

    print(f"Connected to {arm_name} on {port}")
    input("Press ENTER to start replay...")

    print("Replaying...")
    for frame in frames:
        start = time.perf_counter()
        robot.bus.sync_write("Goal_Position", frame, normalize=False)
        elapsed = time.perf_counter() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Replay complete!")

    robot.bus.disable_torque()
    robot.bus.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
