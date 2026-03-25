#!python
"""
Record arm motion (by moving it manually), then replay it back.

Usage:
  python record_and_replay.py black
  python record_and_replay.py white

- Press ENTER to start recording
- Move the arm around by hand
- Press ENTER to stop recording
- The arm will replay the recorded motion
"""

import sys
import time
import json
import threading
from pathlib import Path

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

ARMS = ["black", "white"]
PORTS_FILE = Path(__file__).parent.parent / "ports.json"
RECORD_FPS = 30
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        print(f"Usage: python record_and_replay.py [{' | '.join(ARMS)}]")
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

    port = ports[arm_name]

    config = SOFollowerRobotConfig(
        id=arm_name,
        port=port,
        use_degrees=True,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = SOFollower(config)

    # Connect bus, load calibration, configure motors (avoids interactive prompt)
    robot.bus.connect()
    if robot.calibration:
        robot.bus.write_calibration(robot.calibration)
    robot.configure()

    # Disable torque so you can move the arm by hand
    robot.bus.disable_torque()

    print(f"Connected to {arm_name} on {port}")
    print("All motors should be free to move now.")
    print()

    # --- Record ---
    input("Press ENTER to start recording...")
    print(f"Recording at {RECORD_FPS} FPS. Move the arm around!")
    print("Press ENTER to stop recording...\n")

    frames = []
    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    interval = 1.0 / RECORD_FPS
    while not stop_event.is_set():
        start = time.perf_counter()
        raw = robot.bus.sync_read("Present_Position", normalize=False)
        frames.append(raw)
        elapsed = time.perf_counter() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Recorded {len(frames)} frames ({len(frames)/RECORD_FPS:.1f}s)")

    # Save to file
    save_path = Path(__file__).parent / "recorded_motion.json"
    with open(save_path, "w") as f:
        json.dump({"fps": RECORD_FPS, "frames": frames}, f)
    print(f"Saved to {save_path}")

    # --- Replay ---
    input("\nPress ENTER to replay the recorded motion...")

    # Re-configure and enable torque for replay
    robot.configure()
    time.sleep(0.3)
    print("Replaying...")

    for frame in frames:
        start = time.perf_counter()
        robot.bus.sync_write("Goal_Position", frame, normalize=False)
        elapsed = time.perf_counter() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Replay complete!")

    # Release motors
    robot.bus.disable_torque()
    robot.bus.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
