"""
Record follower arm motion (by moving it manually), then replay it back.

Usage:
  python record_and_replay_follower.py

- Press ENTER to start recording
- Move the follower arm around by hand
- Press ENTER to stop recording
- The arm will replay the recorded motion
"""

import time
import json
import threading
from pathlib import Path

from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

PORT = "/dev/tty.usbmodem5A7A0159151"
RECORD_FPS = 30


def main():
    config = SOFollowerRobotConfig(
        id="my_follower_arm",
        port=PORT,
        use_degrees=True,
    )
    robot = SOFollower(config)

    # Connect bus, load calibration, configure motors (avoids interactive prompt)
    robot.bus.connect()
    if robot.calibration:
        robot.bus.write_calibration(robot.calibration)
    robot.configure()

    # Disable torque so you can move the arm by hand
    robot.bus.disable_torque()

    print(f"Connected to follower arm on {PORT}")
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
    save_path = Path(__file__).parent / "recorded_motion_follower.json"
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
