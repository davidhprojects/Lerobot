"""
Record leader arm motion, then replay it back through the same arm.

Usage:
  python record_and_replay.py

- Press ENTER to start recording
- Move the leader arm around
- Press ENTER to stop recording
- The arm will replay the recorded motion
"""

import time
import json
import threading
from pathlib import Path

from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig, SO101Leader

PORT = "/dev/tty.usbmodem5A7A0156821"
RECORD_FPS = 30  # recording frequency in Hz


def main():
    config = SOLeaderTeleopConfig(
        id="my_leader_arm",
        port=PORT,
        use_degrees=True,
    )
    teleop = SO101Leader(config)
    teleop.connect(calibrate=False)

    print(f"Connected to leader arm on {PORT}")
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
        # Read raw (non-normalized) positions to avoid calibration issues
        raw = teleop.bus.sync_read("Present_Position", normalize=False)
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

    # Disable torque and reset position limits to full range
    for motor in teleop.bus.motors:
        try:
            teleop.bus.write("Torque_Enable", motor, 0)
            teleop.bus.write("Lock", motor, 0)
            teleop.bus.write("Min_Position_Limit", motor, 0, normalize=False)
            teleop.bus.write("Max_Position_Limit", motor, 4095, normalize=False)
        except RuntimeError:
            pass

    time.sleep(0.3)

    # Enable torque on all motors
    for motor in teleop.bus.motors:
        try:
            teleop.bus.write("Torque_Enable", motor, 1)
            teleop.bus.write("Lock", motor, 1)
        except RuntimeError as e:
            print(f"Warning: could not enable torque on {motor}: {e}")

    print("Replaying...")

    for frame in frames:
        start = time.perf_counter()
        teleop.bus.sync_write("Goal_Position", frame, normalize=False)
        elapsed = time.perf_counter() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Replay complete!")

    # Release motors
    for motor in teleop.bus.motors:
        try:
            teleop.bus.write("Torque_Enable", motor, 0)
        except RuntimeError:
            pass

    teleop.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
