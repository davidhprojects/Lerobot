"""Quick test: open and close the gripper."""

import time
from pathlib import Path
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

PORT = "COM3"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"

config = SOFollowerRobotConfig(
    id="follower_arm",
    port=PORT,
    use_degrees=True,
    calibration_dir=CALIBRATION_DIR,
)
robot = SOFollower(config)

# Connect bus and load calibration manually to avoid interactive prompt
robot.bus.connect()
if robot.calibration:
    robot.bus.write_calibration(robot.calibration)
robot.configure()

# Read current state
obs = robot.get_observation()
print(f"Current positions: {obs}")

# Send action — keep all joints at current position, just move gripper
action = {key: val for key, val in obs.items()}

# Gripper uses 0-100 range (percentage open)
print("Opening gripper...")
action["gripper.pos"] = 80.0
robot.send_action(action)
time.sleep(1.5)

print("Closing gripper...")
action["gripper.pos"] = 10.0
robot.send_action(action)
time.sleep(1.5)

robot.bus.disable_torque()
robot.bus.disconnect()
print("Done.")
