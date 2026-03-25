"""Quick test: open and close the gripper."""

import time
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

PORT = "/dev/tty.usbmodem5A7A0159151"

config = SOFollowerRobotConfig(
    id="my_follower_arm",
    port=PORT,
    use_degrees=True,
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
