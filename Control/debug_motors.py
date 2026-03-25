from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

config = SOFollowerRobotConfig(
    id="my_follower_arm",
    port="/dev/tty.usbmodem5A7A0159151",
    use_degrees=True,
)
robot = SOFollower(config)
robot.connect(calibrate=True)

print("Reading positions...")
pos = robot.bus.sync_read("Present_Position", normalize=False)
print(pos)

robot.bus.disable_torque()
robot.disconnect()
