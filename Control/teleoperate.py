from pathlib import Path
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"

robot_config = SO101FollowerConfig(
    port="COM3",
    id="follower_arm",
    calibration_dir=CALIBRATION_DIR,
)

teleop_config = SO101LeaderConfig(
    port="COM2",  # TODO: confirm correct COM port for leader arm
    id="leader_arm",
    calibration_dir=CALIBRATION_DIR,
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)