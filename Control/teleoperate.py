#!python
"""
Teleoperate black (follower) using white (leader).

Ports are loaded automatically from ports.json — run find_ports.py first
if you haven't already.
"""

import json
import sys
from pathlib import Path

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

PORTS_FILE = Path(__file__).parent.parent / "ports.json"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"

if not PORTS_FILE.exists():
    print(f"ports.json not found at {PORTS_FILE}. Run find_ports.py first.")
    sys.exit(1)

with open(PORTS_FILE) as f:
    ports = json.load(f)

for arm in ["black", "white"]:
    if arm not in ports:
        print(f"No port found for '{arm}' in ports.json. Run find_ports.py first.")
        sys.exit(1)

robot_config = SO101FollowerConfig(
    port=ports["black"],
    id="black",
    calibration_dir=CALIBRATION_DIR,
)

teleop_config = SO101LeaderConfig(
    port=ports["white"],
    id="white",
    calibration_dir=CALIBRATION_DIR,
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)
