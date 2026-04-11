#!python
"""
Teleoperate right (follower) using left (leader) with relative positions.

Left's motion delta from its starting pose is applied to right's starting
pose — so right never jumps on startup regardless of calibration differences.

Both arms should be in their intended starting position before running.

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

for arm in ["right", "left"]:
    if arm not in ports:
        print(f"No port found for '{arm}' in ports.json. Run find_ports.py first.")
        sys.exit(1)

robot_config = SO101FollowerConfig(
    port=ports["right"],
    id="right",
    calibration_dir=CALIBRATION_DIR,
)

teleop_config = SO101LeaderConfig(
    port=ports["left"],
    id="left",
    calibration_dir=CALIBRATION_DIR,
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()
teleop_device.bus.disable_torque()  # ensure left moves freely by hand

# Capture starting positions for relative control.
# Both arms should be in their intended start pose at this point.
left_origin = teleop_device.get_action()
_right_raw = robot.bus.sync_read("Present_Position", normalize=True)
# get_action() keys have a '.pos' suffix (e.g. 'shoulder_pan.pos'); remap to match
right_origin = {k + ".pos": v for k, v in _right_raw.items()}

print("Teleoperation started (relative mode). Press Ctrl+C to stop.")

try:
    while True:
        left_current = teleop_device.get_action()
        delta = {k: left_current[k] - left_origin[k] for k in left_current}
        action = {k: right_origin[k] + delta[k] for k in delta}
        robot.send_action(action)
finally:
    robot.bus.disable_torque()
    robot.bus.disconnect()
    teleop_device.bus.disconnect()
    print("Disconnected.")
