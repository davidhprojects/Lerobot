"""
Assign an ID to a single motor without going through all 6.

Usage:
  python assign_single_motor_id.py black
  python assign_single_motor_id.py white

Motor name → ID mapping:
  shoulder_pan  → 1
  shoulder_lift → 2
  elbow_flex    → 3
  wrist_flex    → 4
  wrist_roll    → 5
  gripper       → 6

Edit MOTOR_NAME below, then run the script.
Connect ONLY that motor's wire before pressing Enter.
"""

import json
import sys
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = ["black", "white"]
PORTS_FILE = Path(__file__).parent.parent / "ports.json"

# --- Set this to the motor you want to assign ---
MOTOR_NAME = "wrist_flex"  # e.g. "shoulder_pan", "elbow_flex", "gripper", etc.

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
    print(f"Usage: python assign_single_motor_id.py [{' | '.join(ARMS)}]")
    sys.exit(1)

arm_name = sys.argv[1]

if not PORTS_FILE.exists():
    print(f"ports.json not found at {PORTS_FILE}. Run find_ports.py first.")
    sys.exit(1)

with open(PORTS_FILE) as f:
    ports = json.load(f)

if arm_name not in ports:
    print(f"No port found for '{arm_name}' in ports.json. Run find_ports.py first.")
    sys.exit(1)

port = ports[arm_name]

if MOTOR_NAME not in MOTORS:
    raise ValueError(f"Unknown motor '{MOTOR_NAME}'. Choose from: {list(MOTORS)}")

target_id = MOTORS[MOTOR_NAME].id
print(f"Arm     : {arm_name} on {port}")
print(f"Motor   : {MOTOR_NAME} (ID {target_id})")
print()
input(f"Connect ONLY the '{MOTOR_NAME}' motor wire to the controller board, then press Enter...")

bus = FeetechMotorsBus(port=port, motors=MOTORS)
bus.setup_motor(MOTOR_NAME)
print(f"Done — '{MOTOR_NAME}' is now set to ID {target_id}.")
