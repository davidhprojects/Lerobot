"""
Assign an ID to a single motor without going through all 6.

Motor name → ID mapping for SO-100 follower:
  shoulder_pan  → 1
  shoulder_lift → 2
  elbow_flex    → 3
  wrist_flex    → 4
  wrist_roll    → 5
  gripper       → 6

Usage:
  Edit MOTOR_NAME below, then run the script.
  Connect ONLY that motor's wire before pressing Enter.
"""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "COM3"

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

if MOTOR_NAME not in MOTORS:
    raise ValueError(f"Unknown motor '{MOTOR_NAME}'. Choose from: {list(MOTORS)}")

target_id = MOTORS[MOTOR_NAME].id
print(f"Target motor : {MOTOR_NAME} (ID {target_id})")
print()
input(f"Connect ONLY the '{MOTOR_NAME}' motor wire to the controller board, then press Enter...")

bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
bus.setup_motor(MOTOR_NAME)
print(f"Done — '{MOTOR_NAME}' is now set to ID {target_id}.")
