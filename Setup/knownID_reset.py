#!python
"""
Fix the voltage protection registers on a motor that is already ID-assigned
and connected alongside all other motors.

Use this instead of reset_motor.py when:
  - The motor's ID is already set correctly
  - All motors can stay connected (no need to isolate)
  - You get: [RxPacketError] Input voltage error!

Usage:
  python fix_voltage.py black
  python fix_voltage.py white

Edit MOTOR_NAME below to the motor that needs fixing, then run the script.
"""

import json
import sys
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = ["black", "white"]
PORTS_FILE = Path(__file__).parent.parent / "ports.json"

# --- Set this to the motor you want to fix ---
MOTOR_NAME = "gripper"  # e.g. "shoulder_pan", "elbow_flex", "gripper", etc.

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

# STS3215 EEPROM addresses (addr, length)
ADDR_TORQUE_ENABLE   = (40, 1)
ADDR_LOCK            = (55, 1)
ADDR_MAX_VOLTAGE     = (14, 1)
ADDR_MIN_VOLTAGE     = (15, 1)
ADDR_PRESENT_VOLTAGE = (62, 1)

if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
    print(f"Usage: python fix_voltage.py [{' | '.join(ARMS)}]")
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

motor_id = MOTORS[MOTOR_NAME].id
print(f"Arm  : {arm_name} on {port}")
print(f"Motor: {MOTOR_NAME} (ID {motor_id})")
print("All other motors can stay connected.")
input("\nPress Enter to connect and fix voltage registers...")

bus = FeetechMotorsBus(port=port, motors=MOTORS)
bus._connect(handshake=False)
bus.set_baudrate(1000000)

def read_raw(addr_len, mid):
    val, comm, _ = bus._read(*addr_len, mid, raise_on_error=False)
    return val if bus._is_comm_success(comm) else None

present_v    = read_raw(ADDR_PRESENT_VOLTAGE, motor_id)
min_v_before = read_raw(ADDR_MIN_VOLTAGE, motor_id)
max_v_before = read_raw(ADDR_MAX_VOLTAGE, motor_id)

print(f"  Present voltage : {present_v * 0.1:.1f} V  (raw={present_v})" if present_v is not None else "  Present voltage : read failed")
print(f"  Min voltage limit: {min_v_before * 0.1:.1f} V  (raw={min_v_before})" if min_v_before is not None else "  Min voltage limit: read failed")
print(f"  Max voltage limit: {max_v_before * 0.1:.1f} V  (raw={max_v_before})" if max_v_before is not None else "  Max voltage limit: read failed")

# Set limits wide enough that no real voltage can trigger the error.
# 0 = 0.0 V (min), 254 = 25.4 V (max) — effectively disabled.
NEW_MIN = 40
NEW_MAX = 150

print()
print("Disabling torque and unlocking EEPROM...")
bus._write(*ADDR_TORQUE_ENABLE, motor_id, 0,       raise_on_error=False)
bus._write(*ADDR_LOCK,          motor_id, 0,       raise_on_error=False)

print(f"Writing voltage limits: min={NEW_MIN}, max={NEW_MAX}...")
bus._write(*ADDR_MAX_VOLTAGE, motor_id, NEW_MAX, raise_on_error=False)
bus._write(*ADDR_MIN_VOLTAGE, motor_id, NEW_MIN, raise_on_error=False)

min_v_after = read_raw(ADDR_MIN_VOLTAGE, motor_id)
max_v_after = read_raw(ADDR_MAX_VOLTAGE, motor_id)
print(f"  Read-back min: {min_v_after}  (expected {NEW_MIN}) — {'OK' if min_v_after == NEW_MIN else 'WRITE FAILED'}")
print(f"  Read-back max: {max_v_after}  (expected {NEW_MAX}) — {'OK' if max_v_after == NEW_MAX else 'WRITE FAILED'}")

bus.port_handler.closePort()
print()

if min_v_after != NEW_MIN or max_v_after != NEW_MAX:
    print("WARNING: EEPROM writes did not stick. The motor may have a hardware fault.")
    print("Check the present voltage above — if it's outside 6.0–8.5 V the power supply is the issue.")
else:
    print("Voltage limits written successfully.")
    print("Power-cycle the motor now (unplug and reconnect its cable), then press Enter...")
    input()
    print(f"Done — '{MOTOR_NAME}' should be working. Run calibrate.py if needed.")
