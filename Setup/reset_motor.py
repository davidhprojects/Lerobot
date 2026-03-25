#!python
"""
Reset a single STS3215 motor's voltage protection registers to factory defaults,
then assign it the correct ID.

Use this when a motor reports:
  [RxPacketError] Input voltage error!

Usage:
  python reset_motor.py black
  python reset_motor.py white

Edit MOTOR_NAME below to the motor that needs fixing, then run the script.
Connect ONLY that motor's wire before pressing Enter.
"""

import json
import sys
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = ["black", "white"]
PORTS_FILE = Path(__file__).parent.parent / "ports.json"

# --- Set this to the motor you want to fix ---
MOTOR_NAME = "wrist_flex"  # e.g. "shoulder_pan", "elbow_flex", "gripper", etc.

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
    print(f"Usage: python reset_motor.py [{' | '.join(ARMS)}]")
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

print(f"Arm  : {arm_name} on {port}")
print(f"Motor: {MOTOR_NAME}")
input(f"\nConnect ONLY the '{MOTOR_NAME}' motor wire, then press Enter...")

bus = FeetechMotorsBus(port=port, motors=MOTORS)
bus._connect(handshake=False)

# Scan using raw _broadcast_ping — public API filters out motors with error status
print("Scanning for motor...")
found_baudrate, found_id = None, None
for baudrate in bus.model_baudrate_table["sts3215"]:
    bus.set_baudrate(baudrate)
    ids_status, comm = bus._broadcast_ping()
    if bus._is_comm_success(comm) and ids_status:
        found_id = next(iter(ids_status))
        found_baudrate = baudrate
        print(f"Found motor at baudrate={found_baudrate}, id={found_id} (status={ids_status[found_id]:#04x})")
        break

if found_id is None:
    raise RuntimeError("Motor not found on any baudrate. Make sure only this motor is connected.")

bus.set_baudrate(found_baudrate)

# --- Diagnostics: read current register values ---
def read_raw(addr_len, motor_id):
    val, comm, _ = bus._read(*addr_len, motor_id, raise_on_error=False)
    return val if bus._is_comm_success(comm) else None

present_v    = read_raw(ADDR_PRESENT_VOLTAGE, found_id)
min_v_before = read_raw(ADDR_MIN_VOLTAGE, found_id)
max_v_before = read_raw(ADDR_MAX_VOLTAGE, found_id)

print(f"  Present voltage : {present_v * 0.1:.1f} V  (raw={present_v})" if present_v is not None else "  Present voltage : read failed")
print(f"  Min voltage limit: {min_v_before * 0.1:.1f} V  (raw={min_v_before})" if min_v_before is not None else "  Min voltage limit: read failed")
print(f"  Max voltage limit: {max_v_before * 0.1:.1f} V  (raw={max_v_before})" if max_v_before is not None else "  Max voltage limit: read failed")

# Set limits wide enough that no real voltage can trigger the error.
# 0 = 0.0 V (min), 254 = 25.4 V (max) — effectively disabled.
NEW_MIN = 0
NEW_MAX = 254

print()
print("Disabling torque and unlocking EEPROM...")
bus._write(*ADDR_TORQUE_ENABLE, found_id, 0,       raise_on_error=False)
bus._write(*ADDR_LOCK,          found_id, 0,       raise_on_error=False)

print(f"Writing voltage limits: min={NEW_MIN}, max={NEW_MAX}...")
bus._write(*ADDR_MAX_VOLTAGE, found_id, NEW_MAX, raise_on_error=False)
bus._write(*ADDR_MIN_VOLTAGE, found_id, NEW_MIN, raise_on_error=False)

# Read back to confirm the writes actually stuck in EEPROM
min_v_after = read_raw(ADDR_MIN_VOLTAGE, found_id)
max_v_after = read_raw(ADDR_MAX_VOLTAGE, found_id)
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

    print(f"Assigning ID {MOTORS[MOTOR_NAME].id} to '{MOTOR_NAME}'...")
    bus2 = FeetechMotorsBus(port=port, motors=MOTORS)
    bus2.setup_motor(MOTOR_NAME)
    print(f"Done — '{MOTOR_NAME}' is now set to ID {MOTORS[MOTOR_NAME].id}.")
