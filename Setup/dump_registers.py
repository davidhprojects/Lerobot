#!python
"""
Read every register byte (addresses 0-90) from a single motor on both
arms and print a side-by-side comparison highlighting differences.

Usage:
  python dump_registers.py
  python dump_registers.py elbow_flex

Defaults to shoulder_pan if no motor name is given.
"""

import json
import sys
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORTS_FILE = Path(__file__).parent / "ports.json"

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

# Known register names for annotation (address -> (size, name))
# Registers that span 2 bytes: the name is on the low byte, high byte shows "^"
REG_NAMES = {
    0:  "Firmware_Major",
    1:  "Firmware_Minor",
    3:  "Model_L",
    4:  "Model_H",
    5:  "ID",
    6:  "Baud_Rate",
    7:  "Return_Delay_Time",
    8:  "Response_Status_Level",
    9:  "Min_Position_Limit_L",
    10: "Min_Position_Limit_H",
    11: "Max_Position_Limit_L",
    12: "Max_Position_Limit_H",
    13: "Max_Temperature_Limit",
    14: "Max_Voltage_Limit",
    15: "Min_Voltage_Limit",
    16: "Max_Torque_Limit_L",
    17: "Max_Torque_Limit_H",
    18: "Phase",
    19: "Unloading_Condition",
    20: "LED_Alarm_Condition",
    21: "P_Coefficient",
    22: "D_Coefficient",
    23: "I_Coefficient",
    24: "Minimum_Startup_Force_L",
    25: "Minimum_Startup_Force_H",
    26: "CW_Dead_Zone",
    27: "CCW_Dead_Zone",
    28: "Protection_Current_L",
    29: "Protection_Current_H",
    30: "Angular_Resolution",
    31: "Homing_Offset_L",
    32: "Homing_Offset_H",
    33: "Operating_Mode",
    34: "Protective_Torque",
    35: "Protection_Time",
    36: "Overload_Torque",
    37: "Velocity_P",
    38: "Over_Current_Prot_Time",
    39: "Velocity_I",
    40: "Torque_Enable",
    41: "Acceleration",
    42: "Goal_Position_L",
    43: "Goal_Position_H",
    44: "Goal_Time_L",
    45: "Goal_Time_H",
    46: "Goal_Velocity_L",
    47: "Goal_Velocity_H",
    48: "Torque_Limit_L",
    49: "Torque_Limit_H",
    55: "Lock",
    56: "Present_Position_L",
    57: "Present_Position_H",
    58: "Present_Speed_L",
    59: "Present_Speed_H",
    60: "Present_Load_L",
    61: "Present_Load_H",
    62: "Present_Voltage",
    63: "Present_Temperature",
    65: "Error_Status",
    66: "Moving",
    69: "Present_Current_L",
    70: "Present_Current_H",
    80: "Moving_Vel_Threshold",
    81: "DTs",
    82: "Velocity_Unit_factor",
    83: "Hts",
    84: "Max_Velocity_Limit",
    85: "Maximum_Acceleration",
    86: "Accel_Multiplier",
}

# Registers that are runtime/volatile — differences are expected
VOLATILE = {42, 43, 44, 45, 46, 47, 48, 49, 56, 57, 58, 59, 60, 61,
            62, 63, 65, 66, 69, 70}

MOTOR_NAME = sys.argv[1] if len(sys.argv) > 1 else "shoulder_pan"
if MOTOR_NAME not in MOTORS:
    print(f"Unknown motor '{MOTOR_NAME}'. Choose from: {list(MOTORS)}")
    sys.exit(1)

with open(PORTS_FILE) as f:
    ports = json.load(f)

motor_id = MOTORS[MOTOR_NAME].id
MAX_ADDR = 90


def read_all(port_name):
    """Read every byte from addr 0..MAX_ADDR."""
    bus = FeetechMotorsBus(port=ports[port_name], motors=MOTORS)
    bus._connect(handshake=False)
    bus.set_baudrate(1000000)

    values = {}
    for addr in range(MAX_ADDR + 1):
        val, comm, _ = bus._read(addr, 1, motor_id, raise_on_error=False)
        values[addr] = val if bus._is_comm_success(comm) else None

    bus.port_handler.closePort()
    return values


print(f"Motor: {MOTOR_NAME} (ID {motor_id})")
print(f"Reading right arm ({ports['right']})...")
right = read_all("right")
print(f"Reading left arm ({ports['left']})...")
left = read_all("left")

# Compare
print()
print(f"{'Addr':>4}  {'Register':<25} {'Right':>5} {'Left':>5}  Notes")
print("-" * 70)

diffs = 0
for addr in range(MAX_ADDR + 1):
    r = right[addr]
    l = left[addr]

    name = REG_NAMES.get(addr, "")
    r_str = f"{r:>5}" if r is not None else "  ERR"
    l_str = f"{l:>5}" if l is not None else "  ERR"

    # Determine if this is a meaningful difference
    both_ok = r is not None and l is not None
    different = both_ok and r != l
    volatile = addr in VOLATILE

    if different and not volatile:
        flag = " <-- DIFFERENT"
        diffs += 1
    elif different and volatile:
        flag = " (volatile)"
    elif not both_ok:
        flag = " (read error)"
    else:
        flag = ""

    # Only print rows that have a name or a difference
    if name or different or not both_ok:
        print(f"{addr:>4}  {name:<25} {r_str} {l_str}{flag}")

print()
print(f"Non-volatile differences: {diffs}")
if diffs == 0:
    print("All configuration registers match between arms.")
    print("The issue is likely internal motor state (encoder calibration)")
    print("that isn't accessible through the register interface.")
else:
    print("Differences found — these could be causing the issue.")
