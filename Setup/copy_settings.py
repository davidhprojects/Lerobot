#!python
"""
Copy all motor register settings from one arm to another.

Reads every writable register from each motor on the source arm and
writes those exact values to the corresponding motor on the destination
arm.  Calibration registers (Homing_Offset, Min/Max_Position_Limit) are
copied too — you will need to re-run calibrate.py on the destination arm
afterward.

Usage:
  python copy_settings.py right left
  python copy_settings.py left right
"""

import json
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = ["right", "left"]
PORTS_FILE = Path(__file__).parent / "ports.json"

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Every writable register we want to clone.
# Format: (address, byte_length, register_name)
# ID (5) and Baud_Rate (6) are omitted — both arms already use IDs 1-6
# at 1 Mbps.  Goal_Position (42) is omitted — it's runtime state, not
# configuration.
REGISTERS = [
    # EEPROM
    (7,  1, "Return_Delay_Time"),
    (8,  1, "Response_Status_Level"),
    (9,  2, "Min_Position_Limit"),
    (11, 2, "Max_Position_Limit"),
    (13, 1, "Max_Temperature_Limit"),
    (14, 1, "Max_Voltage_Limit"),
    (15, 1, "Min_Voltage_Limit"),
    (16, 2, "Max_Torque_Limit"),
    (18, 1, "Phase"),
    (19, 1, "Unloading_Condition"),
    (20, 1, "LED_Alarm_Condition"),
    (21, 1, "P_Coefficient"),
    (22, 1, "D_Coefficient"),
    (23, 1, "I_Coefficient"),
    (24, 2, "Minimum_Startup_Force"),
    (26, 1, "CW_Dead_Zone"),
    (27, 1, "CCW_Dead_Zone"),
    (28, 2, "Protection_Current"),
    (30, 1, "Angular_Resolution"),
    (31, 2, "Homing_Offset"),
    (33, 1, "Operating_Mode"),
    (34, 1, "Protective_Torque"),
    (35, 1, "Protection_Time"),
    (36, 1, "Overload_Torque"),
    (37, 1, "Velocity_closed_loop_P"),
    (38, 1, "Over_Current_Protection_Time"),
    (39, 1, "Velocity_closed_loop_I"),
    # SRAM
    (40, 1, "Torque_Enable"),
    (41, 1, "Acceleration"),
    (44, 2, "Goal_Time"),
    (46, 2, "Goal_Velocity"),
    (48, 2, "Torque_Limit"),
    (55, 1, "Lock"),
    # Factory section
    (80, 1, "Moving_Velocity_Threshold"),
    (81, 1, "DTs"),
    (82, 1, "Velocity_Unit_factor"),
    (83, 1, "Hts"),
    (84, 1, "Maximum_Velocity_Limit"),
    (85, 1, "Maximum_Acceleration"),
    (86, 1, "Acceleration_Multiplier"),
]


def read_raw(bus, addr, length, motor_id):
    val, comm, _ = bus._read(addr, length, motor_id, raise_on_error=False)
    return val if bus._is_comm_success(comm) else None


def write_raw(bus, addr, length, motor_id, value):
    bus._write(addr, length, motor_id, value, raise_on_error=False)


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
if len(sys.argv) != 3 or sys.argv[1] not in ARMS or sys.argv[2] not in ARMS:
    print(f"Usage: python copy_settings.py <source> <dest>")
    print(f"  e.g. python copy_settings.py right left")
    sys.exit(1)

src_arm, dst_arm = sys.argv[1], sys.argv[2]

if src_arm == dst_arm:
    print("Source and destination must be different arms.")
    sys.exit(1)

if not PORTS_FILE.exists():
    print(f"ports.json not found at {PORTS_FILE}. Run find_ports.py first.")
    sys.exit(1)

with open(PORTS_FILE) as f:
    ports = json.load(f)

for arm in (src_arm, dst_arm):
    if arm not in ports:
        print(f"No port for '{arm}' in ports.json. Run find_ports.py first.")
        sys.exit(1)

print(f"Source : {src_arm} arm on {ports[src_arm]}")
print(f"Dest   : {dst_arm} arm on {ports[dst_arm]}")
print()
print(f"This will overwrite ALL settings on the {dst_arm} arm with values")
print(f"read from the {src_arm} arm.  You will need to re-run calibrate.py")
print(f"on the {dst_arm} arm afterward.")
input("\nPress Enter to continue (Ctrl-C to abort)...")

# ---------------------------------------------------------------------------
# Phase 1: Read all register values from source arm
# ---------------------------------------------------------------------------
print(f"\n--- Reading from {src_arm} arm ---")

src_bus = FeetechMotorsBus(port=ports[src_arm], motors=MOTORS)
src_bus._connect(handshake=False)
src_bus.set_baudrate(1000000)

# src_values[motor_name] = [(addr, length, name, value), ...]
src_values = {}
src_ok = True

for name in MOTOR_ORDER:
    mid = MOTORS[name].id
    ping = read_raw(src_bus, 62, 1, mid)
    if ping is None:
        print(f"  {name} (ID {mid}): NOT RESPONDING — aborting")
        src_ok = False
        break

    values = []
    for addr, length, reg_name in REGISTERS:
        val = read_raw(src_bus, addr, length, mid)
        if val is None:
            print(f"  {name} (ID {mid}): read failed on {reg_name} — aborting")
            src_ok = False
            break
        values.append((addr, length, reg_name, val))

    if not src_ok:
        break

    src_values[name] = values
    print(f"  {name} (ID {mid}): read {len(values)} registers  "
          f"(voltage = {ping * 0.1:.1f} V)")

src_bus.port_handler.closePort()

if not src_ok:
    print(f"\nFailed to read from {src_arm} arm. Fix the source arm first.")
    sys.exit(1)

print(f"\nAll {len(MOTOR_ORDER)} motors read successfully.")

# ---------------------------------------------------------------------------
# Phase 2: Write all register values to destination arm
# ---------------------------------------------------------------------------
print(f"\n--- Writing to {dst_arm} arm ---")

dst_bus = FeetechMotorsBus(port=ports[dst_arm], motors=MOTORS)
dst_bus._connect(handshake=False)
dst_bus.set_baudrate(1000000)

total_ok = 0
total_fail = 0
total_changed = 0

for name in MOTOR_ORDER:
    mid = MOTORS[name].id
    print(f"\n{'='*64}")
    print(f"  {name} (ID {mid})")
    print(f"{'='*64}")

    # Ping check
    ping = read_raw(dst_bus, 62, 1, mid)
    if ping is None:
        print(f"  SKIP — motor not responding (check wiring)")
        total_fail += len(REGISTERS)
        continue

    print(f"  Responding (voltage = {ping * 0.1:.1f} V)")

    # Disable torque and unlock EEPROM before writing
    write_raw(dst_bus, 40, 1, mid, 0)  # Torque_Enable = 0
    write_raw(dst_bus, 55, 1, mid, 0)  # Lock = 0
    time.sleep(0.05)

    lock_check = read_raw(dst_bus, 55, 1, mid)
    if lock_check != 0:
        print(f"  ERROR — could not unlock EEPROM (Lock = {lock_check})")
        total_fail += len(REGISTERS)
        continue

    print()
    print(f"  {'Register':<35} {'Src':>6} {'Was':>6} {'Now':>6}  Status")
    print(f"  {'-'*68}")

    ok = 0
    fail = 0
    changed = 0

    for addr, length, reg_name, src_val in src_values[name]:
        old = read_raw(dst_bus, addr, length, mid)
        old_str = str(old) if old is not None else "?"

        write_raw(dst_bus, addr, length, mid, src_val)
        time.sleep(0.01)

        verify = read_raw(dst_bus, addr, length, mid)

        if verify == src_val:
            mark = "*" if old != src_val else " "
            if old != src_val:
                changed += 1
            print(f"  {reg_name:<35} {src_val:>6} {old_str:>6} {verify:>6}  OK {mark}")
            ok += 1
        else:
            verify_str = str(verify) if verify is not None else "FAIL"
            print(f"  {reg_name:<35} {src_val:>6} {old_str:>6} {verify_str:>6}  WRITE FAILED")
            fail += 1

    # Set Goal_Position to current physical position so motor doesn't
    # jump if torque is later enabled
    pos = read_raw(dst_bus, 56, 2, mid)
    if pos is not None:
        write_raw(dst_bus, 42, 2, mid, pos)

    total_ok += ok
    total_fail += fail
    total_changed += changed
    print()
    print(f"  {name}: {ok} OK ({changed} changed), {fail} failed")

dst_bus.port_handler.closePort()

print(f"\n{'='*64}")
print(f"  Copy complete: {src_arm} -> {dst_arm}")
print(f"  {total_ok} registers written ({total_changed} changed), {total_fail} failures")
print(f"{'='*64}")

if total_fail > 0:
    print("\nSome writes failed. Check the output above for details.")
    print("Motors with voltage errors may need a power cycle first.")
else:
    print("\nAll registers copied successfully.")

print()
print("Next steps:")
print(f"  1. Power-cycle the {dst_arm} arm (unplug and reconnect).")
print(f"  2. Run: python calibrate.py {dst_arm}")
print()
