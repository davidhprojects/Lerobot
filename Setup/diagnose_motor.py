#!python
"""
Read diagnostic registers from motors.

Usage:
  python diagnose_motor.py              — voltage table for all 12 motors
  python diagnose_motor.py right        — full diagnostics + nudge test for MOTOR_NAME on right
  python diagnose_motor.py left         — full diagnostics + nudge test for MOTOR_NAME on left

Edit MOTOR_NAME below to select which motor gets the full single-motor diagnostics.
"""

import json
import sys
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = ["right", "left"]
PORTS_FILE = Path(__file__).parent / "ports.json"

MOTOR_NAME = "shoulder_pan"

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

ADDR_TORQUE_ENABLE   = (40, 1)
ADDR_LOCK            = (55, 1)
ADDR_MAX_VOLTAGE     = (14, 1)
ADDR_MIN_VOLTAGE     = (15, 1)
ADDR_PRESENT_VOLTAGE = (62, 1)

REGISTERS = [
    (5,  1, "ID"),
    (6,  1, "Baud_Rate"),
    (9,  2, "Min_Angle_Limit (raw)"),
    (11, 2, "Max_Angle_Limit (raw)"),
    (14, 1, "Max_Voltage_Limit"),
    (15, 1, "Min_Voltage_Limit"),
    (16, 2, "Max_Torque_Limit"),
    (18, 1, "Phase"),
    (19, 1, "Unloading_Condition"),
    (21, 1, "P_Coefficient"),
    (22, 1, "D_Coefficient"),
    (23, 1, "I_Coefficient"),
    (24, 2, "Minimum_Startup_Force"),
    (28, 2, "Protection_Current"),
    (33, 1, "Operating_Mode"),
    (34, 1, "Protective_Torque"),
    (35, 1, "Protection_Time"),
    (36, 1, "Overload_Torque"),
    (40, 1, "Torque_Enable"),
    (41, 1, "Acceleration"),
    (48, 2, "Torque_Limit"),
    (55, 1, "Lock"),
    (42, 2, "Goal_Position (raw)"),
    (56, 2, "Present_Position (raw)"),
    (58, 2, "Present_Speed (raw)"),
    (60, 2, "Present_Load (raw)"),
    (62, 1, "Present_Voltage"),
    (63, 1, "Present_Temperature"),
    (65, 1, "Error_Status"),
    (66, 1, "Moving"),
    (69, 2, "Present_Current"),
    (85, 1, "Maximum_Acceleration"),
]

with open(PORTS_FILE) as f:
    ports = json.load(f)


def connect(arm):
    bus = FeetechMotorsBus(port=ports[arm], motors=MOTORS)
    bus._connect(handshake=False)
    bus.set_baudrate(1000000)
    return bus


def read_raw(bus, addr, length, mid):
    val, comm, _ = bus._read(addr, length, mid, raise_on_error=False)
    return val if bus._is_comm_success(comm) else None


def voltage_table():
    print(f"{'Motor':<16}", end="")
    for arm in ARMS:
        print(f"  {arm:>8}", end="")
    print()
    print("-" * (16 + len(ARMS) * 10))

    buses = {arm: connect(arm) for arm in ARMS}

    for name in MOTOR_ORDER:
        mid = MOTORS[name].id
        print(f"{name:<16}", end="")
        for arm in ARMS:
            val = read_raw(buses[arm], 62, 1, mid)
            if val is None:
                print(f"  {'FAIL':>8}", end="")
            else:
                v = val * 0.1
                flag = " !" if v < 6.0 else "  "
                print(f"  {v:>6.1f}V{flag}", end="")
        print()

    print()
    print("! = below 6.0 V (motor will not move)")

    for bus in buses.values():
        bus.port_handler.closePort()


def single_motor_diagnostics(arm_name):
    motor_id = MOTORS[MOTOR_NAME].id
    port = ports[arm_name]
    print(f"Arm  : {arm_name} on {port}")
    print(f"Motor: {MOTOR_NAME} (ID {motor_id})")
    print()

    bus = connect(arm_name)

    print(f"{'Register':<30} {'Raw':>6}  Notes")
    print("-" * 60)
    for addr, length, label in REGISTERS:
        val = read_raw(bus, addr, length, motor_id)
        if val is None:
            print(f"{label:<30} {'READ FAIL':>6}")
            continue

        note = ""
        if label == "Present_Voltage":
            note = f"= {val * 0.1:.1f} V"
        elif label == "Present_Temperature":
            note = f"= {val} °C"
        elif label == "Baud_Rate":
            baud_map = {0: 1000000, 1: 500000, 2: 250000, 3: 128000,
                        4: 115200, 5: 76800, 6: 57600, 7: 38400}
            note = f"= {baud_map.get(val, '?')} bps"
        elif label == "Torque_Enable":
            note = "ENABLED" if val else "disabled"
        elif label == "Moving":
            note = "YES" if val else "no"
        elif label == "Lock":
            note = "LOCKED" if val else "unlocked"
        elif label == "Error_Status":
            errors = []
            if val & 0x01: errors.append("InputVoltage")
            if val & 0x04: errors.append("Overheat")
            if val & 0x08: errors.append("OverCurrent")
            if val & 0x20: errors.append("Overload")
            note = ", ".join(errors) if errors else "none"
        elif "Voltage_Limit" in label:
            note = f"= {val * 0.1:.1f} V"

        print(f"{label:<30} {val:>6}  {note}")

    print()
    import time

    # Read angle limits to find the midpoint of the motor's usable range
    min_limit = read_raw(bus, 9, 2, motor_id) or 0
    max_limit = read_raw(bus, 11, 2, motor_id) or 4095
    midpoint = (min_limit + max_limit) // 2

    print("Movement test with write verification:")
    pos_before = read_raw(bus, 56, 2, motor_id)
    if pos_before is None:
        print("  Could not read position — skipping test.")
        bus.port_handler.closePort()
        print("\nDone.")
        return

    target = midpoint
    print(f"  Current position : {pos_before}")
    print(f"  Target (midpoint): {target}")
    print(f"  Distance         : {abs(target - pos_before)} counts\n")

    # Step 1: Clear Lock and disable torque to get a clean state
    print("  Step 1: Clearing Lock and disabling torque...")
    bus._write(55, 1, motor_id, 0, raise_on_error=False)  # Lock = 0
    bus._write(40, 1, motor_id, 0, raise_on_error=False)  # Torque off
    time.sleep(0.1)
    lock_val = read_raw(bus, 55, 1, motor_id)
    torque_val = read_raw(bus, 40, 1, motor_id)
    print(f"    Lock = {lock_val}, Torque = {torque_val}")

    # Step 2: Write Goal_Position BEFORE enabling torque
    print(f"\n  Step 2: Writing Goal_Position = {target}...")
    bus._write(42, 2, motor_id, target, raise_on_error=False)
    time.sleep(0.05)
    goal_readback = read_raw(bus, 42, 2, motor_id)
    print(f"    Goal readback: {goal_readback}  (expected {target})")
    if goal_readback != target:
        print(f"    *** WRITE FAILED — Goal_Position did not update! ***")

    # Step 3: Enable torque and watch position over 2 seconds
    print(f"\n  Step 3: Enabling torque, monitoring for 2 seconds...")
    bus._write(40, 1, motor_id, 1, raise_on_error=False)
    time.sleep(0.05)
    torque_readback = read_raw(bus, 40, 1, motor_id)
    torque_limit = read_raw(bus, 48, 2, motor_id)
    print(f"    Torque_Enable readback: {torque_readback}  (expected 1) — {'OK' if torque_readback == 1 else '*** FAILED ***'}")
    print(f"    Torque_Limit: {torque_limit}  (should be 1000 = 100%){' — *** ZERO = motor cannot produce force ***' if torque_limit == 0 else ''}")

    for i in range(8):
        time.sleep(0.25)
        pos = read_raw(bus, 56, 2, motor_id)
        speed = read_raw(bus, 58, 2, motor_id)
        load = read_raw(bus, 60, 2, motor_id)
        err = read_raw(bus, 65, 1, motor_id)
        goal = read_raw(bus, 42, 2, motor_id)
        print(f"    t={0.25*(i+1):.2f}s  pos={pos:>5}  goal={goal:>5}  speed={speed:>5}  load={load:>5}  err={err}")

    pos_after = read_raw(bus, 56, 2, motor_id)

    # Step 4: Return to original position and disable torque
    bus._write(42, 2, motor_id, pos_before, raise_on_error=False)
    time.sleep(1.0)
    bus._write(40, 1, motor_id, 0, raise_on_error=False)
    bus._write(55, 1, motor_id, 0, raise_on_error=False)

    delta = abs(pos_after - pos_before) if pos_after is not None else 0
    expected = abs(target - pos_before)
    print(f"\n  Summary:")
    print(f"    Start    : {pos_before}")
    print(f"    Target   : {target}")
    print(f"    Reached  : {pos_after}")
    print(f"    Moved    : {delta} / {expected} counts ({100*delta/expected:.0f}%)" if expected else "")
    if delta > expected * 0.8:
        print(f"    Result   : PASS")
    elif delta > 50:
        print(f"    Result   : PARTIAL — motor moved but fell short")
    else:
        print(f"    Result   : FAIL — motor did not move to target")

    bus.port_handler.closePort()


if len(sys.argv) == 1:
    voltage_table()
elif len(sys.argv) == 2 and sys.argv[1] in ARMS:
    single_motor_diagnostics(sys.argv[1])
else:
    print(f"Usage: python diagnose_motor.py [{' | '.join(ARMS)} | (no args for voltage table)]")
    sys.exit(1)

print("\nDone.")
