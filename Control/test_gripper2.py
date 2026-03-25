"""Debug gripper motor directly via scservo SDK."""

import time
import scservo_sdk as scs

PORT = "COM3"
MOTOR_ID = 6
BAUDRATE = 1000000

port = scs.PortHandler(PORT)
ph = scs.PacketHandler(0)  # protocol 0

if not port.openPort():
    raise RuntimeError("Failed to open port")
if not port.setBaudRate(BAUDRATE):
    raise RuntimeError("Failed to set baud rate")

# Read current position (addr 56, 2 bytes)
pos, result, error = ph.read2ByteTxRx(port, MOTOR_ID, 56)
print(f"Current position: {pos}, result: {result}, error: {error}")

# Read torque enable (addr 40, 1 byte)
torque, result, error = ph.read1ByteTxRx(port, MOTOR_ID, 40)
print(f"Torque enabled: {torque}")

# Read lock (addr 55, 1 byte)
lock, result, error = ph.read1ByteTxRx(port, MOTOR_ID, 55)
print(f"Lock: {lock}")

# Read operating mode (addr 33, 1 byte)
mode, result, error = ph.read1ByteTxRx(port, MOTOR_ID, 33)
print(f"Operating mode: {mode}")

# Read min/max position limits (addr 9=min 2bytes, addr 11=max 2bytes)
minp, _, _ = ph.read2ByteTxRx(port, MOTOR_ID, 9)
maxp, _, _ = ph.read2ByteTxRx(port, MOTOR_ID, 11)
print(f"Position limits: min={minp}, max={maxp}")

# Disable torque, unlock, set position mode, then re-enable
print("\nConfiguring motor...")
ph.write1ByteTxRx(port, MOTOR_ID, 55, 0)  # Lock = 0
ph.write1ByteTxRx(port, MOTOR_ID, 40, 0)  # Torque = 0
ph.write1ByteTxRx(port, MOTOR_ID, 33, 0)  # Operating mode = position
ph.write2ByteTxRx(port, MOTOR_ID, 9, 0)   # Min position = 0
ph.write2ByteTxRx(port, MOTOR_ID, 11, 4095)  # Max position = 4095
ph.write1ByteTxRx(port, MOTOR_ID, 40, 1)  # Torque = 1
ph.write1ByteTxRx(port, MOTOR_ID, 55, 1)  # Lock = 1

# Move to 1623
print("Moving to 1623...")
ph.write2ByteTxRx(port, MOTOR_ID, 42, 1623)  # Goal_Position addr=42
time.sleep(2)
pos, _, _ = ph.read2ByteTxRx(port, MOTOR_ID, 56)
print(f"Position now: {pos}")

# Move to 3071
print("Moving to 3071...")
ph.write2ByteTxRx(port, MOTOR_ID, 42, 3071)
time.sleep(2)
pos, _, _ = ph.read2ByteTxRx(port, MOTOR_ID, 56)
print(f"Position now: {pos}")

# Release
ph.write1ByteTxRx(port, MOTOR_ID, 55, 0)
ph.write1ByteTxRx(port, MOTOR_ID, 40, 0)
port.closePort()
print("Done.")
