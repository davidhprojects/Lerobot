"""Deep debug: check every register and write result for gripper motor."""

import scservo_sdk as scs

PORT = "/dev/tty.usbmodem5A7A0159151"
MOTOR_ID = 6
BAUDRATE = 1000000

port = scs.PortHandler(PORT)
ph = scs.PacketHandler(0)

if not port.openPort():
    raise RuntimeError("Failed to open port")
if not port.setBaudRate(BAUDRATE):
    raise RuntimeError("Failed to set baud rate")

def read_reg(addr, size, name):
    if size == 1:
        val, result, error = ph.read1ByteTxRx(port, MOTOR_ID, addr)
    else:
        val, result, error = ph.read2ByteTxRx(port, MOTOR_ID, addr)
    status = "OK" if result == 0 else f"FAIL(result={result})"
    err_str = f"error={error}" if error != 0 else ""
    print(f"  {name:30s} addr={addr:3d}: {val:6d}  {status} {err_str}")
    return val

def write_reg(addr, size, val, name):
    if size == 1:
        result, error = ph.write1ByteTxRx(port, MOTOR_ID, addr, val)
    else:
        result, error = ph.write2ByteTxRx(port, MOTOR_ID, addr, val)
    status = "OK" if result == 0 else f"FAIL(result={result})"
    err_str = f"error={error}" if error != 0 else ""
    print(f"  WRITE {name:27s} addr={addr:3d} val={val:6d}  {status} {err_str}")
    return result, error

print("=== READING ALL RELEVANT REGISTERS ===")
read_reg(5, 1, "ID")
read_reg(6, 1, "Baud_Rate")
read_reg(9, 2, "Min_Position_Limit")
read_reg(11, 2, "Max_Position_Limit")
read_reg(33, 1, "Operating_Mode")
read_reg(34, 1, "Protective_Torque")
read_reg(35, 1, "Protection_Time")
read_reg(36, 1, "Overload_Torque")
read_reg(37, 1, "Velocity_P")
read_reg(40, 1, "Torque_Enable")
read_reg(41, 1, "Acceleration")
read_reg(42, 2, "Goal_Position")
read_reg(46, 2, "Goal_Speed")
read_reg(55, 1, "Lock")
read_reg(56, 2, "Present_Position")
read_reg(58, 2, "Present_Speed")
read_reg(60, 2, "Present_Load")
read_reg(62, 1, "Present_Voltage")
read_reg(63, 1, "Present_Temperature")
read_reg(69, 1, "Current")
read_reg(85, 1, "Maximum_Acceleration")

print("\n=== ATTEMPTING TO MOVE ===")

# Step 1: Unlock and disable torque
write_reg(55, 1, 0, "Lock")
write_reg(40, 1, 0, "Torque_Enable")

# Step 2: Configure
write_reg(33, 1, 0, "Operating_Mode (position)")
write_reg(9, 2, 0, "Min_Position_Limit")
write_reg(11, 2, 4095, "Max_Position_Limit")
write_reg(85, 1, 254, "Maximum_Acceleration")
write_reg(41, 1, 254, "Acceleration")

# Step 3: Enable torque
write_reg(40, 1, 1, "Torque_Enable")

# Step 4: Write goal position
pos_before = read_reg(56, 2, "Present_Position (before)")
target = pos_before + 500 if pos_before < 3000 else pos_before - 500
write_reg(42, 2, target, "Goal_Position")

# Step 5: Check goal was written
read_reg(42, 2, "Goal_Position (readback)")

import time
time.sleep(2)

pos_after = read_reg(56, 2, "Present_Position (after 2s)")
print(f"\nMoved: {pos_before} -> {pos_after} (delta={pos_after - pos_before})")

# Cleanup
write_reg(55, 1, 0, "Lock")
write_reg(40, 1, 0, "Torque_Enable")
port.closePort()
print("Done.")
