"""Debug: try to enable torque with retries and check Goal_Speed."""

import time
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

def read_reg(addr, size, name=""):
    if size == 1:
        val, result, error = ph.read1ByteTxRx(port, MOTOR_ID, addr)
    else:
        val, result, error = ph.read2ByteTxRx(port, MOTOR_ID, addr)
    return val, result, error

def write_reg_retry(addr, size, val, name, retries=5):
    for i in range(retries):
        if size == 1:
            result, error = ph.write1ByteTxRx(port, MOTOR_ID, addr, val)
        else:
            result, error = ph.write2ByteTxRx(port, MOTOR_ID, addr, val)
        if result == 0 and error == 0:
            print(f"  WRITE {name}: {val} - OK (attempt {i+1})")
            return True
        time.sleep(0.05)
    print(f"  WRITE {name}: {val} - FAILED after {retries} attempts (result={result}, error={error})")
    return False

print("=== Unlock and disable torque ===")
write_reg_retry(55, 1, 0, "Lock")
write_reg_retry(40, 1, 0, "Torque_Enable")
time.sleep(0.1)

print("\n=== Set Goal_Speed (addr 46) to 500 ===")
write_reg_retry(46, 2, 500, "Goal_Speed")

print("\n=== Enable torque ===")
success = write_reg_retry(40, 1, 1, "Torque_Enable")

# Verify torque is actually on
val, _, _ = read_reg(40, 1)
print(f"  Torque_Enable readback: {val}")

if val != 1:
    print("\n  Torque failed to enable! Trying alternative approach...")
    # Some STS3215 firmware needs Lock=0 explicitly before torque enable
    write_reg_retry(55, 1, 0, "Lock")
    time.sleep(0.1)
    write_reg_retry(40, 1, 1, "Torque_Enable")
    val, _, _ = read_reg(40, 1)
    print(f"  Torque_Enable readback (2nd attempt): {val}")

if val == 1:
    print("\n=== Torque is ON, writing Goal_Position ===")
    pos, _, _ = read_reg(56, 2)
    print(f"  Current position: {pos}")

    target = pos + 500 if pos < 3000 else pos - 500
    write_reg_retry(42, 2, target, "Goal_Position")

    # Readback goal
    goal, _, _ = read_reg(42, 2)
    print(f"  Goal_Position readback: {goal}")

    time.sleep(2)

    pos2, _, _ = read_reg(56, 2)
    print(f"  Position after 2s: {pos2} (delta={pos2-pos})")

    load, _, _ = read_reg(60, 2)
    print(f"  Present_Load: {load}")
else:
    print("\n  CANNOT ENABLE TORQUE - this is the root problem!")

    # Check voltage
    v, _, _ = read_reg(62, 1)
    print(f"  Voltage: {v/10.0}V")
    print(f"  (STS3215 needs 6-7.5V, USB alone is ~5V)")

# Cleanup
write_reg_retry(55, 1, 0, "Lock")
write_reg_retry(40, 1, 0, "Torque_Enable")
port.closePort()
print("\nDone.")
