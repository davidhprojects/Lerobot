#!python
"""
Identify which COM port each robot arm is connected to and detect
the Intel RealSense D435i camera.

The script asks you to unplug and replug each arm one at a time,
then auto-detects the RealSense camera. Results are saved to ports.json.
"""

import json
import time
import serial.tools.list_ports
from pathlib import Path

PORTS_FILE = Path(__file__).parent / "ports.json"


def get_ports():
    return {p.device for p in serial.tools.list_ports.comports()}


def wait_for_removal(before):
    print("  Unplug the arm now...")
    while True:
        current = get_ports()
        removed = before - current
        if removed:
            return removed.pop(), current
        time.sleep(0.2)


def wait_for_reconnect(before):
    print("  Plug it back in...")
    while True:
        current = get_ports()
        added = current - before
        if added:
            return added.pop()
        time.sleep(0.2)


def find_realsense():
    """Detect an Intel RealSense camera and return its serial number."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  pyrealsense2 not installed — skipping camera detection.")
        return None

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("  No RealSense camera detected.")
        return None

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  Found: {name}  (S/N: {serial}, FW: {fw})")
        return {"name": name, "serial_number": serial, "firmware_version": fw}


# --- Arm detection ---

ARMS = ["right", "left"]
results = {}

print("This script will detect the COM port for each arm and the RealSense camera.")
print("Follow the prompts for each arm.\n")

for arm in ARMS:
    input(f"--- {arm} ---\nPress Enter when ready, then unplug the {arm} arm...")
    before = get_ports()
    removed_port, after_removal = wait_for_removal(before)
    print(f"  Detected removal of {removed_port}")
    wait_for_reconnect(after_removal)
    time.sleep(0.5)  # let the port settle
    after_reconnect = get_ports()
    added = after_reconnect - after_removal
    port = added.pop() if added else removed_port
    results[arm] = port
    print(f"  {arm} is on {port}\n")

# --- Camera detection ---

print("--- RealSense D435i ---")
camera_info = find_realsense()
if camera_info:
    results["camera"] = camera_info

# --- Save ---

with open(PORTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 30)
print("Results:")
for key, val in results.items():
    if key == "camera":
        print(f"  camera: {val['name']} (S/N: {val['serial_number']})")
    else:
        print(f"  {key}: {val}")
print(f"\nSaved to {PORTS_FILE}")
