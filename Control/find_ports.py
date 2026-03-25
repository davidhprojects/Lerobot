"""
Identify which COM port each robot arm is connected to.

The script asks you to unplug and replug each arm one at a time,
then reports the port for each.
"""

import json
import time
import serial.tools.list_ports
from pathlib import Path

PORTS_FILE = Path(__file__).parent.parent / "ports.json"


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


ARMS = ["black", "white"]
results = {}

print("This script will detect the COM port for each arm.")
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

with open(PORTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 30)
print("Results:")
for arm, port in results.items():
    print(f"  {arm}: {port}")
print(f"\nSaved to {PORTS_FILE}")
