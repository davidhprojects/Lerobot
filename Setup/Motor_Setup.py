#!python
"""
Assign motor IDs to an arm using lerobot-setup-motors.

Usage:
  python Motor_Setup.py black
  python Motor_Setup.py white

Steps:
  1. Run this script with the arm name
  2. For each motor (starting from the last), connect ONLY that motor's wire
  3. Press Enter — the ID will be written automatically
  4. Repeat for all 6 motors
"""

import json
import subprocess
import sys
from pathlib import Path

ARMS = ["black", "white"]
PORTS_FILE = Path(__file__).parent.parent / "ports.json"
PYTHON = sys.executable

if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
    print(f"Usage: python Motor_Setup.py [{' | '.join(ARMS)}]")
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

subprocess.run(
    [
        PYTHON, "-m", "lerobot.scripts.lerobot_setup_motors",
        "--robot.type=so101_follower",
        f"--robot.port={port}",
    ],
    check=True,
)
