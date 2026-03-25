"""
Calibrate an arm using lerobot-calibrate.

Usage:
  python calibrate.py black
  python calibrate.py white

The calibration procedure will guide you through moving each joint
to its min and max positions so lerobot can learn the full range of motion.
"""

import json
import subprocess
import sys
from pathlib import Path

ARMS = ["black", "white"]
PORTS_FILE = Path(__file__).parent.parent / "ports.json"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibrations"
PYTHON = sys.executable

if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
    print(f"Usage: python calibrate.py [{' | '.join(ARMS)}]")
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
        PYTHON, "-m", "lerobot.scripts.lerobot_calibrate",
        "--robot.type=so101_follower",
        f"--robot.port={port}",
        f"--robot.id={arm_name}",
        f"--robot.calibration_dir={CALIBRATION_DIR}",
    ],
    check=True,
)
