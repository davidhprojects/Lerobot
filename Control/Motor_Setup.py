"""
Assign motor IDs to the SO-100 follower arm using lerobot-setup-motors.

Steps:
  1. Run this script
  2. For each motor (starting from the last), connect ONLY that motor's wire
  3. Press Enter — the ID will be written automatically
  4. Repeat for all 6 motors
"""

import subprocess
import sys

PORT = "COM3"
PYTHON = sys.executable

subprocess.run(
    [
        PYTHON, "-m", "lerobot.scripts.lerobot_setup_motors",
        "--robot.type=so101_follower",
        f"--robot.port={PORT}",
    ],
    check=True,
)
