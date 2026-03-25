"""
Calibrate the SO-101 follower arm using lerobot-calibrate.

The calibration procedure will guide you through moving each joint
to its min and max positions so lerobot can learn the full range of motion.
"""

import subprocess
import sys

PORT = "COM3"
PYTHON = sys.executable

subprocess.run(
    [
        PYTHON, "-m", "lerobot.scripts.lerobot_calibrate",
        "--robot.type=so101_follower",
        f"--robot.port={PORT}",
        "--robot.id=follower_arm",
        "--robot.calibration_dir=calibrations",
    ],
    check=True,
)
