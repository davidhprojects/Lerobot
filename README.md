# LeRobot

Control code for the SO-101 follower arm using [HuggingFace LeRobot](https://github.com/huggingface/lerobot).

## Setup

All scripts use the `.venv` in the project root. Activate it before running anything:

```powershell
.venv\Scripts\Activate.ps1
```

The follower arm connects on **COM3**. If it changes, update the `PORT` variable at the top of each script.

---

## 1. Motor Setup

Run once when assembling the arm or replacing a motor. Connect one motor at a time when prompted — the script walks through all 6 and assigns each its ID.

```powershell
python Control\Motor_Setup.py
```

If a single motor needs to be re-assigned (e.g. after a hardware fault), use:

```powershell
python Control\assign_single_motor_id.py
```

Edit `MOTOR_NAME` at the top of that file to the motor you want to fix before running.

---

## 2. Calibration

Run after motor setup, or any time joint ranges feel off. The script guides you through moving each joint through its full range of motion.

```powershell
python Control\calibrate.py
```

Before pressing Enter at the first prompt, position the arm in the **middle of its range** — all joints away from their mechanical limits. Calibration files are saved to `calibrations/` and shared with the repository.

---

## 3. Record and Replay

Records a motion by moving the arm by hand, then replays it back.

```powershell
python Control\record_and_replay_follower.py
```

1. Press Enter to start recording
2. Move the arm through the motion you want to record
3. Press Enter to stop — the recording is saved to `Control\recorded_motion_follower.json`
4. Press Enter to replay — the arm will repeat the motion

---

## Troubleshooting

**Motor not found / missing motor IDs**

Run `scan_motors.py` with all motors connected to see which ones are responding and at what baudrate:

```powershell
python Control\scan_motors.py
```

**Motor flashing LED / Input voltage error**

A motor's voltage protection registers may be corrupted. Run the reset script with only that motor connected:

```powershell
python Control\reset_motor.py
```

Edit `MOTOR_NAME` at the top to the affected motor before running.
