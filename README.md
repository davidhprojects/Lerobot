# LeRobot

Control code for the SO-101 follower arm using [HuggingFace LeRobot](https://github.com/huggingface/lerobot).

## Setup

All scripts use the `.venv` in the project root. Activate it before running anything:

```powershell
.venv\Scripts\Activate.ps1
```

### Port Detection

Run this once whenever arms are connected to new USB ports. It identifies which COM port each arm is on and saves the result to `ports.json` (machine-local, not committed to git).

```powershell
python Setup\find_ports.py
```

All other scripts read from `ports.json` automatically — no hardcoded ports anywhere.

---

## 1. Motor Setup

Run once when assembling the arm or replacing a motor. Connect one motor at a time when prompted — the script walks through all 6 and assigns each its ID.

```powershell
python Setup\Motor_Setup.py black
python Setup\Motor_Setup.py white
```

If a single motor needs to be re-assigned (e.g. after a hardware fault), use:

```powershell
python Setup\assign_single_motor_id.py black
```

Edit `MOTOR_NAME` at the top of that file to the motor you want to fix before running.

---

## 2. Calibration

Run after motor setup, or any time joint ranges feel off. The script guides you through moving each joint through its full range of motion.

```powershell
python Setup\calibrate.py black
python Setup\calibrate.py white
```

Before pressing Enter at the first prompt, position the arm in the **middle of its range** — all joints away from their mechanical limits. Calibration files are saved to `calibrations/` and shared with the repository.

---

## 3. Record and Replay

Records a motion by moving the arm by hand, then replays it back.

```powershell
python Control\record_and_replay.py
```

1. Press Enter to start recording
2. Move the arm through the motion you want to record
3. Press Enter to stop — the recording is saved to `Control\recorded_motion.json`
4. Press Enter to replay — the arm will repeat the motion

---

## Troubleshooting

**Motor flashing LED / Input voltage error**

A motor's voltage protection registers may be corrupted. Edit `MOTOR_NAME` at the top of the script to the affected motor, then run:

```powershell
python Setup\reset_motor.py black
```
