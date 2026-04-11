# How to Measure SO-101 Joint Offsets

The URDF `xyz` values are in the **parent joint's local coordinate frame**, which rotates at every joint — so they don't directly correspond to "up/forward/sideways" in the world. Here's how to measure each one practically.

## General Approach

Put the arm in its **zero/home pose** (all joints at 0). At each joint, you're measuring from the **center of one rotation axis** to the **center of the next rotation axis**. The tricky part is decomposing that distance into the right local axes.

---

## What to Measure, Joint by Joint

### Joint 1: `shoulder_pan` (base → shoulder)

The base rotation. Stand the arm upright.

| Axis | URDF Default | How to Measure |
|------|-------------|----------------|
| **Z** | 62.4 mm | Vertical distance from the bottom of the base to the center of the shoulder pan rotation axis (the point it spins around) |
| **X** | 38.8 mm | Horizontal offset from the center of the base to the shoulder pivot. Look at the arm from above — the shoulder joint isn't centered, it's offset forward |

### Joint 2: `shoulder_lift` (shoulder → upper arm)

This one has a complex RPY that reorients the frame, so the xyz values are unintuitive. You're measuring the offset from the shoulder pan axis to the shoulder lift axis (the hinge that tips the upper arm up/down).

| Axis | URDF Default | How to Measure |
|------|-------------|----------------|
| **X** | 30.4 mm | Component of the 3D offset between the two axes |
| **Y** | 18.3 mm | Component of the 3D offset between the two axes |
| **Z** | 54.2 mm | Component of the 3D offset between the two axes |

> **Easiest approach:** Measure the **straight-line distance** between the two axes (~65 mm) and note the direction. The three components come from the servo being mounted at an angle inside the housing.

### Joint 3: `elbow_flex` — the upper arm

This is the most important one. Measure from the **shoulder lift hinge** to the **elbow hinge**.

| Axis | URDF Default | How to Measure |
|------|-------------|----------------|
| **X** | 112.6 mm | The main arm link length — measure straight along the upper arm tube between the two hinge pins. **This is the big number.** |
| **Y** | 28.0 mm | Lateral offset — the elbow axis is shifted sideways from the upper arm axis due to the servo body. Look at the arm from the front; the elbow pin isn't perfectly in line with the shoulder pin. Measure that sideways step. |

### Joint 4: `wrist_flex` — the forearm

Measure from the **elbow hinge** to the **wrist flex hinge**.

| Axis | URDF Default | How to Measure |
|------|-------------|----------------|
| **X** | 134.9 mm | The forearm link length — straight-line distance along the forearm tube between the two hinge pins. **This is the biggest link.** |
| **Y** | 5.2 mm | Small lateral offset, similar idea as above. Likely negligible but check if the wrist hinge is slightly offset from the elbow. |

### Joint 5: `wrist_roll` (wrist flex → gripper)

Measure from the **wrist flex hinge** to the **center of the roll axis** (the axis the gripper spins around).

| Axis | URDF Default | How to Measure |
|------|-------------|----------------|
| **Y** | 61.1 mm | Distance along the wrist from the flex hinge to where the roll axis begins |
| **Z** | 18.1 mm | Small vertical offset between the two axes |

### Fixed: gripper frame (roll axis → tool center point)

Measure from the **wrist roll axis** to the **tip of the closed gripper** (or wherever you want your "end-effector point" to be).

| Axis | URDF Default | How to Measure |
|------|-------------|----------------|
| **Z** | 98.1 mm | Distance from the roll axis down to the gripper tip. **This is the main one.** |

---

## Tips

- Use **calipers** if you have them — a tape measure is tough for the small offsets (5 mm, 18 mm)
- Mark each joint's **rotation center** with a dot or tape. The rotation center is the middle of the hinge pin, not the outside of the housing
- The two big numbers that matter most are the **upper arm (112.6 mm)** and **forearm (134.9 mm)**. Get these right first — small offsets on other joints have less impact on EE accuracy
- If a measurement is hard to decompose into x/y/z, just measure the straight-line distance and note the direction. You can compute the components with basic trig afterward
