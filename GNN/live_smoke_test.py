"""
Live end-to-end smoke test: camera -> ArUco -> SceneObserver -> build_graph.

Joint angles are fed as zeros (motors not required), so the two arm-EE
positions are constant.  The tray positions are real — move the tray and
you'll see ``tray_left`` / ``tray_right`` positions and velocities update.

Run from the project root:

    python GNN/live_smoke_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.camera import RealSenseCamera
from perception.aruco import ArucoDetector, MarkerConfig
from Algorithmic.calibrate import load_base_transforms
from GNN.scene_observer import SceneObserver
from GNN.graph_builder import build_graph


PRINT_INTERVAL_S = 1.0


def _print_entities(tick: int, ents):
    print(f"\ntick {tick}")
    for e in ents:
        p = e.position * 1000  # mm
        v = e.velocity * 1000  # mm/s
        print(f"  {e.name:11s}  "
              f"pos=({p[0]:+7.1f},{p[1]:+7.1f},{p[2]:+7.1f}) mm   "
              f"vel=({v[0]:+6.1f},{v[1]:+6.1f},{v[2]:+6.1f}) mm/s")


def main():
    transforms = load_base_transforms()
    if "left" not in transforms or "right" not in transforms:
        print("ERROR: base_transforms.json missing a side. Re-run calibrate.py.")
        sys.exit(1)

    camera = RealSenseCamera()
    camera.connect()
    print(f"Camera connected (S/N {camera.serial_number}).")

    detector = ArucoDetector(
        config=MarkerConfig(),
        camera_matrix=camera.get_camera_matrix(),
        dist_coeffs=camera.get_dist_coeffs(),
    )
    observer = SceneObserver(transforms, dt=1.0 / 30.0)

    zero_joints = np.zeros(5)  # no motors connected; arm-EE positions will be static

    print("Streaming... Ctrl+C to stop.")
    print("Move the tray to see tray_left / tray_right velocity change.\n")

    last_print = 0.0
    tick = 0
    try:
        while True:
            frame = camera.get_frame()
            poses = detector.detect(frame)
            ents = observer.observe(poses, zero_joints, zero_joints)
            tick += 1

            now = time.time()
            if now - last_print >= PRINT_INTERVAL_S:
                if ents is None:
                    visible = sorted(poses.keys())
                    print(f"\rtick {tick}: tray markers missing (visible IDs: {visible})",
                          end="", flush=True)
                else:
                    g = build_graph(ents)
                    _print_entities(tick, ents)
                    print(f"  graph.x.shape={tuple(g.x.shape)}  "
                          f"edge_attr.shape={tuple(g.edge_attr.shape)}")
                last_print = now
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.disconnect()
        print("Camera disconnected.")


if __name__ == "__main__":
    main()
