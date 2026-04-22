"""
Abstraction for modular phase controllers.

Each phase of the bimanual tray-lift sequence (HOVER, APPROACH, LIFT,
...) is represented as a ``PhaseController`` subclass. A controller
can be either a hand-coded replica of run.py's logic
(``GNN.phases.deterministic``) or a GNN that was trained on just that
phase's frames (``GNN.phases.learned``).

run_hybrid.py's main() constructs a list of controllers per arm — mix
and match deterministic + learned as you want to swap phases in and
out — and runs them sequentially on a shared PhaseContext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import threading

import numpy as np


@dataclass
class PhaseContext:
    """All shared state one arm's phases need.

    Built once per arm in run_hybrid.py's main() before the per-arm
    threads start. Each thread iterates its list of PhaseController
    instances, calling ``run(ctx)`` on each in order. Everything in
    this object is read-only for the duration of a phase except
    ``carryover``, which phases use to hand data to successors.
    """

    # --- Hardware / perception ---
    side: str
    robot: Any                                      # lerobot SOFollower
    get_frame: Callable[[], Optional[np.ndarray]]   # FrameProvider.get_frame
    detector: Any                                   # perception.aruco.ArucoDetector

    # --- Motion-map & calibration ---
    motion_map: Any                                 # algorithmic.motion_map.MotionMap
    T_base_cam: np.ndarray                          # camera -> own base frame
    descent_delta: dict[str, float]                 # grasp - pre_grasp, per motor

    # --- Gripper commands + tray baseline (per arm) ---
    gripper_open: float
    gripper_closed: float
    baseline_tray_y: dict[str, float]               # "tray_left", "tray_right"

    waypoints: dict                                 # the parsed waypoints.json

    # --- Shutdown signalling ---
    stop_event: threading.Event

    # --- Optional plumbing used only by some phases ---
    observer: Any = None                            # GNN.scene_observer.SceneObserver
    joint_snapshot: Any = None                      # shared dict of latest joint reads

    # Scratch space for phase-to-phase handoff (e.g. a LIFT phase can
    # read the final descent_target left by APPROACH).
    carryover: dict = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Return value from ``PhaseController.run()``.

    ``success`` = False means abort the whole per-arm sequence (used
    when ``stop_event`` is set mid-phase or an unrecoverable error
    occurs). ``carryover`` merges into ``ctx.carryover`` for the
    next phase.
    """

    success: bool = True
    carryover: dict = field(default_factory=dict)


class PhaseController(ABC):
    """One swappable unit of the per-arm state machine.

    Subclasses implement ``run()`` — blocking until the phase
    completes or ``ctx.stop_event`` is set. ``name`` is used for
    logging and for matching against the phase labels written by
    run.STATE_TRACKER during data collection.
    """

    name: str = "UNNAMED"

    @abstractmethod
    def run(self, ctx: PhaseContext) -> PhaseResult: ...
