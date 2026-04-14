"""
Scene state tracker — Phase 1c.

Combines ArUco detections (tray side markers, other arm's gripper) with
forward kinematics (own arm) into a single scene snapshot each frame.
Maintains a short history to compute smoothed velocity estimates via
exponential moving average.

The glass is treated as rigidly attached to the tray — no separate tracking.
Tilt is derived from the height difference between the two tray markers.

This is the top-level perception interface consumed by the graph builder
in the next step of the pipeline.
"""

import numpy as np
from dataclasses import dataclass, field


# All entity keys the tracker produces — matches the 4-node graph in Phase 2.
ENTITY_KEYS = [
    "left_ee",
    "right_ee",
    "tray_left",
    "tray_right",
]


@dataclass
class EntityState:
    """Position + velocity for a single tracked entity."""
    position: np.ndarray          # (3,) xyz in meters, world frame
    velocity: np.ndarray          # (3,) smoothed velocity in m/s
    orientation: np.ndarray | None = None  # (4,) quaternion, if available
    valid: bool = True            # False when marker was not detected this frame


@dataclass
class SceneState:
    """Full snapshot of all tracked entities for one frame."""
    timestamp: float
    entities: dict[str, EntityState] = field(default_factory=dict)

    def positions_dict(self) -> dict[str, np.ndarray]:
        """Return {entity_key: position} for all valid entities."""
        return {k: e.position for k, e in self.entities.items() if e.valid}

    def velocities_dict(self) -> dict[str, np.ndarray]:
        """Return {entity_key: velocity} for all valid entities."""
        return {k: e.velocity for k, e in self.entities.items() if e.valid}

    @property
    def tray_tilt_rad(self) -> float | None:
        """
        Tilt angle of the tray in radians, derived from the height difference
        between the two tray markers.

        Positive = right side higher than left.
        Returns None if either tray marker is invalid.
        """
        left = self.entities.get("tray_left")
        right = self.entities.get("tray_right")
        if left is None or right is None or not left.valid or not right.valid:
            return None
        dz = right.position[2] - left.position[2]
        dx = np.linalg.norm(right.position[:2] - left.position[:2])
        if dx < 1e-6:
            return 0.0
        return float(np.arctan2(dz, dx))


class SceneTracker:
    """
    Fuses perception sources into a tracked scene state with velocities.

    Call update() once per frame with the latest detections. It computes
    velocities as exponentially-smoothed finite differences (Phase 1c).

    Parameters
    ----------
    fps : float
        Expected frame rate (used for dt in velocity calculation).
    ema_alpha : float
        Smoothing factor for velocity EMA.  Higher = more responsive,
        lower = smoother.  0.3 is a good starting point.
    """

    def __init__(self, fps: float = 30.0, ema_alpha: float = 0.3):
        self.dt = 1.0 / fps
        self.alpha = ema_alpha
        self._prev_positions: dict[str, np.ndarray] = {}
        self._smoothed_velocities: dict[str, np.ndarray] = {}

    def update(self, timestamp: float,
               positions: dict[str, np.ndarray],
               orientations: dict[str, np.ndarray] | None = None,
               ) -> SceneState:
        """
        Produce a new SceneState from the latest raw positions.

        Parameters
        ----------
        timestamp : float
            Current time in seconds.
        positions : dict[str, ndarray]
            Raw 3D positions for each detected entity this frame.
            Missing keys = entity not detected.
        orientations : dict[str, ndarray], optional
            Quaternions for entities that have orientation (arm EEs).

        Returns
        -------
        SceneState with position, velocity, and validity for every entity.
        """
        if orientations is None:
            orientations = {}

        entities: dict[str, EntityState] = {}

        for key in ENTITY_KEYS:
            if key not in positions:
                # Entity not detected this frame
                entities[key] = EntityState(
                    position=self._prev_positions.get(key, np.zeros(3)),
                    velocity=self._smoothed_velocities.get(key, np.zeros(3)),
                    valid=False,
                )
                continue

            pos = positions[key]
            vel = self._compute_velocity(key, pos)
            entities[key] = EntityState(
                position=pos,
                velocity=vel,
                orientation=orientations.get(key),
                valid=True,
            )
            self._prev_positions[key] = pos

        return SceneState(timestamp=timestamp, entities=entities)

    def _compute_velocity(self, key: str, current_pos: np.ndarray) -> np.ndarray:
        """
        Finite-difference velocity with exponential moving average smoothing.

        velocity(t) = (pos(t) - pos(t-1)) / dt
        smoothed(t) = alpha * velocity(t) + (1 - alpha) * smoothed(t-1)
        """
        if key not in self._prev_positions:
            self._smoothed_velocities[key] = np.zeros(3)
            return np.zeros(3)

        raw_vel = (current_pos - self._prev_positions[key]) / self.dt
        prev_smooth = self._smoothed_velocities.get(key, np.zeros(3))
        smoothed = self.alpha * raw_vel + (1.0 - self.alpha) * prev_smooth
        self._smoothed_velocities[key] = smoothed
        return smoothed

    def reset(self):
        """Clear all history (e.g. at the start of a new episode)."""
        self._prev_positions.clear()
        self._smoothed_velocities.clear()
