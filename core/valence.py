"""
Valence vectors and trajectories.

A ValenceVector is a single scored observation: how activated is each
axis at a given moment. A ValenceTrajectory is a sequence of vectors
over time — the basis for projection and threshold triggering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ValenceVector:
    """
    A single point in valence space.

    All values are non-negative. Zero is homeostasis.
    Higher values mean greater deviation from neutral.
    """

    domain_name: str
    scores: list[float]
    timestamp: float = 0.0

    def __post_init__(self):
        if any(s < 0 for s in self.scores):
            raise ValueError("Valence scores must be >= 0. Zero is homeostasis.")

    @property
    def magnitude(self) -> float:
        """Total activation — L2 norm. Zero means nothing is happening."""
        return math.sqrt(sum(s * s for s in self.scores))

    @property
    def dominant_axis(self) -> int:
        """Which axis has the highest activation."""
        return max(range(len(self.scores)), key=lambda i: self.scores[i])

    @property
    def is_neutral(self) -> bool:
        return all(s == 0.0 for s in self.scores)

    def deviation_from(self, other: ValenceVector) -> float:
        """Distance between two valence states."""
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.scores, other.scores))
        )

    def __repr__(self):
        rounded = [round(s, 2) for s in self.scores]
        return f"V({self.domain_name} t={self.timestamp:.1f} {rounded})"


@dataclass
class ValenceTrajectory:
    """
    A time-ordered sequence of valence vectors.

    This is where prediction happens. Given the trajectory so far,
    project where each axis is heading. The organism doesn't wait
    for peak activation — it extrapolates and acts at threshold.
    """

    domain_name: str
    vectors: list[ValenceVector] = field(default_factory=list)

    def append(self, v: ValenceVector):
        if v.domain_name != self.domain_name:
            raise ValueError(f"Domain mismatch: {v.domain_name} vs {self.domain_name}")
        self.vectors.append(v)

    @property
    def length(self) -> int:
        return len(self.vectors)

    def velocity(self, axis: int, window: int = 3) -> float:
        """
        Rate of change on a given axis over the last `window` observations.

        Positive = increasing activation (approaching threat/opportunity).
        Negative = returning toward homeostasis.
        Zero = stable.
        """
        if len(self.vectors) < 2:
            return 0.0

        recent = self.vectors[-window:]
        if len(recent) < 2:
            return 0.0

        dt = recent[-1].timestamp - recent[0].timestamp
        if dt == 0:
            return 0.0

        ds = recent[-1].scores[axis] - recent[0].scores[axis]
        return ds / dt

    def project(self, axis: int, steps_ahead: float = 1.0, window: int = 3) -> float:
        """
        Linear projection of where an axis score will be in `steps_ahead` time units.

        This is the simplest projection — the real model will learn
        nonlinear projections from the encoder features directly.
        """
        if len(self.vectors) == 0:
            return 0.0

        current = self.vectors[-1].scores[axis]
        vel = self.velocity(axis, window)
        projected = current + vel * steps_ahead

        return max(0.0, projected)  # Can't go below zero

    def project_all(self, steps_ahead: float = 1.0, window: int = 3) -> list[float]:
        """Project all axes forward."""
        if len(self.vectors) == 0:
            return []
        n_axes = len(self.vectors[-1].scores)
        return [self.project(i, steps_ahead, window) for i in range(n_axes)]

    def acceleration(self, axis: int, window: int = 4) -> float:
        """
        Second derivative — is the rate of change itself increasing?

        Acceleration matters: a fire spreading at constant speed is
        dangerous. A fire accelerating is catastrophic. The threshold
        should be lower when acceleration is positive.
        """
        if len(self.vectors) < 3:
            return 0.0

        recent = self.vectors[-window:]
        if len(recent) < 3:
            return 0.0

        mid = len(recent) // 2
        first_half = recent[:mid + 1]
        second_half = recent[mid:]

        dt1 = first_half[-1].timestamp - first_half[0].timestamp
        dt2 = second_half[-1].timestamp - second_half[0].timestamp

        if dt1 == 0 or dt2 == 0:
            return 0.0

        v1 = (first_half[-1].scores[axis] - first_half[0].scores[axis]) / dt1
        v2 = (second_half[-1].scores[axis] - second_half[0].scores[axis]) / dt2

        total_dt = recent[-1].timestamp - recent[0].timestamp
        if total_dt == 0:
            return 0.0

        return (v2 - v1) / total_dt
