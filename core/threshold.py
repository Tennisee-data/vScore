"""
Threshold triggering system.

Biological systems don't classify — they trigger. The response is
nonlinear: sub-threshold cues are ignored, super-threshold cues
demand immediate action. The threshold itself is dynamic: it lowers
when acceleration is positive (things are getting worse fast) and
raises when the trajectory is returning to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .valence import ValenceTrajectory


class Response(Enum):
    IGNORE = 0     # Below threshold, no activation cost
    ATTEND = 1     # Near threshold, allocate attention
    ACT = 2        # Above threshold, immediate response required


@dataclass
class ThresholdConfig:
    """Per-axis threshold configuration."""
    axis_name: str
    trigger_level: float = 7.0        # Score at which ACT fires
    attention_level: float = 4.0      # Score at which ATTEND fires
    acceleration_weight: float = 1.0  # How much acceleration lowers threshold
    projection_horizon: float = 2.0   # How far ahead to project


@dataclass
class TriggerEvent:
    """A threshold crossing — the system's output."""
    domain_name: str
    axis: int
    axis_name: str
    current_score: float
    projected_score: float
    velocity: float
    acceleration: float
    response: Response
    timestamp: float


@dataclass
class ThresholdTrigger:
    """
    Evaluates a valence trajectory against thresholds and fires triggers.

    The key insight: thresholds are not fixed. They adapt based on
    the dynamics of the trajectory. A slowly rising score needs to
    reach the full threshold. A rapidly accelerating score triggers
    earlier — because waiting is costly.

    effective_threshold = base_threshold - (acceleration * weight)
    """

    domain_name: str
    axis_names: list[str]
    configs: dict[str, ThresholdConfig] = field(default_factory=dict)

    def __post_init__(self):
        for name in self.axis_names:
            if name not in self.configs:
                self.configs[name] = ThresholdConfig(axis_name=name)

    def evaluate(self, trajectory: ValenceTrajectory) -> list[TriggerEvent]:
        """
        Evaluate all axes against their thresholds.
        Returns trigger events for any axis at ATTEND or ACT level.
        """
        if trajectory.length == 0:
            return []

        current = trajectory.vectors[-1]
        events = []

        for i, axis_name in enumerate(self.axis_names):
            config = self.configs[axis_name]

            score = current.scores[i]
            vel = trajectory.velocity(i)
            acc = trajectory.acceleration(i)
            projected = trajectory.project(i, config.projection_horizon)

            # Dynamic threshold: lower when accelerating toward danger
            accel_adjustment = max(0, acc) * config.acceleration_weight
            effective_trigger = max(1.0, config.trigger_level - accel_adjustment)
            effective_attention = max(0.5, config.attention_level - accel_adjustment * 0.5)

            # Evaluate against both current score and projected score
            eval_score = max(score, projected)

            if eval_score >= effective_trigger:
                response = Response.ACT
            elif eval_score >= effective_attention:
                response = Response.ATTEND
            else:
                continue  # IGNORE — no event generated

            events.append(TriggerEvent(
                domain_name=self.domain_name,
                axis=i,
                axis_name=axis_name,
                current_score=score,
                projected_score=projected,
                velocity=vel,
                acceleration=acc,
                response=response,
                timestamp=current.timestamp,
            ))

        return events
