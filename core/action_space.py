"""
Action inference from valence vectors.

The question: how do multiple axes combine to produce action?

A single axis is a scalar. Two axes define a plane. Three define
a volume. The full valence vector defines a point in an n-dimensional
space where regions map to qualitatively different responses.

The key insight: actions are not triggered by individual axes
crossing thresholds. They emerge from the GEOMETRY of the full
vector — its direction, its magnitude, and its trajectory through
valence space.

    [fear=8, rage=0] → flee
    [fear=0, rage=8] → fight
    [fear=6, rage=6] → freeze (conflicting drives, paralysis)
    [fear=8, care=8] → protect (defend offspring despite danger)

Same magnitude. Different directions. Completely different actions.

The action space is the mapping:
    valence vector (where am I?) +
    valence trajectory (where am I heading?) →
    action region (what should I do?)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from .valence import ValenceVector, ValenceTrajectory


@dataclass
class ActionRegion:
    """
    A named region in valence space that maps to a response.

    Defined by a condition function over the full valence vector.
    The condition considers the INTERACTION between axes, not
    individual thresholds.
    """
    name: str
    condition: Callable[[ValenceVector], float]  # Returns 0-1 activation
    priority: int = 0  # Higher priority wins ties

    def evaluate(self, v: ValenceVector) -> float:
        return self.condition(v)


@dataclass
class ActionField:
    """
    The full action space for a domain.

    A vector field over valence space: at every point in valence space,
    there is a recommended action (or set of competing actions with
    different activation levels).

    This is how biological systems work. There is no decision tree.
    There is a field of competing action tendencies, and the strongest
    one wins. When two are equally strong, the organism freezes —
    Buridan's donkey, biologically real.
    """

    domain_name: str
    axis_names: list[str]
    regions: list[ActionRegion] = field(default_factory=list)

    def add_region(self, region: ActionRegion):
        self.regions.append(region)

    def evaluate(self, v: ValenceVector) -> list[tuple[str, float]]:
        """
        Given a valence vector, return all action activations sorted
        by strength. The organism does the top one — unless two are
        close, in which case it hesitates.
        """
        activations = []
        for region in self.regions:
            strength = region.evaluate(v)
            if strength > 0.01:
                activations.append((region.name, strength, region.priority))

        # Sort by strength * priority weight
        activations.sort(key=lambda x: x[1] * (1 + x[2] * 0.1), reverse=True)
        return [(name, strength) for name, strength, _ in activations]

    def evaluate_trajectory(
        self, trajectory: ValenceTrajectory, horizon: float = 2.0
    ) -> dict:
        """
        Evaluate actions based on both current state AND projected state.

        This is the predictive layer. The organism doesn't just ask
        "what should I do now?" It asks "what should I do given where
        things are heading?"

        Returns current actions, projected actions, and whether the
        projected state demands a different response than the current one
        (which signals: act NOW before it's too late).
        """
        if trajectory.length == 0:
            return {"current": [], "projected": [], "preempt": False}

        current = trajectory.vectors[-1]

        # Project forward
        projected_scores = trajectory.project_all(steps_ahead=horizon)
        projected_scores = [max(0, s) for s in projected_scores]
        projected = ValenceVector(
            domain_name=current.domain_name,
            scores=projected_scores,
            timestamp=current.timestamp + horizon,
        )

        current_actions = self.evaluate(current)
        projected_actions = self.evaluate(projected)

        # Preemption: the projected state demands a higher-urgency
        # action than the current state. Act now.
        preempt = False
        if current_actions and projected_actions:
            if current_actions[0][0] != projected_actions[0][0]:
                if projected_actions[0][1] > current_actions[0][1]:
                    preempt = True

        return {
            "current": current_actions,
            "projected": projected_actions,
            "preempt": preempt,
            "preempt_reason": (
                f"projected {projected_actions[0][0]} "
                f"({projected_actions[0][1]:.2f}) overtakes "
                f"current {current_actions[0][0]} "
                f"({current_actions[0][1]:.2f})"
                if preempt else None
            ),
        }

    def conflict_level(self, v: ValenceVector) -> float:
        """
        How much conflict exists between competing actions.

        High conflict = the organism freezes / hesitates.
        Low conflict = clear action, fast response.

        Measured as the ratio between the top two action activations.
        Ratio near 1.0 = high conflict (equal competition).
        Ratio near 0.0 = no conflict (clear winner).
        """
        actions = self.evaluate(v)
        if len(actions) < 2:
            return 0.0

        top = actions[0][1]
        second = actions[1][1]

        if top == 0:
            return 0.0

        return second / top  # 0 = no conflict, 1 = deadlock


# ── Helper: build action fields from axis interactions ─────────

def _axis_score(v: ValenceVector, axis_names: list[str], name: str) -> float:
    """Get score by axis name."""
    idx = axis_names.index(name)
    return v.scores[idx]


def build_survival_actions(axis_names: list[str]) -> ActionField:
    """
    Survival domain action field.

    Actions emerge from axis INTERACTIONS, not individual thresholds.
    """
    af = ActionField(domain_name="Survival", axis_names=axis_names)

    def _s(v, name):
        return _axis_score(v, axis_names, name)

    # FLEE: fear dominant, rage low
    af.add_region(ActionRegion(
        name="FLEE",
        condition=lambda v: max(0, _s(v, "fear") - _s(v, "rage")) / 10.0
            * (_s(v, "fear") / 10.0),
        priority=3,
    ))

    # FIGHT: rage dominant, fear low
    af.add_region(ActionRegion(
        name="FIGHT",
        condition=lambda v: max(0, _s(v, "rage") - _s(v, "fear")) / 10.0
            * (_s(v, "rage") / 10.0),
        priority=3,
    ))

    # FREEZE: fear AND rage both high (competing drives)
    af.add_region(ActionRegion(
        name="FREEZE",
        condition=lambda v: min(_s(v, "fear"), _s(v, "rage")) / 10.0
            * (1 - abs(_s(v, "fear") - _s(v, "rage")) / 10.0),
        priority=2,
    ))

    # PROTECT: care AND fear both high (defend despite danger)
    af.add_region(ActionRegion(
        name="PROTECT",
        condition=lambda v: min(_s(v, "care"), _s(v, "fear")) / 10.0
            * (_s(v, "care") / 10.0),
        priority=4,
    ))

    # EXPLORE: seeking high, fear low
    af.add_region(ActionRegion(
        name="EXPLORE",
        condition=lambda v: _s(v, "seeking") / 10.0
            * max(0, 1 - _s(v, "fear") / 5.0),
        priority=1,
    ))

    # BOND: care high, panic low (nurturing)
    af.add_region(ActionRegion(
        name="BOND",
        condition=lambda v: _s(v, "care") / 10.0
            * max(0, 1 - _s(v, "panic") / 8.0)
            * max(0, 1 - _s(v, "fear") / 5.0),
        priority=1,
    ))

    # SEEK_CONTACT: panic high (separation distress → find others)
    af.add_region(ActionRegion(
        name="SEEK_CONTACT",
        condition=lambda v: _s(v, "panic") / 10.0
            * max(0, 1 - _s(v, "fear") / 8.0),
        priority=2,
    ))

    # PLAY: play high, fear and rage both low
    af.add_region(ActionRegion(
        name="PLAY",
        condition=lambda v: _s(v, "play") / 10.0
            * max(0, 1 - _s(v, "fear") / 3.0)
            * max(0, 1 - _s(v, "rage") / 3.0),
        priority=0,
    ))

    # REST: everything near zero — homeostasis
    af.add_region(ActionRegion(
        name="REST",
        condition=lambda v: max(0, 1 - sum(v.scores) / 15.0),
        priority=0,
    ))

    return af


def build_fire_actions(axis_names: list[str]) -> ActionField:
    """
    Fire domain action field.

    A firefighter's decisions emerge from axis interactions.
    """
    af = ActionField(domain_name="Fire", axis_names=axis_names)

    def _s(v, name):
        return _axis_score(v, axis_names, name)

    # HOLD_POSITION: contained, low risk
    af.add_region(ActionRegion(
        name="HOLD_POSITION",
        condition=lambda v: _s(v, "containment") / 10.0
            * max(0, 1 - _s(v, "structural_risk") / 5.0)
            * max(0, 1 - _s(v, "escape_route") / 5.0),
        priority=1,
    ))

    # ADVANCE: spread manageable, containment possible
    af.add_region(ActionRegion(
        name="ADVANCE",
        condition=lambda v: max(0, _s(v, "containment") - _s(v, "spread_rate")) / 10.0
            * max(0, 1 - _s(v, "structural_risk") / 7.0),
        priority=2,
    ))

    # DEFENSIVE: losing containment but escape still viable
    af.add_region(ActionRegion(
        name="DEFENSIVE",
        condition=lambda v: max(0, _s(v, "spread_rate") - _s(v, "containment")) / 10.0
            * max(0, 1 - _s(v, "escape_route") / 7.0),
        priority=2,
    ))

    # EVACUATE: escape route closing AND structural risk high
    af.add_region(ActionRegion(
        name="EVACUATE",
        condition=lambda v: _s(v, "escape_route") / 10.0
            * max(_s(v, "structural_risk"), _s(v, "smoke_toxicity")) / 10.0,
        priority=4,
    ))

    # MAYDAY: escape route nearly gone, structural collapse imminent
    af.add_region(ActionRegion(
        name="MAYDAY",
        condition=lambda v: max(0, (_s(v, "escape_route") - 7) / 3.0)
            * max(0, (_s(v, "structural_risk") - 6) / 4.0),
        priority=5,
    ))

    # VENTILATE: smoke toxicity high but structure stable
    af.add_region(ActionRegion(
        name="VENTILATE",
        condition=lambda v: _s(v, "smoke_toxicity") / 10.0
            * max(0, 1 - _s(v, "structural_risk") / 6.0)
            * max(0, 1 - _s(v, "spread_rate") / 7.0),
        priority=1,
    ))

    # MONITOR: everything low
    af.add_region(ActionRegion(
        name="MONITOR",
        condition=lambda v: max(0, 1 - sum(v.scores) / 20.0),
        priority=0,
    ))

    return af
