"""
Hockey domain — visual scoring axes for game state.

A coach doesn't think "the left winger is skating at 30km/h toward
the slot with the defenseman 2 meters behind." They see a pattern
and score it: high threat, breakaway forming, momentum shifting.
"""

from ..core.metaclass import DomainBase


class Hockey(DomainBase):
    axes = [
        "scoring_threat",       # Probability of a goal from current play pattern.
        "possession_pressure",  # How contested/stable possession is.
        "breakaway",            # Open-ice advantage developing.
        "penalty_risk",         # Body positions suggesting infraction.
        "momentum",             # Shift in territorial control.
        "fatigue",              # Visual cues of player deceleration.
    ]
