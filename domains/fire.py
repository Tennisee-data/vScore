"""
Fire domain — visual scoring axes for fire behavior.

An expert firefighter reads a blaze and scores it instantly without
words. The pattern of flame color, smoke density, spread direction —
all map to outcome axes before any verbal assessment.
"""

from ..core.metaclass import DomainBase


class Fire(DomainBase):
    axes = [
        "spread_rate",     # How fast the fire is expanding.
        "proximity",       # Distance to assets/people/egress.
        "intensity",       # Heat output, flame height, color.
        "containment",     # Degree to which boundaries hold.
        "escape_route",    # Viability of egress paths.
        "structural_risk", # Collapse indicators in the visual field.
        "smoke_toxicity",  # Smoke color/density as proxy for composition.
    ]
