"""
Weather domain — visual scoring axes for atmospheric conditions.

A sailor reads the sky. A farmer reads the clouds. The visual
patterns — color, movement, texture of cloud formations, light
quality — map directly to outcome predictions without language.
"""

from ..core.metaclass import DomainBase


class Weather(DomainBase):
    axes = [
        "wind",              # Movement patterns in clouds, trees, water.
        "precipitation",     # Visual cues for incoming rain/snow/hail.
        "temperature_delta", # Rate of thermal change (light quality, haze).
        "visibility",        # How far you can see, degradation rate.
        "structural_risk",   # Threat to buildings, trees, infrastructure.
        "electrical",        # Lightning probability from cloud patterns.
    ]
