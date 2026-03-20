"""
Industrial sound domain — machine/environment auditory scoring.

A factory worker doesn't think "the bearing on turbine 3 is
developing a 47Hz harmonic." They hear the sound change and
score it: something is wrong, it's getting worse, that machine.

The expert's ear is a calibrated valence head over spectral
features — trained by years of outcomes, not vocabulary.
"""

from ..core.metaclass import DomainBase


class IndustrialSound(DomainBase):
    axes = [
        "anomaly",         # Deviation from normal operating sound.
        "degradation",     # Progressive change suggesting wear/failure.
        "impact_event",    # Sudden transient — something broke/hit/fell.
        "proximity",       # Distance to source.
        "resonance_shift", # Change in modal frequencies — structural change.
        "load_stress",     # Sound patterns indicating strain/overload.
    ]
