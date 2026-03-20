"""
Music domain — affective scoring of musical patterns.

Music is the purest test case for pre-linguistic valence scoring.
There are no words, no objects, no threats — just sound patterns
that reliably produce valence responses across cultures.

A minor chord doesn't need a label to produce tension.
A resolved cadence doesn't need a name to produce relief.
The valence IS the music. The theory is post-hoc.
"""

from ..core.metaclass import DomainBase


class Music(DomainBase):
    axes = [
        "tension",          # Harmonic instability, unresolved expectation.
        "release",          # Resolution, arrival, cadential closure.
        "energy",           # Dynamic level, density, rhythmic drive.
        "intimacy",         # Sparse, close, soft — vs. distant/massive.
        "novelty",          # Deviation from established pattern.
        "groove",           # Rhythmic entrainment strength.
        "melancholy",       # Minor-mode gravity, descending contours.
    ]
