"""
Sound domain — pre-linguistic auditory scoring axes.

A baby doesn't know the word "explosion." It hears a sudden,
loud, broadband impulse and scores it: high threat, high proximity,
high urgency. The word comes years later. The flinch comes in
milliseconds.

The axes here are not acoustic measurements (dB, Hz, spectral
centroid). They are OUTCOME-RELEVANT evaluations: what does this
sound mean for the organism? The same 90dB can be threat (gunshot)
or reward (concert climax). The difference is in the pattern,
not the amplitude.
"""

from ..core.metaclass import DomainBase


class Sound(DomainBase):
    axes = [
        "threat",          # How dangerous does this sound? Sudden, loud, sharp.
        "proximity",       # How close is the source? Amplitude + reverb ratio.
        "urgency",         # How fast must I respond? Onset speed, rhythm.
        "familiarity",     # Have I heard this pattern before? Deviation from prior.
        "social_signal",   # Is this communication? Voice-like spectral structure.
        "rhythmic_pull",   # Does this entrain me? Regularity, beat, periodicity.
        "dissonance",      # How much spectral conflict? Roughness, beating.
    ]
