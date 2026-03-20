"""
Survival domain — Panksepp's 7 primal emotional circuits.

These are the phylogenetically oldest scoring axes. Every mammalian
brain runs this domain continuously. The axes are pre-linguistic,
pre-cortical, operating at brainstem/limbic level.
"""

from ..core.metaclass import DomainBase


class Survival(DomainBase):
    axes = [
        "seeking",   # Curiosity, drive, exploration. Activated by novelty.
        "rage",      # Boundary violation, frustration, blocked goals.
        "fear",      # Threat detection, escape urgency.
        "lust",      # Reproductive drive, attraction signals.
        "care",      # Nurture/protect activation, vulnerability detection.
        "panic",     # Separation, loss, isolation signals.
        "play",      # Social exploration, safe boundary-testing.
    ]
