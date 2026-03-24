"""
Tennis domain — pre-linguistic scoring of tennis shot dynamics.

A coach watching a rally does not think "forehand flat." They see
the racquet path, the body rotation, the ball trajectory, and
score it: fast, aggressive, low precision, high impact. The word
for the shot comes after the evaluation, not before.

Two visually similar shots (forehand flat and forehand open stance)
should produce similar valence vectors without being told they
share a name. If they do, the encoder captures the visual dynamics
that make them similar, not the label.
"""

from ..core.metaclass import DomainBase


class Tennis(DomainBase):
    axes = [
        "speed",          # Racquet and ball speed. Flat serve vs. slice.
        "impact",         # Force transfer at contact. Smash vs. drop shot.
        "precision",      # Placement accuracy. Slice vs. flat power shot.
        "verticality",    # Upward/downward motion. Serve toss, smash, lob.
        "aggression",     # Intent to win the point now. Winner attempt vs. rally.
        "tension",        # Buildup before contact. Service motion, wind-up.
    ]
