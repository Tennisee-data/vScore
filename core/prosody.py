"""
Prosodic valence — voice intonation scoring.

The bridge between vScore and language.

Words carry semantic content (LLM territory).
Voice carries PROSODIC content (vScore territory):
    - Pitch contour: rising = question/uncertainty, falling = assertion
    - Volume: loud = threat/urgency, soft = intimacy/secrecy
    - Tempo: fast = excitement/panic, slow = calm/authority
    - Roughness: harsh = anger/threat, smooth = trust/care
    - Rhythm: regular = calm, irregular = distress

A sentence like "everything is fine" carries opposite meaning
depending on prosody:
    - Flat, calm, low pitch → truly fine → valence near zero
    - High pitch, trembling, fast → lying/distressed → fear/panic high

The LLM processes the words. vScore processes the voice.
Together they detect the lie.

This is not sentiment analysis. Sentiment analysis assigns
a word ("positive", "negative") to text. Prosodic valence
assigns NUMBERS to the acoustic envelope — the melody of
speech, stripped of all semantic content. A language you
don't speak still conveys anger, joy, fear, authority
through prosody alone.
"""

from ..core.metaclass import DomainBase


class Prosody(DomainBase):
    """
    Voice intonation scoring — what the voice says
    independent of the words.

    A baby responds to prosody before it understands a single word.
    An adult in a foreign country reads prosody to gauge intent.
    A dog responds entirely to prosody, never to semantics.

    These axes score the VOICE, not the WORDS.
    """
    axes = [
        "threat",          # Harsh, loud, abrupt. Raised voice, growl.
        "authority",       # Low pitch, slow tempo, measured pauses.
        "distress",        # High pitch, trembling, irregular rhythm.
        "warmth",          # Soft, melodic, smooth. Motherese.
        "urgency",         # Fast tempo, rising pitch, short phrases.
        "deception",       # Pitch/rhythm mismatch with content valence.
        "arousal",         # Overall energy in the vocal signal.
    ]
