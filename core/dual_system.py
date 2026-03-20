"""
Dual-system architecture: vScore + LLM.

Two systems. Two speeds. Two purposes.

System 1 (vScore): Pre-linguistic. Milliseconds. Pattern → valence → action.
System 2 (LLM):    Linguistic. Seconds. Tokens → meaning → reasoning.

They are NOT competing. They are complementary:

    ┌──────────────────────────────────────────────────┐
    │                    STIMULUS                      │
    │              (visual + auditory)                  │
    └────────────────┬───────────────┬─────────────────┘
                     │               │
              ┌──────┴──────┐ ┌──────┴──────┐
              │  System 1   │ │  System 2   │
              │  (vScore)   │ │  (LLM)      │
              │             │ │             │
              │  Pixels →   │ │  Tokens →   │
              │  Waveforms →│ │  Words →    │
              │  Prosody →  │ │  Grammar →  │
              │             │ │             │
              │  Valence    │ │  Semantics  │
              │  vector     │ │  + reasoning│
              │             │ │             │
              │  50ms       │ │  500ms+     │
              └──────┬──────┘ └──────┬──────┘
                     │               │
              ┌──────┴───────────────┴──────┐
              │       INTEGRATION           │
              │                             │
              │  vScore gates attention:    │
              │    "is this urgent?"        │
              │    "which modality?"        │
              │    "threat level?"          │
              │                             │
              │  LLM provides context:     │
              │    "what does this mean?"   │
              │    "what are my options?"   │
              │    "what happened before?"  │
              │                             │
              │  Together:                  │
              │    valence + semantics =    │
              │    situated understanding   │
              └─────────────────────────────┘

When do you NEED the LLM?
    - When valence alone is ambiguous
    - When the action requires planning (not just reacting)
    - When social context matters (who said it, why, to whom)
    - When the situation is novel AND non-urgent

When is vScore SUFFICIENT?
    - When speed matters more than accuracy
    - When the pattern is well-learned (strong prior)
    - When the valence is unambiguous (high activation, low conflict)
    - When language is unavailable (foreign country, noisy, nonverbal)

The critical insight: vScore can GATE the LLM.
    - Low valence magnitude → don't bother the LLM (nothing happening)
    - High valence + low conflict → act on vScore alone (clear threat)
    - High valence + high conflict → engage the LLM (need reasoning)
    - Prosody/semantics mismatch → engage the LLM (potential deception)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .valence import ValenceVector


class ProcessingMode(Enum):
    """Which system handles this stimulus."""
    VSCORE_ONLY = auto()    # Fast path: act on valence alone
    LLM_ASSIST = auto()     # Slow path: need linguistic reasoning
    BOTH_AGREE = auto()     # Both systems engaged, conclusions aligned
    BOTH_CONFLICT = auto()  # Both systems engaged, conclusions differ


@dataclass
class SystemGate:
    """
    Determines when to engage the LLM.

    vScore is always running (it's fast and cheap).
    The LLM is engaged only when vScore signals uncertainty
    or when the situation requires linguistic reasoning.

    This mirrors Kahneman's System 1 / System 2, but grounded
    in specific computational mechanisms rather than metaphor.
    """
    # Thresholds for gating
    urgency_threshold: float = 7.0     # Above this: vScore acts alone, no time for LLM
    ambiguity_threshold: float = 0.6   # Conflict above this: engage LLM for disambiguation
    novelty_threshold: float = 0.8     # Surprise above this: engage LLM for reasoning
    magnitude_floor: float = 2.0       # Below this: nothing happening, ignore

    def should_engage_llm(
        self,
        valence: ValenceVector,
        action_conflict: float,
        surprise: float,
        prosody_semantic_mismatch: float = 0.0,
    ) -> tuple[ProcessingMode, str]:
        """
        Gate decision: does the LLM need to be involved?

        Returns (mode, reason).
        """
        magnitude = sum(s * s for s in valence.scores) ** 0.5

        # Nothing happening → ignore everything
        if magnitude < self.magnitude_floor:
            return ProcessingMode.VSCORE_ONLY, "sub-threshold, nothing happening"

        # Clear urgent threat → vScore acts alone, no time for words
        max_score = max(valence.scores)
        if max_score >= self.urgency_threshold and action_conflict < self.ambiguity_threshold:
            return ProcessingMode.VSCORE_ONLY, "unambiguous urgency, act now"

        # Prosody contradicts semantics → engage LLM (deception detection)
        if prosody_semantic_mismatch > 0.5:
            return ProcessingMode.BOTH_CONFLICT, "prosody/semantics mismatch — possible deception"

        # High conflict between actions → engage LLM for reasoning
        if action_conflict > self.ambiguity_threshold:
            return ProcessingMode.LLM_ASSIST, "ambiguous situation, need reasoning"

        # Novel situation → engage LLM for context
        if surprise > self.novelty_threshold:
            return ProcessingMode.LLM_ASSIST, "novel pattern, need context"

        # Default: vScore handles it
        return ProcessingMode.VSCORE_ONLY, "routine, well-learned pattern"
