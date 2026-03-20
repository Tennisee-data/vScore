"""
Dual-system demo: vScore (fast) + LLM (slow).

Shows when vScore handles things alone (clear threat, routine pattern)
and when it escalates to the LLM (ambiguity, novelty, deception).

Run: python -m vScore.demo_dual_system
"""

from vScore.core.valence import ValenceVector
from vScore.core.dual_system import SystemGate, ProcessingMode
from vScore.core.action_space import build_survival_actions


def main():
    print("=" * 70)
    print("Dual System: vScore (System 1) + LLM (System 2)")
    print("=" * 70)

    gate = SystemGate()
    af = build_survival_actions(
        ["seeking", "rage", "fear", "lust", "care", "panic", "play"]
    )

    scenarios = [
        {
            "name": "Loud bang nearby",
            "valence": [0, 0, 9, 0, 0, 2, 0],
            "conflict": 0.15,
            "surprise": 0.3,
            "prosody_mismatch": 0.0,
            "what_happens": "vScore: FLEE immediately. No time for words.",
        },
        {
            "name": "Empty room, quiet",
            "valence": [0.5, 0, 0, 0, 0, 0, 0.5],
            "conflict": 0.0,
            "surprise": 0.0,
            "prosody_mismatch": 0.0,
            "what_happens": "Nothing happening. Neither system activates.",
        },
        {
            "name": "Stranger says 'I'm friendly' in aggressive tone",
            "valence": [0, 3, 5, 0, 0, 2, 0],
            "conflict": 0.65,
            "surprise": 0.5,
            "prosody_mismatch": 0.8,
            "what_happens": "Prosody says threat. Words say safe. Engage LLM: who is this?",
        },
        {
            "name": "Child laughing, then suddenly crying",
            "valence": [0, 0, 2, 0, 7, 5, 0],
            "conflict": 0.71,
            "surprise": 0.6,
            "prosody_mismatch": 0.0,
            "what_happens": "CARE and PANIC compete. Engage LLM: what happened?",
        },
        {
            "name": "Known coworker greets you normally",
            "valence": [1, 0, 0, 0, 2, 0, 1],
            "conflict": 0.3,
            "surprise": 0.1,
            "prosody_mismatch": 0.0,
            "what_happens": "Routine social. vScore handles it: low magnitude, no threat.",
        },
        {
            "name": "Unfamiliar sound in the dark",
            "valence": [4, 0, 5, 0, 0, 3, 0],
            "conflict": 0.45,
            "surprise": 0.9,
            "prosody_mismatch": 0.0,
            "what_happens": "Novel pattern. vScore says ATTEND. Engage LLM: what is that?",
        },
        {
            "name": "Boss says 'great job' in flat, cold tone",
            "valence": [0, 1, 2, 0, 0, 1, 0],
            "conflict": 0.4,
            "surprise": 0.3,
            "prosody_mismatch": 0.7,
            "what_happens": "Words are positive. Voice is not. Deception/sarcasm detected.",
        },
        {
            "name": "Bear charging at you",
            "valence": [0, 0, 10, 0, 0, 3, 0],
            "conflict": 0.10,
            "surprise": 0.2,
            "prosody_mismatch": 0.0,
            "what_happens": "Maximum threat, zero ambiguity. vScore: FLEE. LLM irrelevant.",
        },
    ]

    mode_symbols = {
        ProcessingMode.VSCORE_ONLY: "  [vScore ONLY]",
        ProcessingMode.LLM_ASSIST: "  [→ LLM needed]",
        ProcessingMode.BOTH_AGREE: "  [Both: agree]",
        ProcessingMode.BOTH_CONFLICT: "  [Both: CONFLICT]",
    }

    for s in scenarios:
        v = ValenceVector("Survival", s["valence"])
        magnitude = sum(x*x for x in s["valence"]) ** 0.5

        mode, reason = gate.should_engage_llm(
            valence=v,
            action_conflict=s["conflict"],
            surprise=s["surprise"],
            prosody_semantic_mismatch=s["prosody_mismatch"],
        )

        actions = af.evaluate(v)
        top_action = actions[0][0] if actions else "REST"

        print(f"\n  {s['name']}")
        print(f"  magnitude={magnitude:.1f}  conflict={s['conflict']:.2f}  "
              f"surprise={s['surprise']:.1f}  prosody_mismatch={s['prosody_mismatch']:.1f}")
        print(f"  vScore action: {top_action}")
        print(f"  {mode_symbols[mode]}  {reason}")
        if s.get("what_happens"):
            print(f"  → {s['what_happens']}")

    print(f"""

{'=' * 70}
THE TWO SPEEDS OF INTELLIGENCE
{'=' * 70}

  FAST (vScore, ~50ms):
    Pattern → valence → threshold → act.
    No words. No reasoning. No deliberation.
    Handles: threats, routine, well-learned patterns.
    Sufficient for: 80%+ of moment-to-moment decisions.

  SLOW (LLM, ~500ms+):
    Tokens → parse → reason → plan → respond.
    Requires: words, context, memory retrieval.
    Handles: ambiguity, novelty, social complexity, deception.
    Engaged only when: vScore signals uncertainty.

  THE GATE:
    vScore gates the LLM, not the reverse.
    The fast system decides IF the slow system is needed.

    This is biologically real:
    - Amygdala (fast) gates prefrontal cortex (slow)
    - Startle response precedes conscious evaluation
    - "I flinched before I knew why" = vScore fired, LLM hadn't started

  THE BRIDGE (Prosody):
    Voice intonation is scored by vScore (acoustic pattern → valence).
    Word content is scored by LLM (tokens → semantics).
    When they disagree → deception, sarcasm, distress masked by words.

    "I'm fine" + trembling voice = prosody_mismatch > 0.5
    vScore catches what the words hide.

  TOGETHER:
    vScore provides: speed, pre-linguistic evaluation, modality fusion.
    LLM provides: reasoning, planning, social understanding, language.
    Neither alone is intelligence. Both together are.
""")


if __name__ == "__main__":
    main()
