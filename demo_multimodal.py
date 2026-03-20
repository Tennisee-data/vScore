"""
Multimodal valence demo.

Shows that the same scoring mechanism works across modalities,
and that fusion in VALENCE space (not feature space) produces
diagnostic information about modality agreement/conflict.

Run: python -m vScore.demo_multimodal
"""

import torch
from vScore.core.multimodal import Modality, ValenceFusion
from vScore.core.valence import ValenceVector
from vScore.core.action_space import build_survival_actions


def main():
    print("=" * 70)
    print("vScore Multimodal — Same Axes, Different Sensors")
    print("=" * 70)

    af = build_survival_actions(
        ["seeking", "rage", "fear", "lust", "care", "panic", "play"]
    )
    axis_names = ["seeking", "rage", "fear", "lust", "care", "panic", "play"]

    scenarios = [
        {
            "name": "Bear in the forest",
            "description": "You see a large dark shape moving AND hear a growl",
            "vision": [2.0, 0.0, 7.0, 0.0, 0.0, 1.0, 0.0],
            "audio":  [0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            "expected": "Both modalities agree: FEAR. High confidence → FLEE",
        },
        {
            "name": "Occluded threat",
            "description": "You hear a growl but see only trees",
            "vision": [3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "audio":  [0.0, 0.0, 8.0, 0.0, 0.0, 2.0, 0.0],
            "expected": "Audio threat >> visual → search for source. High conflict.",
        },
        {
            "name": "Baby crying",
            "description": "You hear distress cry. You see the baby is safe.",
            "vision": [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            "audio":  [0.0, 0.0, 2.0, 0.0, 8.0, 7.0, 0.0],
            "expected": "Audio: panic+care. Vision: mild care. → SEEK_CONTACT.",
        },
        {
            "name": "Locomotive approaching",
            "description": "Rumble grows louder. Track vibration. Mass approaching.",
            "vision": [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            "audio":  [0.0, 0.0, 9.0, 0.0, 0.0, 3.0, 0.0],
            "expected": "Audio scores threat higher (sound arrives first). MAX → FLEE.",
        },
        {
            "name": "Campfire with guitar",
            "description": "Warm fire, soft music, familiar voices.",
            "vision": [1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0],
            "audio":  [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 5.0],
            "expected": "Both say care+play, near zero threat. → REST/PLAY.",
        },
        {
            "name": "Thunderstorm",
            "description": "Dark sky, flash of lightning. CRACK of thunder.",
            "vision": [0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0],
            "audio":  [0.0, 0.0, 9.0, 0.0, 0.0, 4.0, 0.0],
            "expected": "Thunder (audio) is more threatening than lightning (visual).",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"  {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'─' * 70}")

        v_scores = torch.tensor(scenario["vision"])
        a_scores = torch.tensor(scenario["audio"])

        modality_scores = {
            Modality.VISION: v_scores,
            Modality.AUDIO: a_scores,
        }

        # Per-modality display
        print(f"\n  {'':>8s}  ", end="")
        for ax in axis_names:
            print(f"{ax[:4]:>5s}", end="")
        print()

        print(f"  {'VISION':>8s}  ", end="")
        for s in v_scores:
            print(f"{s.item():5.1f}", end="")
        print()

        print(f"  {'AUDIO':>8s}  ", end="")
        for s in a_scores:
            print(f"{s.item():5.1f}", end="")
        print()

        # Fusion methods
        fused_max = ValenceFusion.max_fusion(modality_scores)
        fused_mean = ValenceFusion.mean_fusion(modality_scores)
        conflict_per_axis, conflict_overall = ValenceFusion.modality_conflict(modality_scores)

        print(f"\n  {'MAX':>8s}  ", end="")
        for s in fused_max:
            print(f"{s.item():5.1f}", end="")
        print(f"  (if ANY sense says danger, respond)")

        print(f"  {'MEAN':>8s}  ", end="")
        for s in fused_mean:
            print(f"{s.item():5.1f}", end="")
        print(f"  (cross-modal average)")

        print(f"  {'CONFLICT':>8s}  ", end="")
        for s in conflict_per_axis:
            print(f"{s.item():5.1f}", end="")
        print(f"  overall={conflict_overall:.2f}")

        # Action from MAX fusion (survival default)
        v_fused = ValenceVector("Survival", fused_max.tolist())
        actions = af.evaluate(v_fused)
        conflict_action = af.conflict_level(v_fused)

        if actions:
            top = actions[0]
            print(f"\n  → ACTION: {top[0]} ({top[1]:.2f})"
                  f"  decision_conflict={conflict_action:.2f}")
            if len(actions) > 1:
                print(f"    runner-up: {actions[1][0]} ({actions[1][1]:.2f})")

        # Diagnostic: modality agreement
        if conflict_overall > 2.0:
            # Find which axis has highest conflict
            max_conflict_axis = conflict_per_axis.argmax().item()
            print(f"\n  ⚠ HIGH MODALITY CONFLICT on {axis_names[max_conflict_axis]}: "
                  f"vision={v_scores[max_conflict_axis]:.1f} vs "
                  f"audio={a_scores[max_conflict_axis]:.1f}")
            if a_scores[max_conflict_axis] > v_scores[max_conflict_axis]:
                print(f"    Audio threat exceeds visual → source may be OCCLUDED or APPROACHING")
            else:
                print(f"    Visual threat exceeds audio → threat is VISIBLE but SILENT (stalking?)")

    print(f"""

{'=' * 70}
THE UNIVERSAL PRINCIPLE
{'=' * 70}

  The valence vector is the universal interface.

  Vision encodes: pixels → spatial features → valence vector
  Audio encodes:  waveform → spectral features → valence vector
  Touch would encode: pressure → somatosensory features → valence vector

  Downstream of the valence vector, NOTHING knows which sense
  produced the scores. The action space, the trajectory projector,
  the threshold trigger, the Bayesian memory — they all operate
  on the same numerical vector regardless of source.

  This is not a design choice. It is a claim about intelligence:
  the organism's evaluation system is modality-agnostic.
  The amygdala processes threat from vision AND audition.
  The same FEAR circuit fires for a seen predator and a heard growl.

  The word "locomotive" is irrelevant.
  The sound pattern + visual pattern → valence vector → MOVE.
  That's all. That's intelligence.

  Modality fusion happens in VALENCE space, not feature space.
  This means modality conflict is observable:
    - Audio threat >> visual threat = occluded danger
    - Visual threat >> audio threat = silent stalker
    - Both agree = high confidence, fast response

  The fusion rule is itself a domain parameter:
    - Survival default: MAX (if ANY sense says danger, flee)
    - Diagnostic: CONFLICT detection (where do senses disagree?)
    - Learned: confidence-weighted (trust the experienced sense)
""")


if __name__ == "__main__":
    main()
