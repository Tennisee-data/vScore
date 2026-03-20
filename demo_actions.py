"""
Action inference demo.

Shows how the same magnitude produces different actions depending
on the DIRECTION in valence space. And how trajectory projection
triggers preemptive action.

Run: python -m vScore.demo_actions
"""

from vScore.core.valence import ValenceVector, ValenceTrajectory
from vScore.core.action_space import build_survival_actions, build_fire_actions


def print_state(label, v, actions, conflict):
    scores_str = "  ".join(f"{s:.0f}" for s in v.scores)
    mag = sum(s * s for s in v.scores) ** 0.5
    print(f"\n  {label}")
    print(f"  vector: [{scores_str}]  magnitude: {mag:.1f}  conflict: {conflict:.2f}")
    for name, strength in actions[:3]:
        bar = "#" * int(strength * 30)
        print(f"    {name:>15s}  {strength:.2f}  {bar}")


def survival_demo():
    print("=" * 65)
    print("SURVIVAL: Same magnitude, different directions, different actions")
    print("=" * 65)
    print("  axes: seeking  rage  fear  lust  care  panic  play")

    af = build_survival_actions(
        ["seeking", "rage", "fear", "lust", "care", "panic", "play"]
    )

    scenarios = [
        ("Predator spotted (fear dominant)",
         [0, 0, 8, 0, 0, 0, 0]),
        ("Territory invaded (rage dominant)",
         [0, 8, 0, 0, 0, 0, 0]),
        ("Cornered (fear + rage = FREEZE)",
         [0, 7, 7, 0, 0, 0, 0]),
        ("Mother with cub, predator near (care + fear = PROTECT)",
         [0, 0, 8, 0, 8, 0, 0]),
        ("Safe meadow (seeking + play)",
         [6, 0, 0, 0, 0, 0, 5]),
        ("Lost offspring (panic dominant)",
         [0, 0, 0, 0, 3, 8, 0]),
        ("Sleeping (homeostasis)",
         [0, 0, 0, 0, 0, 0, 0]),
    ]

    for label, scores in scenarios:
        v = ValenceVector("Survival", scores)
        actions = af.evaluate(v)
        conflict = af.conflict_level(v)
        print_state(label, v, actions, conflict)


def fire_demo():
    print("\n\n" + "=" * 65)
    print("FIRE: Trajectory changes action BEFORE the crisis arrives")
    print("=" * 65)
    print("  axes: spread  prox  intens  contain  escape  struct  smoke")

    af = build_fire_actions(
        ["spread_rate", "proximity", "intensity", "containment",
         "escape_route", "structural_risk", "smoke_toxicity"]
    )

    # A fire that deteriorates over time
    trajectory = ValenceTrajectory(domain_name="Fire")
    #                    spread prox  int  contain escape struct smoke
    timeline = [
        (0.0,  "Initial assessment",
         [2.0, 2.0, 2.0, 7.0, 1.0, 1.0, 1.0]),
        (1.0,  "Spreading, still contained",
         [3.5, 2.5, 3.0, 6.0, 1.5, 1.5, 2.0]),
        (2.0,  "Containment breaking down",
         [5.0, 3.5, 4.5, 4.0, 2.5, 2.5, 3.5]),
        (3.0,  "Losing control",
         [6.5, 5.0, 6.0, 2.5, 4.0, 4.0, 5.0]),
        (4.0,  "Structure compromised",
         [7.5, 6.5, 7.5, 1.5, 6.0, 6.5, 7.0]),
        (5.0,  "Collapse imminent",
         [8.5, 8.0, 9.0, 0.5, 8.0, 8.5, 8.5]),
    ]

    for t, label, scores in timeline:
        v = ValenceVector("Fire", scores, timestamp=t)
        trajectory.append(v)

        current_actions = af.evaluate(v)
        conflict = af.conflict_level(v)

        result = af.evaluate_trajectory(trajectory, horizon=2.0)

        print(f"\n  t={t:.0f}  {label}")
        scores_str = "  ".join(f"{s:.1f}" for s in scores)
        print(f"  vector: [{scores_str}]")

        print(f"  NOW:      ", end="")
        for name, strength in result["current"][:2]:
            print(f"{name}({strength:.2f})  ", end="")
        print()

        print(f"  t+2:      ", end="")
        for name, strength in result["projected"][:2]:
            print(f"{name}({strength:.2f})  ", end="")
        print()

        if result["preempt"]:
            print(f"  >>> PREEMPT: {result['preempt_reason']}")

        print(f"  conflict: {conflict:.2f}", end="")
        if conflict > 0.7:
            print("  [HIGH — competing actions, hesitation risk]", end="")
        print()


def geometry_demo():
    print("\n\n" + "=" * 65)
    print("GEOMETRY: Why direction matters more than magnitude")
    print("=" * 65)

    af = build_survival_actions(
        ["seeking", "rage", "fear", "lust", "care", "panic", "play"]
    )

    # Three vectors with IDENTICAL magnitude but different actions
    import math
    vectors = [
        ("Pure fear",            [0, 0, 10, 0, 0, 0, 0]),
        ("Pure rage",            [0, 10, 0, 0, 0, 0, 0]),
        ("Pure care",            [0, 0, 0, 0, 10, 0, 0]),
        ("Distributed low",      [3.8, 3.8, 3.8, 3.8, 3.8, 3.8, 3.8]),  # ~10.0 magnitude
    ]

    print(f"\n  All vectors have magnitude ~10.0")
    print(f"  Same energy, completely different behavior:\n")

    for label, scores in vectors:
        v = ValenceVector("Survival", scores)
        mag = sum(s*s for s in scores) ** 0.5
        actions = af.evaluate(v)
        conflict = af.conflict_level(v)

        top_action = actions[0][0] if actions else "REST"
        top_strength = actions[0][1] if actions else 0

        print(f"  {label:>20s}  mag={mag:5.1f}  →  {top_action:<15s} "
              f"({top_strength:.2f})  conflict={conflict:.2f}")

    print(f"""
  The classifier says: "high activation" for all four.
  The valence scorer says: flee / fight / nurture / anxious vigilance.
  Same energy. Four completely different organisms.

  Magnitude is arousal. Direction is meaning.
  A 10-dimensional classifier needs 10 thresholds.
  A 10-dimensional valence space has infinite directions —
  each one a different way of being activated, a different
  reason to act, a different future to project.
  """)


def main():
    survival_demo()
    fire_demo()
    geometry_demo()


if __name__ == "__main__":
    main()
