"""
vScore demo — the full pipeline without a real encoder.

Simulates what happens when a fire video is scored:
    pixels → encoder → valence scores → trajectory → projection → trigger

Run: python -m vScore.demo
"""

from vScore.core import ValenceVector, ValenceTrajectory, ThresholdTrigger, ScoredDomain
from vScore.core.threshold import ThresholdConfig, Response
from vScore.domains import Fire


def main():
    print("=" * 60)
    print("vScore — Visual Scoring Intelligence")
    print("=" * 60)
    print()

    # Show all registered domains
    print("Registered domains:")
    for name in ScoredDomain.list_domains():
        domain = ScoredDomain.get_domain(name)
        print(f"  {domain.describe()}")
    print()

    # Simulate a fire scenario
    # A fire that starts small and accelerates
    print("-" * 60)
    print("Scenario: Warehouse fire — scoring over time")
    print(f"Axes: {Fire.axis_names()}")
    print(f"Homeostasis: {Fire.zero_state()}")
    print("-" * 60)
    print()

    trajectory = ValenceTrajectory(domain_name="Fire")

    # Simulated scores over time (what a trained encoder would output)
    # spread_rate, proximity, intensity, containment, escape_route, structural_risk, smoke_toxicity
    frames = [
        (0.0,  [1.0, 2.0, 1.5, 8.0, 1.0, 0.5, 1.0]),  # Small, contained
        (1.0,  [2.0, 2.0, 2.5, 7.0, 1.5, 1.0, 2.0]),  # Growing
        (2.0,  [3.5, 3.0, 4.0, 5.5, 2.5, 2.0, 3.5]),  # Spreading
        (3.0,  [5.0, 4.0, 5.5, 4.0, 3.5, 3.0, 5.0]),  # Losing containment
        (4.0,  [7.0, 5.5, 7.0, 2.5, 5.0, 5.0, 6.5]),  # Accelerating
        (5.0,  [8.5, 7.0, 8.5, 1.5, 7.0, 7.0, 8.0]),  # Critical
    ]

    # Set up threshold trigger
    trigger = ThresholdTrigger(
        domain_name="Fire",
        axis_names=Fire.axis_names(),
        configs={
            "spread_rate": ThresholdConfig("spread_rate", trigger_level=7.0, attention_level=4.0),
            "proximity": ThresholdConfig("proximity", trigger_level=6.0, attention_level=3.5),
            "intensity": ThresholdConfig("intensity", trigger_level=7.0, attention_level=4.0),
            "containment": ThresholdConfig("containment", trigger_level=6.0, attention_level=4.0),
            "escape_route": ThresholdConfig("escape_route", trigger_level=5.0, attention_level=3.0),
            "structural_risk": ThresholdConfig("structural_risk", trigger_level=5.0, attention_level=3.0),
            "smoke_toxicity": ThresholdConfig("smoke_toxicity", trigger_level=6.0, attention_level=3.5),
        },
    )

    for t, scores in frames:
        v = ValenceVector(domain_name="Fire", scores=scores, timestamp=t)
        trajectory.append(v)

        print(f"t={t:.0f}  {v}")
        print(f"     magnitude: {v.magnitude:.2f}  dominant: {Fire.axis_names()[v.dominant_axis]}")

        # Show velocity and acceleration for the dominant axis
        if trajectory.length >= 2:
            dom = v.dominant_axis
            vel = trajectory.velocity(dom)
            acc = trajectory.acceleration(dom)
            proj = trajectory.project(dom, steps_ahead=2.0)
            print(f"     {Fire.axis_names()[dom]}: vel={vel:+.2f}  acc={acc:+.2f}  projected(t+2)={proj:.2f}")

        # Check triggers
        events = trigger.evaluate(trajectory)
        for e in events:
            marker = "!!!" if e.response == Response.ACT else " ! "
            print(f"     {marker} {e.response.name}: {e.axis_name} "
                  f"current={e.current_score:.1f} projected={e.projected_score:.1f}")

        print()

    # Final state
    print("=" * 60)
    print("Final projection (all axes, 2 steps ahead):")
    projected = trajectory.project_all(steps_ahead=2.0)
    for name, val in zip(Fire.axis_names(), projected):
        bar = "#" * int(val)
        print(f"  {name:>18s}: {val:5.1f} {bar}")
    print()
    print("Zero is safety. Everything above zero is cost.")
    print("=" * 60)


if __name__ == "__main__":
    main()
