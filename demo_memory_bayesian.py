"""
Bayesian memory with posterior-aware eviction — rigorous version.

Demonstrates three mechanisms that eliminate sequential bias:
    1. Influence functions (LOO): keep what the posterior needs
    2. Coverage regions: maintain distributional representation
    3. Periodic replay: backward pass recomputes all retention scores

Run: python -m vScore.demo_memory_bayesian
"""

import random
import torch

from vScore.core.memory_bayesian import BayesianMemory


def generate_fire_observation(profile: str) -> dict:
    """Generate fire observations with controlled variation."""
    noise = lambda: random.gauss(0, 0.5)

    profiles = {
        "routine_small": {
            "scores": [2+noise(), 1.5+noise(), 2+noise(), 7.5+noise(),
                       1+noise(), 0.5+noise(), 1.5+noise()],
            "action": "MONITOR", "outcome": 0.4 + random.random() * 0.3,
        },
        "growing": {
            "scores": [4.5+noise(), 3.5+noise(), 4.5+noise(), 4.5+noise(),
                       2.5+noise(), 2.5+noise(), 3.5+noise()],
            "action": "DEFENSIVE", "outcome": random.gauss(0, 0.3),
        },
        "crisis": {
            "scores": [8+noise(), 7+noise(), 8.5+noise(), 1+noise(),
                       7+noise(), 7.5+noise(), 8+noise()],
            "action": "EVACUATE", "outcome": -0.5 + random.gauss(0, 0.2),
        },
        "backdraft": {
            "scores": [3+noise(), 8+noise(), 9+noise(), 2+noise(),
                       8+noise(), 9+noise(), 7+noise()],
            "action": "EVACUATE", "outcome": -0.8 + random.gauss(0, 0.1),
        },
        "electrical": {
            "scores": [1+noise(), 2+noise(), 6+noise(), 5+noise(),
                       1+noise(), 2+noise(), 8+noise()],
            "action": "VENTILATE", "outcome": -0.3 + random.gauss(0, 0.2),
        },
    }

    obs = profiles[profile].copy()
    obs["scores"] = [max(0, s) for s in obs["scores"]]
    obs["profile"] = profile
    return obs


def main():
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("Bayesian Memory — Posterior-Aware Eviction")
    print("=" * 70)

    memory = BayesianMemory(
        capacity=20,
        feature_dim=64,
        n_coverage_regions=8,
        replay_interval=10,
    )

    # ── Phase 1: Mixed experience stream ──────────────────────

    print("\n── Phase 1: 80 observations, mixed fire conditions ──")
    print("  Simulating a firefighter's career in fast-forward...\n")

    sequence = (
        # Early career: mostly routine, occasional excitement
        ["routine_small"] * 8
        + ["growing"] * 3
        + ["routine_small"] * 5
        + ["crisis"]          # First real crisis
        + ["routine_small"] * 10
        + ["growing"] * 4
        + ["routine_small"] * 5
        + ["backdraft"]       # Rare event
        + ["routine_small"] * 10
        + ["crisis"]          # Second crisis
        + ["routine_small"] * 8
        + ["electrical"]      # Novel fire type
        + ["routine_small"] * 5
        + ["growing"] * 3
        + ["routine_small"] * 5
        + ["crisis"]          # Third crisis
        + ["backdraft"]       # Second backdraft
        + ["routine_small"] * 5
    )

    profile_counts = {}
    stored_by_profile = {}
    discarded_by_profile = {}

    for i, profile in enumerate(sequence):
        profile_counts[profile] = profile_counts.get(profile, 0) + 1

        obs = generate_fire_observation(profile)

        feature = torch.randn(64) * 0.3
        feature[:7] += torch.tensor(obs["scores"], dtype=torch.float32) * 0.2

        valence = torch.tensor(obs["scores"], dtype=torch.float32)

        exp = memory.record(
            feature_vector=feature,
            domain="Fire",
            valence_vector=valence,
            action_taken=obs["action"],
            outcome=obs["outcome"],
        )

        if exp:
            stored_by_profile[profile] = stored_by_profile.get(profile, 0) + 1
        else:
            discarded_by_profile[profile] = discarded_by_profile.get(profile, 0) + 1

    # ── Results ───────────────────────────────────────────────

    print(f"  {'profile':>15s}  {'seen':>5s}  {'stored':>6s}  {'discard':>7s}  {'retain%':>7s}")
    print(f"  {'─'*15}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")
    for profile in ["routine_small", "growing", "crisis", "backdraft", "electrical"]:
        seen = profile_counts.get(profile, 0)
        stored = stored_by_profile.get(profile, 0)
        discarded = discarded_by_profile.get(profile, 0)
        retain = stored / seen * 100 if seen > 0 else 0
        print(f"  {profile:>15s}  {seen:5d}  {stored:6d}  {discarded:7d}  {retain:6.1f}%")

    # ── What's in memory now ──────────────────────────────────

    print(f"\n── Memory contents ({len(memory.experiences)}/{memory.capacity}) ──")
    print(f"  {'id':>4s}  {'action':>12s}  {'out':>5s}  {'raw_infl':>8s}  "
          f"{'norm_infl':>9s}  {'n_obs':>5s}  {'sparse':>6s}  "
          f"{'retain':>7s}  {'rg':>2s}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*2}")

    for exp in sorted(memory.experiences, key=lambda e: e.retention_score(), reverse=True):
        print(f"  {exp.id:4d}  {exp.action_taken:>12s}  {exp.outcome:+4.1f}  "
              f"  {exp.current_influence:6.4f}  "
              f"  {exp.normalized_influence:7.4f}  "
              f"  {exp.n_obs_at_storage:3d}  "
              f"  {exp.coverage_sparsity:4.2f}  "
              f" {exp.retention_score():6.1f}  "
              f"  {exp.coverage_region:2d}")

    # ── Coverage report ───────────────────────────────────────

    print(f"\n── Coverage ──")
    print(memory.coverage_report())

    # ── Posterior state ───────────────────────────────────────

    print(f"\n── Learned posterior for Fire ──")
    posterior = memory.posteriors["Fire"]
    mean, uncertainty = posterior.predict()
    axis_names = ["spread", "prox", "intens", "contain", "escape", "struct", "smoke"]
    print(f"  Observations: {posterior.n_obs}")
    print(f"  Confidence: {posterior.confidence:.3f}")
    print(f"  {'axis':>10s}  {'mean':>6s}  {'uncert':>6s}")
    for name, m, u in zip(axis_names, mean, uncertainty):
        print(f"  {name:>10s}  {m.item():6.2f}  {u.item():6.4f}")

    # ── Compare: what would naive eviction have kept? ─────────

    print(f"\n── Statistical rigor check ──")

    stats = memory.stats()
    print(f"  Regions occupied: {stats['n_regions_occupied']}/{stats['n_regions_total']}")
    print(f"  Mean raw influence: {stats['mean_raw_influence']:.4f}")
    print(f"  Mean norm influence: {stats['mean_norm_influence']:.4f}")
    print(f"  Mean redundancy: {stats['mean_redundancy']:.3f}")
    print(f"  Retention range: [{stats['min_retention']:.2f}, {stats['max_retention']:.2f}]")

    # Check for sequential bias
    ids = [e.id for e in memory.experiences]
    early = sum(1 for i in ids if i < 20)
    middle = sum(1 for i in ids if 20 <= i < 50)
    late = sum(1 for i in ids if i >= 50)
    print(f"\n  Temporal distribution of retained experiences:")
    print(f"    Early (id < 20):  {early:2d}  {'#' * early}")
    print(f"    Middle (20-49):   {middle:2d}  {'#' * middle}")
    print(f"    Late (≥ 50):      {late:2d}  {'#' * late}")

    total = len(memory.experiences)
    # Expected if unbiased: proportional to # of observations in each period
    early_expected = 20 / len(sequence) * total
    middle_expected = 30 / len(sequence) * total
    late_expected = (len(sequence) - 50) / len(sequence) * total
    print(f"    Expected (unbiased): {early_expected:.1f} / {middle_expected:.1f} / {late_expected:.1f}")

    # Action distribution
    action_counts = {}
    for exp in memory.experiences:
        action_counts[exp.action_taken] = action_counts.get(exp.action_taken, 0) + 1
    print(f"\n  Action distribution in memory:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        print(f"    {action:>12s}: {count:2d} ({pct:5.1f}%) {'#' * count}")

    print(f"""
{'=' * 70}
WHAT THE RIGOROUS VERSION FIXES
{'=' * 70}

  1. NO PRIMACY BIAS
     Experience #0 is not privileged just because the prior was
     flat when it arrived. Influence is recomputed on replay —
     if #0 is now redundant with later experiences, its influence
     drops and it becomes evictable.

  2. NO SEQUENTIAL BIAS
     The temporal distribution of retained experiences should
     roughly match where informative events occurred in the
     sequence, not when they were first seen.

  3. COVERAGE MAINTENANCE
     Eviction never empties a coverage region. Even under
     memory pressure, the stored set represents the full
     feature space — not just the extremes.

  4. REDUNDANCY PENALTY
     Having 5 copies of "routine small fire" is not 5x valuable.
     Redundancy is computed pairwise and penalizes duplicates.
     One representative of each pattern is worth more than
     five copies of one pattern.

  5. REPLAY = BACKPROPAGATION
     Every {memory.replay_interval} insertions, all experiences are re-evaluated
     against the current posterior. Retention scores are
     recomputed. Stale assessments are corrected.
     This IS the backward pass.

  Bayes is the teacher.
  Influence is the grade.
  Coverage is the curriculum.
  Replay is the exam.
""")


if __name__ == "__main__":
    main()
