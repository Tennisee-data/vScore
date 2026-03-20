"""
Bayesian memory demo.

Shows the full learning loop:
    1. Encounter visual pattern → score it
    2. Prior says "I expect X" → observe Y → surprise = KL(X, Y)
    3. High surprise → store. Low surprise → discard.
    4. Memory full → evict the boring, keep the surprising.
    5. Prior sharpens over time → faster, more confident scoring.

Run: python -m vScore.demo_memory
"""

import torch
import random
import math

from vScore.core.memory import ExperienceMemory, Prior


def main():
    print("=" * 70)
    print("vScore Bayesian Memory — Learning Loop")
    print("=" * 70)

    # Small memory to demonstrate eviction
    memory = ExperienceMemory(capacity=15, feature_dim=64)

    # ── Phase 1: Learn fire patterns ──────────────────────────

    print("\n── Phase 1: Learning fire patterns ──")
    print("  The system has never seen fire. Prior is flat.\n")

    n_axes = 7  # spread, proximity, intensity, containment, escape, structural, smoke
    axis_names = ["spread", "prox", "intens", "contain", "escape", "struct", "smoke"]

    # Simulate 20 fire observations with varying conditions
    fire_observations = [
        # Routine small fires (low intensity, contained)
        {"scores": [2, 2, 2, 8, 1, 1, 1], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [2, 1, 3, 7, 1, 1, 2], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [3, 2, 2, 7, 1, 1, 1], "action": "MONITOR",   "outcome":  0.3, "label": "small contained"},
        {"scores": [2, 2, 3, 8, 1, 0, 2], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [1, 1, 2, 9, 1, 0, 1], "action": "MONITOR",   "outcome":  0.8, "label": "small contained"},
        # Growing fires (medium intensity)
        {"scores": [5, 4, 5, 4, 3, 3, 4], "action": "DEFENSIVE", "outcome":  0.0, "label": "growing fire"},
        {"scores": [4, 3, 4, 5, 2, 2, 3], "action": "ADVANCE",   "outcome":  0.3, "label": "growing fire"},
        {"scores": [5, 5, 5, 3, 3, 3, 5], "action": "DEFENSIVE", "outcome": -0.2, "label": "growing fire"},
        {"scores": [6, 4, 6, 3, 4, 3, 5], "action": "DEFENSIVE", "outcome": -0.3, "label": "growing fire"},
        # Crisis fires (high intensity — THESE SHOULD BE REMEMBERED)
        {"scores": [8, 7, 8, 1, 7, 7, 8], "action": "EVACUATE",  "outcome": -0.5, "label": "crisis"},
        {"scores": [9, 8, 9, 1, 8, 8, 9], "action": "MAYDAY",    "outcome": -0.9, "label": "near-miss"},
        # Then more routine fires
        {"scores": [2, 2, 2, 7, 1, 1, 1], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [3, 2, 3, 7, 1, 1, 2], "action": "MONITOR",   "outcome":  0.4, "label": "small contained"},
        {"scores": [2, 1, 2, 8, 1, 1, 1], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [2, 2, 3, 8, 1, 0, 1], "action": "MONITOR",   "outcome":  0.6, "label": "small contained"},
        # ANOMALY: unusual fire — backdraft (sudden high intensity from low)
        {"scores": [3, 8, 9, 2, 8, 9, 7], "action": "EVACUATE",  "outcome": -0.8, "label": "BACKDRAFT"},
        # More routine
        {"scores": [2, 2, 2, 8, 1, 1, 1], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [3, 3, 3, 6, 2, 1, 2], "action": "MONITOR",   "outcome":  0.3, "label": "small contained"},
        {"scores": [2, 1, 2, 8, 1, 0, 1], "action": "MONITOR",   "outcome":  0.5, "label": "small contained"},
        {"scores": [4, 3, 4, 5, 2, 2, 3], "action": "ADVANCE",   "outcome":  0.2, "label": "growing fire"},
    ]

    for i, obs in enumerate(fire_observations):
        # Simulate feature vector (in real system, this comes from V-JEPA)
        feature = torch.randn(64) * 0.5
        # Add signal correlated with intensity
        feature[:7] += torch.tensor(obs["scores"], dtype=torch.float32) * 0.3

        valence = torch.tensor(obs["scores"], dtype=torch.float32)

        exp = memory.record(
            feature_vector=feature,
            domain="Fire",
            valence_vector=valence,
            action_taken=obs["action"],
            outcome=obs["outcome"],
        )

        stored = "STORED" if exp else "discarded"
        surprise = memory.priors["Fire"]._kl_divergence(
            memory.priors["Fire"].mean,
            memory.priors["Fire"].precision,
            valence,
            torch.ones(n_axes),
        ) if exp is None else exp.surprise

        label = obs["label"]
        print(f"  [{i+1:2d}] {label:>18s}  {obs['action']:>12s}  "
              f"outcome={obs['outcome']:+.1f}  "
              f"surprise={surprise:.3f}  → {stored}")

    # ── Show what's in memory ─────────────────────────────────

    print(f"\n── Memory contents ({len(memory.experiences)}/{memory.capacity}) ──")
    print(f"  {'id':>3s}  {'action':>12s}  {'outcome':>7s}  {'surprise':>8s}  "
          f"{'intensity':>9s}  {'complex':>7s}  {'consol':>6s}  label")
    print(f"  {'─'*3}  {'─'*12}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*15}")

    # Reconstruct labels for display
    for exp in sorted(memory.experiences, key=lambda e: e.consolidation_score, reverse=True):
        intensity_bar = "#" * min(10, int(exp.intensity / 2))
        print(f"  {exp.id:3d}  {exp.action_taken:>12s}  {exp.outcome:+5.1f}  "
              f"  {exp.surprise:6.3f}  {exp.intensity:7.1f}  "
              f"  {exp.complexity:.3f}   {exp.consolidation_score:5.1f}  {intensity_bar}")

    # ── Show the prior ────────────────────────────────────────

    print(f"\n── Learned prior for Fire domain ──")
    prior = memory.priors["Fire"]
    mean, uncertainty = prior.predict()
    print(f"  Observations: {prior.n_observations}")
    print(f"  Confidence: {prior.confidence:.3f}")
    print(f"  {'axis':>10s}  {'mean':>6s}  {'uncert':>6s}  {'precision':>9s}")
    for name, m, u, p in zip(axis_names, mean, uncertainty, prior.precision):
        bar = "#" * int(p.item())
        print(f"  {name:>10s}  {m.item():6.2f}  {u.item():6.3f}  {p.item():9.2f}  {bar}")

    # ── Phase 2: Recall and prediction ────────────────────────

    print(f"\n── Phase 2: New fire — what does memory say? ──")

    # New observation: looks like a growing fire
    new_feature = torch.randn(64) * 0.5
    new_feature[:7] += torch.tensor([5, 4, 5, 3, 3, 3, 4], dtype=torch.float32) * 0.3

    recalled = memory.recall(new_feature, top_k=3)
    print(f"  New observation resembles a growing fire")
    print(f"  Recalled {len(recalled)} similar experiences:")
    for exp in recalled:
        sim = exp.similarity_to(new_feature)
        print(f"    #{exp.id}  action={exp.action_taken:>12s}  "
              f"outcome={exp.outcome:+.1f}  sim={sim:.3f}")

    # Prior prediction
    pred_mean, pred_uncert, confidence = memory.predict("Fire")
    print(f"\n  Prior prediction: confidence={confidence:.3f}")
    print(f"  Expected valence: [{', '.join(f'{v:.1f}' for v in pred_mean.tolist())}]")
    print(f"  Uncertainty:      [{', '.join(f'{v:.3f}' for v in pred_uncert.tolist())}]")

    # ── Phase 3: Demonstrate selective forgetting ─────────────

    print(f"\n── Phase 3: Memory under pressure ──")
    print(f"  Flooding memory with 50 routine observations...")

    stored_count = 0
    discarded_count = 0
    for i in range(50):
        feature = torch.randn(64) * 0.5
        # Routine fire
        scores = [2 + random.random(), 1 + random.random(),
                  2 + random.random(), 7 + random.random(),
                  1 + random.random(), 0.5 + random.random(),
                  1 + random.random()]
        feature[:7] += torch.tensor(scores, dtype=torch.float32) * 0.3
        valence = torch.tensor(scores, dtype=torch.float32)

        exp = memory.record(
            feature_vector=feature,
            domain="Fire",
            valence_vector=valence,
            action_taken="MONITOR",
            outcome=0.3 + random.random() * 0.4,
        )
        if exp:
            stored_count += 1
        else:
            discarded_count += 1

    print(f"  Stored: {stored_count}  Discarded: {discarded_count}")
    print(f"  Memory: {len(memory.experiences)}/{memory.capacity}")

    # Show what survived
    print(f"\n── Survivors: what memory kept ──")
    for exp in sorted(memory.experiences, key=lambda e: e.consolidation_score, reverse=True):
        print(f"  #{exp.id:3d}  {exp.action_taken:>12s}  outcome={exp.outcome:+.1f}  "
              f"surprise={exp.surprise:.3f}  intensity={exp.intensity:.1f}  "
              f"consol={exp.consolidation_score:.1f}")

    # ── Summary ───────────────────────────────────────────────

    stats = memory.stats()
    print(f"\n── Memory statistics ──")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:>25s}: {v:.3f}")
        else:
            print(f"  {k:>25s}: {v}")

    print(f"\n  Compression ratio: {memory.compression_ratio():.2f}")
    print(f"  (fraction of stored experiences that were highly surprising)")

    print(f"""
{'=' * 70}
KEY OBSERVATIONS
{'=' * 70}

  1. SELECTIVE STORAGE: Routine small fires were mostly discarded.
     Crisis events and the backdraft anomaly were kept.

  2. BAYESIAN PRIOR: After 20+ observations, the prior has
     sharpened. New observations that match the prior cause
     LOW surprise → discarded. Anomalies cause HIGH surprise → kept.

  3. EVICTION POLICY: When memory filled, low-consolidation
     experiences (routine, low surprise, neutral outcome) were
     evicted first. High-consolidation ones (crisis, backdraft)
     survived the pressure.

  4. COMPRESSED STORAGE: Each experience is a feature vector +
     valence vector + metadata. Not a video. The experience
     IS the compressed vector. Reconstruction is the encoder's
     job, not memory's.

  The prior IS the learning.
  Surprise IS the curriculum.
  Consolidation IS the forgetting policy.
  Bayes IS the teacher.
""")


if __name__ == "__main__":
    main()
