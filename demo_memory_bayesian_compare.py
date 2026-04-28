"""
End-to-end comparison: BayesianMemory with the original Gaussian
posterior vs the new Normal-Gamma posterior.

Same observation stream (same seed), same capacity, same thresholds.
We let the gate (`_should_store`), eviction, and replay all run as
production-configured. The point is to see how the change of
posterior affects:

    - per-profile retention rate (gate selectivity)
    - final memory composition (action and profile mix)
    - temporal distribution of retained experiences
    - eviction count and region coverage

The thresholds in `_should_store` (0.1, 0.3, 0.7, 6.0) were calibrated
against the Gaussian posterior's KL-units surprise. The Normal-Gamma
posterior's surprise is in log-density nats, so the same thresholds
correspond to a different operating point. This script exposes that.
"""

from __future__ import annotations

import random
import torch

from vScore.core.memory_bayesian import BayesianMemory
from vScore.demo_memory_bayesian import generate_fire_observation


# ---------------------------------------------------------------------------
# Same fire-career sequence as demo_memory_bayesian.py
# ---------------------------------------------------------------------------

SEQUENCE = (
    ["routine_small"] * 8
    + ["growing"] * 3
    + ["routine_small"] * 5
    + ["crisis"]
    + ["routine_small"] * 10
    + ["growing"] * 4
    + ["routine_small"] * 5
    + ["backdraft"]
    + ["routine_small"] * 10
    + ["crisis"]
    + ["routine_small"] * 8
    + ["electrical"]
    + ["routine_small"] * 5
    + ["growing"] * 3
    + ["routine_small"] * 5
    + ["crisis"]
    + ["backdraft"]
    + ["routine_small"] * 5
)

PROFILES = ["routine_small", "growing", "crisis", "backdraft", "electrical"]


# ---------------------------------------------------------------------------
# Run one configuration with reproducible RNG and collect everything
# ---------------------------------------------------------------------------


def run_one(posterior_kind: str, seed: int = 42) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    memory = BayesianMemory(
        capacity=20,
        feature_dim=64,
        n_coverage_regions=8,
        replay_interval=10,
        posterior_kind=posterior_kind,
    )

    profile_seen = {p: 0 for p in PROFILES}
    profile_stored = {p: 0 for p in PROFILES}
    profile_discarded = {p: 0 for p in PROFILES}
    surprise_trace: list[float] = []
    influence_trace: list[float] = []
    gate_attr_counts: dict[str, int] = {
        "influence": 0, "surprise": 0, "outcome": 0, "intensity": 0, "rejected": 0,
    }

    for i, profile in enumerate(SEQUENCE):
        profile_seen[profile] += 1
        obs = generate_fire_observation(profile)

        feature = torch.randn(64) * 0.3
        feature[:7] += torch.tensor(obs["scores"], dtype=torch.float32) * 0.2
        valence = torch.tensor(obs["scores"], dtype=torch.float32)

        # Capture the raw posterior signals before record() makes its
        # gate decision, so we can characterize what the model sees.
        post = memory.get_or_create_posterior("Fire", len(valence))
        # observe() mutates state, so we need to compute surprise/influence
        # against the pre-update state, then let record() do its own update.
        # record() will call observe() again — but observe is the natural
        # entry point. To avoid double-update, we read from a snapshot.
        if posterior_kind == "gaussian":
            old_mean = post.posterior_mean.clone()
            old_prec = post.posterior_prec.clone()
            old_n = post.n_obs
            old_sum = post.sum_obs.clone()
            old_sumsq = post.sum_sq_obs.clone()

            exp = memory.record(
                feature_vector=feature,
                domain="Fire",
                valence_vector=valence,
                action_taken=obs["action"],
                outcome=obs["outcome"],
            )
            # The post-update LOO influence reflects what the gate saw.
            influence_trace.append(post.influence_of(valence))
            # Surprise: the KL between posteriors before/after — recover from delta.
            surprise_trace.append(
                post._kl(post.posterior_mean, post.posterior_prec, old_mean, old_prec)
            )
        else:
            # Normal-Gamma — snapshot sufficient stats
            old_n = post.n
            old_sum = post.sum_x.clone()
            old_sumsq = post.sum_x2.clone()

            exp = memory.record(
                feature_vector=feature,
                domain="Fire",
                valence_vector=valence,
                action_taken=obs["action"],
                outcome=obs["outcome"],
            )
            # Surprise: -log p(valence | data-before-this-obs)
            pred_before = post._predictive(old_n, old_sum, old_sumsq)
            surprise_trace.append(-pred_before.log_prob(valence).sum().item())
            influence_trace.append(post.influence_of(valence))

        if exp is not None:
            profile_stored[profile] += 1
        else:
            profile_discarded[profile] += 1
        if memory.last_gate_attribution is not None:
            gate_attr_counts[memory.last_gate_attribution] += 1

    # Inventory of stored experiences
    profile_in_memory = {p: 0 for p in PROFILES}
    action_in_memory: dict[str, int] = {}
    for exp in memory.experiences:
        # Profile isn't stored on Experience, recover from action_taken
        # using the unique action-per-profile mapping in the generator.
        action_in_memory[exp.action_taken] = action_in_memory.get(exp.action_taken, 0) + 1

    ids = [e.id for e in memory.experiences]
    early = sum(1 for i in ids if i < 20)
    middle = sum(1 for i in ids if 20 <= i < 50)
    late = sum(1 for i in ids if i >= 50)

    return {
        "kind": posterior_kind,
        "memory": memory,
        "profile_seen": profile_seen,
        "profile_stored": profile_stored,
        "profile_discarded": profile_discarded,
        "action_in_memory": action_in_memory,
        "temporal": (early, middle, late),
        "surprise_trace": surprise_trace,
        "influence_trace": influence_trace,
        "gate_attribution": gate_attr_counts,
        "n_observed": len(SEQUENCE),
        "n_stored_total": sum(profile_stored.values()),
        "n_in_memory": len(memory.experiences),
        "n_evicted": sum(profile_stored.values()) - len(memory.experiences),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(num: int, denom: int) -> str:
    return f"{(num/denom*100 if denom else 0):5.1f}%"


def report(g: dict, ng: dict) -> None:
    n_obs = g["n_observed"]
    print("=" * 76)
    print("BayesianMemory: Gaussian vs Normal-Gamma posterior")
    print("=" * 76)
    print(f"Identical RNG, identical thresholds, {n_obs} observations, capacity 20.")

    # ---- Gate behavior ----
    print("\n── Gate behavior (per-profile retention) ──")
    print(f"  {'profile':>14}  {'seen':>4}  "
          f"{'old stored':>10} {'old retain':>10}    "
          f"{'new stored':>10} {'new retain':>10}")
    for p in PROFILES:
        seen = g["profile_seen"][p]
        gs = g["profile_stored"][p]
        ns = ng["profile_stored"][p]
        print(f"  {p:>14}  {seen:>4d}  "
              f"{gs:>10d} {_fmt_pct(gs, seen):>10}    "
              f"{ns:>10d} {_fmt_pct(ns, seen):>10}")

    print(f"\n  total stored:   old = {g['n_stored_total']:3d}    "
          f"new = {ng['n_stored_total']:3d}")
    print(f"  evictions:      old = {g['n_evicted']:3d}    "
          f"new = {ng['n_evicted']:3d}")
    print(f"  in memory:      old = {g['n_in_memory']:3d}    "
          f"new = {ng['n_in_memory']:3d}")

    # ---- Gate attribution: which condition fired? ----
    print("\n── Gate attribution (which OR-condition triggered first) ──")
    print(f"  {'condition':>12}  {'old':>5}  {'old%':>6}  {'new':>5}  {'new%':>6}")
    for cond in ["influence", "surprise", "outcome", "intensity", "rejected"]:
        og = g["gate_attribution"][cond]
        on = ng["gate_attribution"][cond]
        print(f"  {cond:>12}  {og:>5d}  {_fmt_pct(og, n_obs):>6}  "
              f"{on:>5d}  {_fmt_pct(on, n_obs):>6}")

    # ---- Posterior signals ----
    import statistics as st
    print("\n── Posterior signals (per-observation, mean across the run) ──")
    print(f"  {'metric':>22}  {'old':>14}  {'new':>14}")
    print(f"  {'mean surprise':>22}  "
          f"{st.mean(g['surprise_trace']):>14.4f}  "
          f"{st.mean(ng['surprise_trace']):>14.4f}")
    print(f"  {'median surprise':>22}  "
          f"{st.median(g['surprise_trace']):>14.4f}  "
          f"{st.median(ng['surprise_trace']):>14.4f}")
    print(f"  {'max surprise':>22}  "
          f"{max(g['surprise_trace']):>14.4f}  "
          f"{max(ng['surprise_trace']):>14.4f}")
    g_infl_finite = [x for x in g['influence_trace'] if x != float('inf')]
    ng_infl_finite = [x for x in ng['influence_trace'] if x != float('inf')]
    print(f"  {'mean LOO influence':>22}  "
          f"{st.mean(g_infl_finite):>14.4f}  "
          f"{st.mean(ng_infl_finite):>14.4f}")
    print(f"  {'max LOO influence':>22}  "
          f"{max(g_infl_finite):>14.4f}  "
          f"{max(ng_infl_finite):>14.4f}")
    print("  (LOO is +inf at the very first observation — n=1 has nothing to leave out.")
    print("   Means above are over the n>=2 portion of the trace.)")
    print("\n  Note: old surprise = KL between Gaussian posteriors (mean shift");
    print("        only). New surprise = -log p(x | data) in nats.")

    # ---- Final memory composition ----
    print("\n── Memory composition (action distribution) ──")
    actions = sorted(set(list(g['action_in_memory']) + list(ng['action_in_memory'])))
    print(f"  {'action':>14}  {'old':>4} {'old%':>6}  {'new':>4} {'new%':>6}")
    for a in actions:
        og = g["action_in_memory"].get(a, 0)
        on = ng["action_in_memory"].get(a, 0)
        print(f"  {a:>14}  {og:>4d} {_fmt_pct(og, g['n_in_memory']):>6}  "
              f"{on:>4d} {_fmt_pct(on, ng['n_in_memory']):>6}")

    # ---- Temporal distribution ----
    print("\n── Temporal distribution (sequential bias check) ──")
    print("  expected if unbiased ≈ proportional to observations in each slice")
    e_g, m_g, l_g = g["temporal"]
    e_n, m_n, l_n = ng["temporal"]
    print(f"  {'slice':>22}  {'old':>5}  {'new':>5}")
    print(f"  {'early (id < 20)':>22}  {e_g:>5d}  {e_n:>5d}")
    print(f"  {'middle (20 <= id < 50)':>22}  {m_g:>5d}  {m_n:>5d}")
    print(f"  {'late (id >= 50)':>22}  {l_g:>5d}  {l_n:>5d}")

    # ---- Coverage ----
    print("\n── Coverage region occupancy ──")
    g_cov = g["memory"]._region_counts()
    ng_cov = ng["memory"]._region_counts()
    n_regions = g["memory"].n_coverage_regions
    print(f"  {'region':>6}  {'old':>5}  {'new':>5}")
    for r in range(n_regions):
        print(f"  {r:>6d}  {g_cov.get(r, 0):>5d}  {ng_cov.get(r, 0):>5d}")
    print(f"  occupied: old = {len([r for r, c in g_cov.items() if c > 0])}/{n_regions}    "
          f"new = {len([r for r, c in ng_cov.items() if c > 0])}/{n_regions}")

    # ---- Posterior state ----
    print("\n── Final posterior, axis 'intens' (axis index 2) ──")
    g_post = g["memory"].posteriors["Fire"]
    ng_post = ng["memory"].posteriors["Fire"]
    g_mean, g_unc = g_post.predict()
    ng_mean, ng_unc = ng_post.predict()
    print(f"  {'metric':>22}  {'old':>10}  {'new':>10}")
    print(f"  {'predicted mean':>22}  {g_mean[2].item():>10.4f}  {ng_mean[2].item():>10.4f}")
    print(f"  {'predicted variance':>22}  {g_unc[2].item():>10.4f}  {ng_unc[2].item():>10.4f}")
    print(f"  {'confidence':>22}  {g_post.confidence:>10.4f}  {ng_post.confidence:>10.4f}")
    print(f"  {'n_obs':>22}  {g_post.n_obs:>10d}  {ng_post.n_obs:>10d}")
    print("\n  (old's variance is 1/posterior_prec — ignores data scatter;")
    print("   new's variance is the predictive Student-t variance.)")


def main():
    g = run_one("gaussian")
    ng = run_one("normal_gamma")
    report(g, ng)


if __name__ == "__main__":
    main()
