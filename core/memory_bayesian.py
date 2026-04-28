"""
Bayesian memory with posterior-aware eviction.

The naive approach (evict lowest consolidation) has sequential bias:
    - Early experiences get inflated surprise because the prior was flat
    - Late experiences get penalized because the prior is already sharp
    - The stored distribution drifts from the true distribution
    - Evicting linearly compounds these errors over time

The fix: every eviction decision must be statistically rigorous.

Three mechanisms replace linear eviction:

1. INFLUENCE FUNCTION
   For each stored experience, compute: "if I removed this from
   the posterior, how much would the posterior change?"
   This is the leave-one-out (LOO) influence. High influence = keep.
   Low influence = the posterior barely needs this experience.

2. COVERAGE MAINTENANCE
   The stored set must be representative, not biased toward extremes.
   A memory full of crises doesn't help recognize routine situations.
   We enforce coverage by partitioning feature space into regions
   and ensuring each region retains proportional representation.

3. POSTERIOR REPLAY
   Periodically re-evaluate ALL stored experiences against the
   CURRENT posterior. An experience that was surprising at storage
   time may be well-explained now (redundant with later experiences).
   An experience that seemed routine may have become load-bearing
   because similar experiences were evicted.

   This is the backpropagation: propagate the current posterior
   backward through all stored experiences and recompute their value.

The combination prevents:
    - Recency bias (recent ≠ important)
    - Primacy bias (first ≠ important)
    - Extremity bias (intense ≠ important)
    - Redundancy (10 copies of the same crisis ≠ 10x value)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Experience:
    """A stored experience with influence metadata."""
    id: int
    timestamp: float
    feature_vector: torch.Tensor      # (D,)
    domain: str
    valence_vector: torch.Tensor      # (N_axes,)
    action_taken: str
    outcome: float                    # -1 to +1

    # Bayesian metadata (recomputed on replay)
    initial_surprise: float           # KL at time of storage
    current_influence: float = 0.0    # LOO influence on current posterior
    n_obs_at_storage: int = 1         # Posterior size when this was stored
    coverage_region: int = -1         # Which region of feature space
    coverage_sparsity: float = 0.0    # Inverse density of this region
    redundancy: float = 0.0           # How many similar experiences exist
    posterior_surprise: float = 0.0   # KL against CURRENT posterior (not storage-time)

    access_count: int = 0
    last_accessed: float = 0.0

    @property
    def normalized_influence(self) -> float:
        """
        Influence normalized by posterior size at storage time.

        Raw LOO influence is degenerate when n_obs is small:
        removing 1 of 2 observations shifts the posterior massively,
        but that's trivially true, not deeply informative.

        Normalization: influence / sqrt(n_obs_at_storage)
        This discounts early experiences whose high influence
        is an artifact of a thin posterior, not genuine importance.
        """
        return self.current_influence / math.sqrt(max(1, self.n_obs_at_storage))

    def retention_score(self) -> float:
        """
        Statistically rigorous retention priority.

        Unlike consolidation_score, this is recomputed on every
        replay cycle using the CURRENT posterior state.

        Uses NORMALIZED influence to eliminate primacy bias.
        Includes coverage sparsity to favor experiences in
        underrepresented regions of feature space.
        """
        return (
            self.normalized_influence * 4.0    # Posterior need, primacy-corrected
            + self.posterior_surprise * 2.0     # How surprising it is NOW
            + abs(self.outcome) * 2.0           # Extreme outcomes
            + (1.0 - self.redundancy) * 3.0    # Uniqueness in memory
            + self.coverage_sparsity * 2.0     # Bonus for sparse regions
            # No recency bonus — recency is not importance
        )

    def similarity_to(self, other_features: torch.Tensor) -> float:
        return F.cosine_similarity(
            self.feature_vector.unsqueeze(0),
            other_features.unsqueeze(0),
        ).item()


class BayesianPosterior:
    """
    Gaussian posterior with full LOO influence computation.

    Unlike the naive Prior, this maintains sufficient statistics
    that allow efficient leave-one-out recomputation.
    """

    def __init__(self, n_axes: int, prior_precision: float = 0.01):
        self.n_axes = n_axes
        self.prior_precision = prior_precision

        # Prior parameters (fixed)
        self.prior_mean = torch.zeros(n_axes)
        self.prior_prec = torch.ones(n_axes) * prior_precision

        # Sufficient statistics for incremental updates
        self.sum_obs = torch.zeros(n_axes)         # Sum of observations
        self.sum_sq_obs = torch.zeros(n_axes)      # Sum of squared observations
        self.n_obs = 0

        # Cache for posterior
        self._update_posterior()

    def _update_posterior(self):
        """Recompute posterior from sufficient statistics."""
        obs_prec = torch.ones(self.n_axes) * self.n_obs  # Assuming unit observation precision
        self.posterior_prec = self.prior_prec + obs_prec

        if self.n_obs > 0:
            obs_mean = self.sum_obs / self.n_obs
            self.posterior_mean = (
                self.prior_prec * self.prior_mean + obs_prec * obs_mean
            ) / self.posterior_prec
        else:
            self.posterior_mean = self.prior_mean.clone()

    def observe(self, valence: torch.Tensor) -> float:
        """Add observation and return KL surprise."""
        old_mean = self.posterior_mean.clone()
        old_prec = self.posterior_prec.clone()

        self.sum_obs += valence
        self.sum_sq_obs += valence ** 2
        self.n_obs += 1
        self._update_posterior()

        return self._kl(self.posterior_mean, self.posterior_prec, old_mean, old_prec)

    def influence_of(self, valence: torch.Tensor) -> float:
        """
        Leave-one-out influence: how much would the posterior change
        if this observation were removed?

        This is the key computation. High influence = this experience
        is load-bearing for the current posterior. Low influence = the
        posterior would be nearly identical without it.

        Computed analytically for Gaussian posterior (no retraining needed).
        """
        if self.n_obs <= 1:
            return float('inf')  # Can't remove the only observation

        # Posterior WITHOUT this observation
        loo_sum = self.sum_obs - valence
        loo_n = self.n_obs - 1

        loo_obs_prec = torch.ones(self.n_axes) * loo_n
        loo_prec = self.prior_prec + loo_obs_prec

        if loo_n > 0:
            loo_mean_obs = loo_sum / loo_n
            loo_mean = (
                self.prior_prec * self.prior_mean + loo_obs_prec * loo_mean_obs
            ) / loo_prec
        else:
            loo_mean = self.prior_mean.clone()

        # KL between full posterior and LOO posterior
        return self._kl(self.posterior_mean, self.posterior_prec, loo_mean, loo_prec)

    def importance_weight(self, valence: torch.Tensor) -> float:
        """
        Importance weight for replay: how much should this experience
        contribute to the posterior relative to a uniform weighting?

        Experiences far from the posterior mean get higher weight —
        they are informative precisely because they are unusual.
        Experiences near the mean get lower weight — they are
        redundant with the bulk of the data.

        This is the importance sampling correction that prevents
        the posterior from being dominated by the most common
        observation type.
        """
        # Likelihood ratio: P(observation | current posterior) vs uniform
        var = 1.0 / (self.posterior_prec + 1e-8)
        diff = valence - self.posterior_mean
        log_likelihood = -0.5 * torch.sum(diff ** 2 / var).item()

        # Invert: low likelihood = high importance weight
        # (unusual observations contribute more to learning)
        # Clamp to prevent overflow
        exponent = min(20.0, max(-20.0, -log_likelihood / self.n_axes))
        return math.exp(exponent)

    def surprise_against_current(self, valence: torch.Tensor) -> float:
        """
        How surprising is this observation against the CURRENT posterior?

        This differs from initial_surprise, which was computed against
        the posterior at storage time. An experience that was surprising
        then may be routine now (many similar experiences followed).
        Or vice versa: an experience that seemed routine may now be
        an outlier because similar experiences were evicted.
        """
        var = 1.0 / (self.posterior_prec + 1e-8)
        # Mahalanobis-like distance
        diff = valence - self.posterior_mean
        return 0.5 * torch.sum(diff ** 2 / var).item()

    def predict(self) -> tuple[torch.Tensor, torch.Tensor]:
        uncertainty = 1.0 / (self.posterior_prec + 1e-8)
        return self.posterior_mean.clone(), uncertainty

    @property
    def confidence(self) -> float:
        return self.posterior_prec.log().mean().exp().item()

    def expected_surprise(self) -> float:
        """
        Baseline surprise expected from a typical observation, used by
        the storage gate. For the Gaussian/KL formulation, the surprise
        units are KL of mean-shift between successive posteriors — these
        approach 0 as the posterior stabilizes, so the natural baseline
        is 0. Override in subclasses with a different surprise definition.
        """
        return 0.0

    @staticmethod
    def _kl(mu1, prec1, mu2, prec2) -> float:
        var1 = 1.0 / (prec1 + 1e-8)
        var2 = 1.0 / (prec2 + 1e-8)
        kl = 0.5 * torch.sum(
            torch.log(var2 / var1) + var1 / var2 + (mu2 - mu1) ** 2 / var2 - 1.0
        )
        return max(0.0, kl.item())


class BayesianMemory:
    """
    Statistically rigorous experience memory.

    Three departures from naive eviction:

    1. INFLUENCE-BASED RETENTION
       Each experience is scored by its leave-one-out influence on
       the current posterior — not by how surprising it was at storage
       time. This eliminates primacy bias.

    2. COVERAGE REGIONS
       Feature space is partitioned into regions (via simple binning
       of the dominant feature dimensions). Eviction is constrained
       to never empty a region entirely — maintaining distributional
       coverage even under memory pressure.

    3. PERIODIC REPLAY
       All stored experiences are re-evaluated against the current
       posterior at configurable intervals. This is the backward pass:
       propagate current knowledge back through stored experiences
       and recompute their value. Without replay, influence scores
       become stale and eviction decisions degrade.
    """

    def __init__(
        self,
        capacity: int = 1000,
        feature_dim: int = 1024,
        n_coverage_regions: int = 16,
        replay_interval: int = 10,    # Replay every N insertions
        posterior_kind: str = "gaussian",   # "gaussian" or "normal_gamma"
    ):
        if posterior_kind not in ("gaussian", "normal_gamma"):
            raise ValueError(f"posterior_kind must be 'gaussian' or 'normal_gamma', got {posterior_kind!r}")
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.n_coverage_regions = n_coverage_regions
        self.replay_interval = replay_interval
        self.posterior_kind = posterior_kind

        self.experiences: list[Experience] = []
        self.posteriors: dict = {}        # values: BayesianPosterior or NormalGammaPosterior
        self._next_id = 0
        self._insertions_since_replay = 0
        self.last_gate_attribution: Optional[str] = None     # "influence" | "surprise" | "outcome" | "intensity" | "rejected"

    def get_or_create_posterior(self, domain: str, n_axes: int):
        if domain not in self.posteriors:
            if self.posterior_kind == "normal_gamma":
                # Local import keeps the torch.distributions cost out of the
                # default-Gaussian path.
                from .memory_bayesian_ng import NormalGammaPosterior
                self.posteriors[domain] = NormalGammaPosterior(n_axes)
            else:
                self.posteriors[domain] = BayesianPosterior(n_axes)
        return self.posteriors[domain]

    def _assign_region(self, feature_vector: torch.Tensor) -> int:
        """
        Assign a feature vector to a coverage region.

        Simple approach: hash the signs of the top-k feature dimensions
        into a region index. This partitions the feature space into
        hyperoctants that are stable across observations.
        """
        # Use top dimensions by variance (approximated by magnitude)
        n_bits = int(math.log2(self.n_coverage_regions))
        top_dims = feature_vector.abs().topk(n_bits).indices
        signs = (feature_vector[top_dims] > 0).int()
        region = 0
        for i, s in enumerate(signs):
            region += s.item() << i
        return region % self.n_coverage_regions

    def _region_counts(self) -> dict[int, int]:
        """How many experiences in each coverage region."""
        counts: dict[int, int] = {}
        for exp in self.experiences:
            counts[exp.coverage_region] = counts.get(exp.coverage_region, 0) + 1
        return counts

    def record(
        self,
        feature_vector: torch.Tensor,
        domain: str,
        valence_vector: torch.Tensor,
        action_taken: str,
        outcome: float,
    ) -> Optional[Experience]:
        """
        Record with Bayesian rigor.

        1. Update posterior → compute surprise
        2. Gate: should this be stored?
        3. If memory full → statistically rigorous eviction
        4. Periodically replay all experiences
        """
        posterior = self.get_or_create_posterior(domain, len(valence_vector))
        surprise = posterior.observe(valence_vector)
        influence = posterior.influence_of(valence_vector)

        intensity = valence_vector.norm().item()
        region = self._assign_region(feature_vector)

        # Entropy-relative gating: subtract the baseline (= predictive entropy
        # for NG, 0 for KL-Gaussian). Gate thresholds then represent
        # information beyond typical, with consistent semantics across kinds.
        baseline = posterior.expected_surprise()
        excess_surprise = max(0.0, surprise - baseline)
        excess_influence = max(0.0, influence - baseline) if influence != float('inf') else float('inf')

        # Gate: should we store?
        if not self._should_store(excess_surprise, intensity, outcome, excess_influence):
            return None

        exp = Experience(
            id=self._next_id,
            timestamp=time.time(),
            feature_vector=feature_vector.detach().clone(),
            domain=domain,
            valence_vector=valence_vector.detach().clone(),
            action_taken=action_taken,
            outcome=outcome,
            initial_surprise=surprise,
            current_influence=influence,
            n_obs_at_storage=posterior.n_obs,
            coverage_region=region,
            posterior_surprise=surprise,  # At storage time, these are equal
            last_accessed=time.time(),
        )
        self._next_id += 1

        # Evict if needed (BEFORE inserting)
        if len(self.experiences) >= self.capacity:
            self._evict_rigorous(exp.coverage_region)

        self.experiences.append(exp)
        self._insertions_since_replay += 1

        # Periodic replay: the backward pass
        if self._insertions_since_replay >= self.replay_interval:
            self._replay()
            self._insertions_since_replay = 0

        return exp

    def _should_store(
        self,
        surprise: float,
        intensity: float,
        outcome: float,
        influence: float,
    ) -> bool:
        """Store if statistically informative, not just intense.

        Records `last_gate_attribution` indicating which OR-condition
        (or "rejected") drove the decision, for diagnostics.
        """
        fill = len(self.experiences) / max(1, self.capacity)
        selectivity = 1.0 + fill * 3.0

        # Influence is the primary criterion — does the posterior need this?
        if influence > 0.1 / selectivity:
            self.last_gate_attribution = "influence"
            return True
        if surprise > 0.3 / selectivity:
            self.last_gate_attribution = "surprise"
            return True
        if abs(outcome) > 0.7 / selectivity:
            self.last_gate_attribution = "outcome"
            return True
        if intensity > 6.0 / selectivity:
            self.last_gate_attribution = "intensity"
            return True

        self.last_gate_attribution = "rejected"
        return False

    def _evict_rigorous(self, incoming_region: int):
        """
        Statistically rigorous eviction.

        Three-pass eviction that eliminates sequential bias:

        Pass 1: RECOMPUTE all metadata against current state
            - Redundancy (pairwise similarity)
            - Coverage sparsity (inverse region density)
            - Retention scores (using normalized influence)

        Pass 2: IDENTIFY candidates
            - Never empty a coverage region
            - Prefer evicting from dense regions (coverage-driven)
            - Among candidates in dense regions, evict lowest retention

        Pass 3: EVICT with coverage redistribution
            - If incoming experience fills a sparse region, allow
              eviction from a dense region even if retention is higher
            - This actively rebalances the stored distribution
        """
        n_exp = len(self.experiences)

        # Pass 1: Recompute redundancy and coverage sparsity
        for i, exp_i in enumerate(self.experiences):
            similar_count = 0
            for j, exp_j in enumerate(self.experiences):
                if i != j and exp_i.similarity_to(exp_j.feature_vector) > 0.7:
                    similar_count += 1
            exp_i.redundancy = similar_count / max(1, n_exp)

        region_counts = self._region_counts()
        max_region_count = max(region_counts.values()) if region_counts else 1

        for exp in self.experiences:
            rc = region_counts.get(exp.coverage_region, 1)
            # Sparsity: experiences in rare regions get bonus
            exp.coverage_sparsity = 1.0 - (rc / max(1, max_region_count))

        # Pass 2: Identify candidates
        # Strategy: evict from the DENSEST region to rebalance
        densest_region = max(region_counts, key=region_counts.get)

        # Prefer evicting from dense regions
        candidates_dense = [
            e for e in self.experiences
            if e.coverage_region == densest_region
        ]

        # But never empty a region
        candidates_safe = [
            e for e in self.experiences
            if region_counts.get(e.coverage_region, 0) > 1
        ]

        # Pass 3: Evict
        if candidates_dense and len(candidates_dense) > 1:
            # Evict the weakest from the densest region
            worst = min(candidates_dense, key=lambda e: e.retention_score())
        elif candidates_safe:
            # Fallback: evict weakest from any non-singleton region
            worst = min(candidates_safe, key=lambda e: e.retention_score())
        else:
            # All regions are singletons — evict globally weakest
            worst = min(self.experiences, key=lambda e: e.retention_score())

        self.experiences.remove(worst)

    def _replay(self):
        """
        The backward pass: re-evaluate all stored experiences
        against the current posterior.

        This is the Bayes-on-steroids mechanism:

        Step 1: RECOMPUTE INFLUENCE
            For each experience, compute LOO influence against the
            CURRENT posterior (not the posterior at storage time).
            Use normalized influence to correct for primacy bias.

        Step 2: IMPORTANCE-WEIGHTED SURPRISE
            Recompute surprise against the current posterior, weighted
            by importance: experiences far from the posterior mean
            are more informative and get higher retention scores.

        Step 3: RECOMPUTE REDUNDANCY
            Pairwise similarity within current memory. If two experiences
            are near-identical, one is redundant and evictable.

        Step 4: UPDATE COVERAGE SPARSITY
            Recompute region densities. Experiences in sparse regions
            get retention bonus — they represent undersampled parts
            of the feature space.

        Without replay, the system drifts:
        - An experience stored with high surprise may now be redundant
          (later experiences covered the same territory)
        - An experience stored with low surprise may now be unique
          (similar experiences were evicted)

        Replay catches both cases. This IS the backward pass.
        """
        # Step 1 + 2: Recompute influence and importance-weighted surprise
        for domain, posterior in self.posteriors.items():
            domain_exps = [e for e in self.experiences if e.domain == domain]
            for exp in domain_exps:
                exp.current_influence = posterior.influence_of(exp.valence_vector)
                exp.posterior_surprise = posterior.surprise_against_current(exp.valence_vector)

                # Importance weight modulates posterior_surprise:
                # unusual observations (far from mean) get amplified
                iw = posterior.importance_weight(exp.valence_vector)
                exp.posterior_surprise *= min(iw, 5.0)  # Cap to prevent explosion

        # Step 3: Recompute redundancy
        n_exp = len(self.experiences)
        for i, exp_i in enumerate(self.experiences):
            similar_count = 0
            for j, exp_j in enumerate(self.experiences):
                if i != j and exp_i.similarity_to(exp_j.feature_vector) > 0.7:
                    similar_count += 1
            exp_i.redundancy = similar_count / max(1, n_exp)

        # Step 4: Update coverage sparsity
        region_counts = self._region_counts()
        max_rc = max(region_counts.values()) if region_counts else 1
        for exp in self.experiences:
            rc = region_counts.get(exp.coverage_region, 1)
            exp.coverage_sparsity = 1.0 - (rc / max(1, max_rc))

    def recall(self, feature_vector: torch.Tensor, top_k: int = 5) -> list[Experience]:
        """Recall similar experiences. Updates access metadata."""
        if not self.experiences:
            return []

        scored = [
            (exp.similarity_to(feature_vector), exp)
            for exp in self.experiences
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, exp in scored[:top_k]:
            if sim < 0.1:
                break
            exp.access_count += 1
            exp.last_accessed = time.time()
            results.append(exp)

        return results

    def predict(self, domain: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        if domain not in self.posteriors:
            return None, None, 0.0
        posterior = self.posteriors[domain]
        mean, uncertainty = posterior.predict()
        return mean, uncertainty, posterior.confidence

    def stats(self) -> dict:
        if not self.experiences:
            return {"n_stored": 0, "capacity": self.capacity, "fill": 0.0}

        return {
            "n_stored": len(self.experiences),
            "capacity": self.capacity,
            "fill": len(self.experiences) / self.capacity,
            "n_domains": len(self.posteriors),
            "n_regions_occupied": len(set(e.coverage_region for e in self.experiences)),
            "n_regions_total": self.n_coverage_regions,
            "mean_raw_influence": sum(e.current_influence for e in self.experiences) / len(self.experiences),
            "mean_norm_influence": sum(e.normalized_influence for e in self.experiences) / len(self.experiences),
            "mean_redundancy": sum(e.redundancy for e in self.experiences) / len(self.experiences),
            "mean_retention": sum(e.retention_score() for e in self.experiences) / len(self.experiences),
            "min_retention": min(e.retention_score() for e in self.experiences),
            "max_retention": max(e.retention_score() for e in self.experiences),
            "mean_outcome": sum(e.outcome for e in self.experiences) / len(self.experiences),
        }

    def coverage_report(self) -> str:
        """Show how well memory covers the feature space."""
        counts = self._region_counts()
        total = len(self.experiences)
        lines = ["  Coverage regions:"]
        for r in range(self.n_coverage_regions):
            c = counts.get(r, 0)
            pct = c / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            lines.append(f"    region {r:2d}: {c:3d} ({pct:5.1f}%) {bar}")
        return "\n".join(lines)
