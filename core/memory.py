"""
Bayesian experience memory.

The missing piece: a system that scores but does not learn is a
thermometer. A system that updates its scoring based on accumulated
experience is an organism.

The core Bayesian loop:
    1. Observe visual pattern → extract features
    2. Recall prior: "patterns like this previously led to valence V"
    3. Score current observation → likelihood
    4. Update posterior: "now I believe patterns like this lead to V'"
    5. If posterior changed significantly → store this experience
    6. If memory is full → evict the least informative experience

What gets stored is NOT the raw video. It is:
    - A compressed feature vector (the visual signature)
    - The valence vector at the moment of action
    - The action taken
    - The outcome (did the action succeed?)
    - The surprise: how much this experience changed the prior

What gets evicted:
    - Low surprise (confirmed what we already knew)
    - Low intensity (near-zero valence — nothing happened)
    - Low complexity (simple pattern, easy to relearn)
    - Old age with low access count (hasn't been recalled)

What NEVER gets evicted:
    - High surprise (changed our beliefs significantly)
    - Survival-critical outcomes (pain, near-miss, loss)
    - Unique patterns (nothing else in memory resembles it)

This mirrors biological memory consolidation:
    - Emotional intensity enhances encoding (amygdala → hippocampus)
    - Surprise enhances encoding (prediction error → dopamine)
    - Sleep consolidation preferentially retains high-value memories
    - Routine experiences fade; exceptional ones persist
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Experience:
    """
    A single stored experience. Not a video — a compressed
    record of what happened, what it meant, and how surprising
    it was.
    """
    # Identity
    id: int
    timestamp: float

    # What was seen (compressed)
    feature_vector: torch.Tensor       # Encoder output, (D,)
    domain: str

    # What it meant
    valence_vector: torch.Tensor       # Valence scores at moment of action, (N_axes,)
    action_taken: str                  # Which action the system took
    outcome: float                     # -1 (bad) to +1 (good), 0 = neutral

    # How informative it was
    surprise: float                    # KL divergence: how much this changed the prior
    intensity: float                   # L2 norm of valence vector
    complexity: float                  # How hard this pattern is to reconstruct

    # Memory metadata
    access_count: int = 0              # How often this memory has been recalled
    last_accessed: float = 0.0         # When it was last recalled
    consolidation_score: float = 0.0   # Computed priority for retention

    def update_consolidation(self):
        """
        Compute retention priority.

        High consolidation = keep. Low consolidation = evictable.

        Consolidation is driven by:
            - Surprise (how much it changed beliefs)
            - Intensity (how activated the valence state was)
            - Complexity (how hard to relearn if forgotten)
            - Recency × access (recently recalled = still relevant)
            - Outcome extremity (very good or very bad outcomes stick)
        """
        # Time decay: memories lose consolidation over time
        # unless they are accessed (which refreshes them)
        age = time.time() - self.last_accessed if self.last_accessed > 0 else 1.0
        recency = 1.0 / (1.0 + math.log1p(age / 3600))  # Log decay over hours

        # Access frequency boost
        access_boost = math.log1p(self.access_count) / 5.0

        # Outcome extremity: both very good and very bad outcomes
        # are worth remembering. Neutral outcomes are forgettable.
        outcome_weight = abs(self.outcome)

        self.consolidation_score = (
            self.surprise * 3.0           # Surprise dominates
            + self.intensity * 1.0        # High activation matters
            + self.complexity * 2.0       # Hard-to-relearn matters
            + outcome_weight * 2.0        # Extreme outcomes matter
            + recency * 1.0              # Recent access matters
            + access_boost               # Frequently recalled matters
        )

    def similarity_to(self, feature_vector: torch.Tensor) -> float:
        """Cosine similarity to another feature vector."""
        return torch.nn.functional.cosine_similarity(
            self.feature_vector.unsqueeze(0),
            feature_vector.unsqueeze(0)
        ).item()


@dataclass
class Prior:
    """
    Bayesian prior over valence outcomes for a region of feature space.

    Parameterized as a Gaussian: mean and precision (inverse variance)
    per axis. Updated via conjugate Bayesian update.

    The prior represents: "for visual patterns in this region,
    I expect these valence scores with this confidence."
    """
    mean: torch.Tensor           # Expected valence, (N_axes,)
    precision: torch.Tensor      # Confidence per axis, (N_axes,). Higher = more certain
    n_observations: int = 0      # How many experiences built this prior

    @classmethod
    def uninformative(cls, n_axes: int) -> Prior:
        """
        Start with no opinion. Flat prior.
        The system has seen nothing — every outcome is equally likely.
        """
        return cls(
            mean=torch.zeros(n_axes),
            precision=torch.ones(n_axes) * 0.01,  # Very low confidence
            n_observations=0,
        )

    def update(self, observation: torch.Tensor, observation_precision: float = 1.0) -> float:
        """
        Bayesian conjugate update for Gaussian prior.

        Given a new observation (valence vector), update the prior.
        Returns the surprise (KL divergence between old and new posterior).

        This is the core learning step. Every experience either:
            - Confirms the prior (low surprise → forgettable)
            - Challenges the prior (high surprise → must remember)
        """
        old_mean = self.mean.clone()
        old_precision = self.precision.clone()

        obs_prec = torch.ones_like(self.precision) * observation_precision

        # Conjugate Gaussian update
        new_precision = self.precision + obs_prec
        new_mean = (self.precision * self.mean + obs_prec * observation) / new_precision

        self.mean = new_mean
        self.precision = new_precision
        self.n_observations += 1

        # Surprise: KL divergence between old and new posterior
        # KL(new || old) for diagonal Gaussians
        surprise = self._kl_divergence(new_mean, new_precision, old_mean, old_precision)

        return surprise

    @staticmethod
    def _kl_divergence(
        mu1: torch.Tensor, prec1: torch.Tensor,
        mu2: torch.Tensor, prec2: torch.Tensor,
    ) -> float:
        """KL(N(mu1, prec1^-1) || N(mu2, prec2^-1))"""
        var1 = 1.0 / (prec1 + 1e-8)
        var2 = 1.0 / (prec2 + 1e-8)

        kl = 0.5 * torch.sum(
            torch.log(var2 / var1)
            + var1 / var2
            + (mu2 - mu1) ** 2 / var2
            - 1.0
        )
        return max(0.0, kl.item())

    @property
    def confidence(self) -> float:
        """Overall confidence — geometric mean of per-axis precision."""
        return self.precision.log().mean().exp().item()

    def predict(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        What does the prior predict?
        Returns (expected_valence, uncertainty_per_axis).
        """
        uncertainty = 1.0 / (self.precision + 1e-8)
        return self.mean, uncertainty


class ExperienceMemory:
    """
    Bayesian experience memory with capacity management.

    This is the organism's accumulated knowledge:
        - What visual patterns it has seen
        - What valence states they produced
        - What actions it took and whether they worked
        - How surprised it was (how much it learned)

    When memory is full, the least informative experiences
    are evicted. The system preferentially retains:
        - Surprising experiences (changed beliefs)
        - Intense experiences (high valence activation)
        - Complex patterns (hard to relearn from scratch)
        - Extreme outcomes (very good or very bad)

    Neutral, unsurprising, simple experiences are forgotten.
    They can be relearned cheaply if needed later.
    """

    def __init__(
        self,
        capacity: int = 1000,
        feature_dim: int = 1024,
        similarity_threshold: float = 0.85,
    ):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold

        self.experiences: list[Experience] = []
        self.priors: dict[str, Prior] = {}       # domain → Prior
        self._next_id = 0

    def get_or_create_prior(self, domain: str, n_axes: int) -> Prior:
        if domain not in self.priors:
            self.priors[domain] = Prior.uninformative(n_axes)
        return self.priors[domain]

    def record(
        self,
        feature_vector: torch.Tensor,
        domain: str,
        valence_vector: torch.Tensor,
        action_taken: str,
        outcome: float,
    ) -> Optional[Experience]:
        """
        Record a new experience. The Bayesian loop:

        1. Retrieve the prior for this domain
        2. Update the prior with this observation → compute surprise
        3. If surprise is above threshold OR outcome is extreme → store
        4. If memory is full → evict the least informative experience

        Returns the Experience if stored, None if discarded (too boring).
        """
        prior = self.get_or_create_prior(domain, len(valence_vector))

        # Bayesian update — this IS the learning
        surprise = prior.update(valence_vector)

        # Intensity
        intensity = valence_vector.norm().item()

        # Complexity: how different is this from existing memories?
        complexity = self._compute_complexity(feature_vector)

        # Should we store this?
        store = self._should_store(surprise, intensity, outcome, complexity)

        if not store:
            return None

        exp = Experience(
            id=self._next_id,
            timestamp=time.time(),
            feature_vector=feature_vector.detach().clone(),
            domain=domain,
            valence_vector=valence_vector.detach().clone(),
            action_taken=action_taken,
            outcome=outcome,
            surprise=surprise,
            intensity=intensity,
            complexity=complexity,
            last_accessed=time.time(),
        )
        exp.update_consolidation()
        self._next_id += 1

        # Evict if at capacity
        if len(self.experiences) >= self.capacity:
            self._evict()

        self.experiences.append(exp)
        return exp

    def recall(self, feature_vector: torch.Tensor, top_k: int = 5) -> list[Experience]:
        """
        Recall experiences similar to the current observation.

        This is the retrieval step: "have I seen something like
        this before? What happened? What did I do?"

        Recalled experiences update their access count and timestamp,
        strengthening their consolidation (they're still useful).
        """
        if not self.experiences:
            return []

        similarities = []
        for exp in self.experiences:
            sim = exp.similarity_to(feature_vector)
            similarities.append((sim, exp))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, exp in similarities[:top_k]:
            if sim < 0.1:
                break
            exp.access_count += 1
            exp.last_accessed = time.time()
            exp.update_consolidation()
            results.append(exp)

        return results

    def predict(self, domain: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        What does accumulated experience predict for this domain?

        Returns (expected_valence, uncertainty, confidence).

        High confidence + low uncertainty = strong prior.
        The system has seen many examples and knows what to expect.

        Low confidence + high uncertainty = weak prior.
        The system is unsure. New observations will cause high surprise.
        """
        if domain not in self.priors:
            return None, None, 0.0

        prior = self.priors[domain]
        mean, uncertainty = prior.predict()
        return mean, uncertainty, prior.confidence

    def _compute_complexity(self, feature_vector: torch.Tensor) -> float:
        """
        How complex/unique is this pattern relative to existing memory?

        High complexity = nothing in memory looks like this.
        Low complexity = we've seen very similar patterns before.

        Complex experiences are harder to reconstruct if forgotten,
        so they get higher consolidation scores.
        """
        if not self.experiences:
            return 1.0  # First experience is always novel

        max_sim = max(
            exp.similarity_to(feature_vector) for exp in self.experiences
        )

        # Complexity is inverse of maximum similarity
        return 1.0 - max_sim

    def _should_store(
        self,
        surprise: float,
        intensity: float,
        outcome: float,
        complexity: float,
    ) -> bool:
        """
        Should this experience be stored?

        Store if ANY of:
            - High surprise (changed our beliefs)
            - High intensity (strong valence activation)
            - Extreme outcome (very good or very bad)
            - High complexity (novel pattern)

        Do NOT store if ALL of:
            - Low surprise (confirmed prior)
            - Low intensity (near homeostasis)
            - Neutral outcome
            - Low complexity (seen it before)

        "There is no much point in storing neutral experiences."
        """
        # Adaptive thresholds based on memory fill level
        fill = len(self.experiences) / max(1, self.capacity)
        # As memory fills, become more selective
        selectivity = 1.0 + fill * 2.0

        if surprise > 0.5 / selectivity:
            return True
        if intensity > 5.0 / selectivity:
            return True
        if abs(outcome) > 0.7 / selectivity:
            return True
        if complexity > 0.8 / selectivity:
            return True

        return False

    def _evict(self):
        """
        Evict the least informative experience.

        Update all consolidation scores, then remove the lowest.
        This is the forgetting mechanism — but structured forgetting.
        We forget the boring, the redundant, the simple.
        We keep the surprising, the intense, the unique.
        """
        for exp in self.experiences:
            exp.update_consolidation()

        # Find the experience with lowest consolidation
        worst = min(self.experiences, key=lambda e: e.consolidation_score)
        self.experiences.remove(worst)

    def stats(self) -> dict:
        """Memory statistics — no words, just numbers."""
        if not self.experiences:
            return {
                "n_stored": 0,
                "capacity": self.capacity,
                "fill": 0.0,
                "n_domains": len(self.priors),
            }

        surprises = [e.surprise for e in self.experiences]
        intensities = [e.intensity for e in self.experiences]
        consolidations = [e.consolidation_score for e in self.experiences]
        outcomes = [e.outcome for e in self.experiences]

        return {
            "n_stored": len(self.experiences),
            "capacity": self.capacity,
            "fill": len(self.experiences) / self.capacity,
            "n_domains": len(self.priors),
            "mean_surprise": sum(surprises) / len(surprises),
            "max_surprise": max(surprises),
            "mean_intensity": sum(intensities) / len(intensities),
            "mean_consolidation": sum(consolidations) / len(consolidations),
            "min_consolidation": min(consolidations),
            "mean_outcome": sum(outcomes) / len(outcomes),
            "n_positive_outcomes": sum(1 for o in outcomes if o > 0),
            "n_negative_outcomes": sum(1 for o in outcomes if o < 0),
        }

    def compression_ratio(self) -> float:
        """
        How efficiently is memory being used?

        Ratio of unique information (high-surprise experiences)
        to total stored experiences. Higher is better — means
        we're keeping the informative stuff and discarding the rest.
        """
        if not self.experiences:
            return 0.0

        high_surprise = sum(1 for e in self.experiences if e.surprise > 0.5)
        return high_surprise / len(self.experiences)
