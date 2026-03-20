"""
Modality-agnostic valence scoring.

The central claim: the valence scoring mechanism is INDEPENDENT
of input modality. Vision and audition are different sensors
feeding the same evaluation system.

    Visual encoder → feature vector → valence head → valence vector
    Audio encoder  → feature vector → valence head → valence vector
                                                      ↓
                                              SAME action space
                                              SAME trajectory projection
                                              SAME threshold triggering
                                              SAME Bayesian memory

The valence vector is the universal interface. Downstream of it,
nothing knows or cares whether the input was pixels or pressure waves.

This is biologically real:
    - A loud bang and a flash of light both produce FEAR activation
    - A soft voice and a familiar face both produce CARE activation
    - The amygdala receives BOTH visual and auditory input and
      produces the SAME threat response regardless of modality
    - Multisensory integration happens IN valence space, not
      in sensor space

The architecture:
    ┌─────────────┐     ┌─────────────┐
    │ Visual      │     │ Audio       │
    │ Encoder     │     │ Encoder     │
    │ (V-JEPA 2)  │     │ (BEATs/     │
    │             │     │  AudioMAE)  │
    └──────┬──────┘     └──────┬──────┘
           │                   │
     (D_v features)      (D_a features)
           │                   │
    ┌──────┴──────┐     ┌──────┴──────┐
    │ Visual      │     │ Audio       │
    │ Valence     │     │ Valence     │
    │ Head        │     │ Head        │
    └──────┬──────┘     └──────┴──────┘
           │                   │
     (N_axes scores)     (N_axes scores)
           │                   │
           └────────┬──────────┘
                    │
              FUSION in valence space
              (not in feature space!)
                    │
              ┌─────┴─────┐
              │ Fused      │
              │ Valence    │
              │ Vector     │
              └─────┬──────┘
                    │
              Same downstream:
              - Action inference
              - Trajectory projection
              - Threshold triggering
              - Bayesian memory
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn

from .valence import ValenceVector


class Modality(Enum):
    VISION = auto()
    AUDIO = auto()
    TACTILE = auto()   # Future
    OLFACTORY = auto() # Future


@dataclass
class ModalityConfig:
    """Configuration for a sensory modality."""
    modality: Modality
    encoder_dim: int           # Output dimension of the modality's encoder
    temporal_resolution: float # Seconds per frame/window


class MultimodalValenceHead(nn.Module):
    """
    Per-modality valence heads that output to the SAME valence space.

    Each modality has its own encoder and its own head, but they
    all produce vectors in the same N_axes dimensional space.
    This is the key constraint: the axes are domain-specific,
    not modality-specific.

    A fire's threat is threat whether you see the flames or hear
    the roar. The visual head and the audio head must agree on the
    threat axis — that's what training enforces.
    """

    def __init__(self, n_axes: int, modality_configs: dict[Modality, ModalityConfig]):
        super().__init__()
        self.n_axes = n_axes

        self.heads = nn.ModuleDict()
        for modality, config in modality_configs.items():
            self.heads[modality.name] = nn.Sequential(
                nn.Linear(config.encoder_dim, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, n_axes),
                nn.ReLU(),  # Non-negative: zero is homeostasis
            )

    def forward(
        self,
        features: dict[Modality, torch.Tensor],
    ) -> dict[Modality, torch.Tensor]:
        """
        Score each modality independently.

        Returns per-modality valence vectors. Fusion happens
        downstream, not here — because the fusion rule itself
        is informative (do modalities agree or conflict?).
        """
        results = {}
        for modality, feat in features.items():
            if modality.name in self.heads:
                results[modality] = self.heads[modality.name](feat)
        return results


class ValenceFusion:
    """
    Fuse multi-modal valence vectors in valence space.

    NOT feature fusion. We don't concatenate audio and visual features
    and run them through a joint network. Instead, each modality
    independently scores the same axes, and then we fuse the SCORES.

    Why? Because modality agreement/conflict is itself a signal:
        - Vision says threat=2, audio says threat=8 →
          something dangerous is nearby but occluded (audio threat
          exceeds visual threat → search for the source)
        - Vision says threat=8, audio says threat=2 →
          the threat is visible but quiet (visual threat exceeds
          audio → keep eyes on it but don't panic)
        - Both say threat=8 →
          confirmed danger, high confidence, act immediately

    Fusion rules:
        1. MAX fusion: take the highest score per axis.
           Conservative — if ANY modality signals danger, respond.
           This is the survival default.

        2. MEAN fusion: average across modalities.
           Moderate — requires cross-modal confirmation.

        3. BAYESIAN fusion: weight by modality confidence.
           Principled — modalities with sharper priors contribute more.

        4. CONFLICT-AWARE fusion: detect modality disagreement.
           Diagnostic — high conflict = something interesting is happening.
    """

    @staticmethod
    def max_fusion(
        modality_scores: dict[Modality, torch.Tensor],
    ) -> torch.Tensor:
        """
        Take the maximum score per axis across modalities.
        If ANY sense says danger, the organism responds.
        """
        stacked = torch.stack(list(modality_scores.values()))
        return stacked.max(dim=0).values

    @staticmethod
    def mean_fusion(
        modality_scores: dict[Modality, torch.Tensor],
    ) -> torch.Tensor:
        """Average across modalities."""
        stacked = torch.stack(list(modality_scores.values()))
        return stacked.mean(dim=0)

    @staticmethod
    def confidence_weighted_fusion(
        modality_scores: dict[Modality, torch.Tensor],
        confidences: dict[Modality, float],
    ) -> torch.Tensor:
        """
        Weight each modality by its confidence (posterior precision).
        A modality that has seen many examples of this pattern
        contributes more than one that is uncertain.
        """
        total_conf = sum(confidences.values())
        if total_conf == 0:
            return ValenceFusion.mean_fusion(modality_scores)

        fused = torch.zeros_like(next(iter(modality_scores.values())))
        for modality, scores in modality_scores.items():
            weight = confidences.get(modality, 1.0) / total_conf
            fused += scores * weight
        return fused

    @staticmethod
    def modality_conflict(
        modality_scores: dict[Modality, torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        """
        Compute per-axis conflict between modalities.

        High conflict on an axis = modalities disagree about
        that outcome dimension. This is diagnostic:
            - Occluded threat (audio high, visual low)
            - Deceptive signal (visual threat, audio calm)
            - Sensory illusion (modalities contradict)

        Returns (per_axis_conflict, overall_conflict).
        """
        if len(modality_scores) < 2:
            scores = next(iter(modality_scores.values()))
            return torch.zeros_like(scores), 0.0

        stacked = torch.stack(list(modality_scores.values()))
        # Per-axis standard deviation across modalities
        per_axis = stacked.std(dim=0)
        overall = per_axis.mean().item()

        return per_axis, overall
