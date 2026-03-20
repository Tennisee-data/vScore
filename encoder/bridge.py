"""
Encoder bridge — connects a visual encoder (V-JEPA 2, etc.)
to the valence scoring system.

The encoder is Level 2: learned feature groups (visual patterns).
This bridge maps those features to Level 0: valence vectors.

The encoder is treated as a black box. It takes video frames and
produces dense feature tensors. The bridge takes those tensors and
regresses them to domain-specific valence scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.metaclass import ScoredDomain


class ValenceHead(nn.Module):
    """
    Maps encoder features to a valence vector for a specific domain.

    This is a regression head, not a classifier. The output is a
    continuous vector of non-negative scores. ReLU enforces the
    constraint that scores can't go below zero (homeostasis floor).
    """

    def __init__(self, encoder_dim: int, domain_name: str, hidden_dim: int = 512):
        super().__init__()

        domain_cls = ScoredDomain.get_domain(domain_name)
        self.domain_name = domain_name
        self.n_axes = domain_cls.n_axes()
        self.axis_names = domain_cls.axis_names()

        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_axes),
            nn.ReLU(),  # Scores are non-negative. Zero is homeostasis.
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: encoder output, shape (batch, encoder_dim)
                      or (batch, time, encoder_dim) for temporal

        Returns:
            valence scores, shape (batch, n_axes)
            or (batch, time, n_axes) for temporal
        """
        return self.head(features)


class vScore(nn.Module):
    """
    Full vScore model: frozen encoder + valence heads per domain.

    The encoder sees pixels. The heads score outcomes.
    Language is not involved anywhere in this pipeline.
    """

    def __init__(self, encoder: nn.Module, encoder_dim: int):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.heads = nn.ModuleDict()

        # Freeze encoder by default — we're learning the scoring, not the seeing
        for param in self.encoder.parameters():
            param.requires_grad = False

    def add_domain(self, domain_name: str, hidden_dim: int = 512):
        """Register a new domain with its own valence head."""
        self.heads[domain_name] = ValenceHead(
            encoder_dim=self.encoder_dim,
            domain_name=domain_name,
            hidden_dim=hidden_dim,
        )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Extract features from the visual encoder."""
        with torch.no_grad():
            return self.encoder(video)

    def score(self, video: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Pixels in, valence vector out.

        video → encoder → features → domain head → valence scores
        """
        features = self.encode(video)

        # Pool spatial/temporal dims if needed (global average)
        if features.dim() > 2:
            features = features.mean(dim=list(range(1, features.dim() - 1)))

        return self.heads[domain](features)

    def score_all_domains(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score a video across all registered domains."""
        features = self.encode(video)

        if features.dim() > 2:
            features = features.mean(dim=list(range(1, features.dim() - 1)))

        return {name: head(features) for name, head in self.heads.items()}
