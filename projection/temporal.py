"""
Temporal projection module.

The core predictive mechanism. Given a sequence of valence vectors,
project where each axis is heading. This is what separates reactive
intelligence from predictive intelligence.

The gazelle doesn't wait for the lion to arrive. It reads the
trajectory and acts on the projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalProjector(nn.Module):
    """
    Learns to project valence trajectories forward in time.

    Input: sequence of valence vectors (batch, seq_len, n_axes)
    Output: projected valence vector at t+horizon (batch, n_axes)

    Uses a small transformer to capture nonlinear temporal dynamics.
    The linear projection in ValenceTrajectory is the baseline;
    this module learns the real dynamics from data.
    """

    def __init__(
        self,
        n_axes: int,
        seq_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        self.n_axes = n_axes
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_axes, d_model)

        # Learnable positional encoding for temporal order
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_axes),
            nn.ReLU(),  # Projected scores are non-negative
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (batch, seq_len, n_axes) — recent valence history

        Returns:
            projected: (batch, n_axes) — where each axis is heading
        """
        x = self.input_proj(trajectory)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)

        # Use the last position's output for projection
        x = x[:, -1, :]
        return self.output_proj(x)
