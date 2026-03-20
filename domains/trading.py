"""
Trading domain — visual scoring axes for market dynamics.

A trader watching a chart or a trading floor reacts to visual
patterns: the shape of the candle, the slope, the volume bars.
These are pixel patterns that map to outcome projections.
"""

from ..core.metaclass import DomainBase


class Trading(DomainBase):
    axes = [
        "volatility",          # Rate of price oscillation.
        "trend_strength",      # Directional conviction in the pattern.
        "reversal_signal",     # Patterns suggesting trend exhaustion.
        "volume_anomaly",      # Unusual volume relative to baseline.
        "correlation_break",   # Divergence from expected co-movement.
        "liquidity_stress",    # Visual cues of thinning order book.
    ]
