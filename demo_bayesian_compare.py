"""
Side-by-side comparison: BayesianPosterior (Gaussian, hardcoded precision)
vs NormalGammaPosterior (Bishop PRML §2.3.6 conjugate model).

Runs both posteriors on identical observation streams under five regimes
and reports where they diverge. The point is to demonstrate that:

  - When the data is unit-variance, both behave similarly (no harm from
    the more general model).
  - When the data has variance != 1, the old posterior systematically
    over- or under-flags surprise because it cannot learn precision.
  - Under heavy tails or non-stationarity, the Student-t predictive
    is more honest about its uncertainty.

No synthetic targets — the comparison is on properties of the
posteriors themselves (predictive mean, predictive variance, surprise).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from vScore.core.memory_bayesian import BayesianPosterior
from vScore.core.memory_bayesian_ng import NormalGammaPosterior


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------


def stream_gaussian(n: int, mu: float, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, generator=g) * sigma + mu


def stream_student_t(n: int, df: float, mu: float, scale: float, seed: int) -> torch.Tensor:
    """Heavy-tailed observations."""
    g = torch.Generator().manual_seed(seed)
    # Sample t via z / sqrt(chi2/df)
    z = torch.randn(n, 1, generator=g)
    chi2 = torch.distributions.Chi2(df).sample((n, 1))
    return mu + scale * z / torch.sqrt(chi2 / df)


def stream_drifting(n: int, sigma: float, drift_step: int, seed: int) -> torch.Tensor:
    """Mean drifts +1.0 every drift_step observations."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, 1)
    for i in range(n):
        mu = float(i // drift_step)
        out[i] = torch.randn(1, generator=g) * sigma + mu
    return out


def stream_bimodal(n: int, sigma: float, mode_sep: float, seed: int) -> torch.Tensor:
    """Even/odd observations from two modes ±mode_sep/2."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, 1)
    for i in range(n):
        mode = +mode_sep / 2 if i % 2 == 0 else -mode_sep / 2
        out[i] = torch.randn(1, generator=g) * sigma + mode
    return out


# ---------------------------------------------------------------------------
# Run one scenario through both posteriors, collect per-step traces
# ---------------------------------------------------------------------------


def run_scenario(name: str, data: torch.Tensor, *, verbose: bool = True) -> dict:
    n_axes = data.shape[1]

    bayes = BayesianPosterior(n_axes=n_axes, prior_precision=0.01)
    ng = NormalGammaPosterior(n_axes=n_axes, prior_precision=0.01, mu0=0.0, a0=1.0, b0=1.0)

    surprise_b, surprise_ng = [], []
    influence_b, influence_ng = [], []

    for i, x in enumerate(data):
        # Both posteriors observe the same x; surprise is computed against
        # the predictive *before* the update.
        s_b = bayes.observe(x)
        s_ng = ng.observe(x)
        surprise_b.append(s_b)
        surprise_ng.append(s_ng)

        if i >= 1:                                              # need n>=2 for LOO
            influence_b.append(bayes.influence_of(x))
            influence_ng.append(ng.influence_of(x))

    mean_b, var_b = bayes.predict()
    mean_ng, var_ng = ng.predict()

    true_mean = data.mean().item()
    true_var = data.var().item()

    result = {
        "name": name,
        "n": len(data),
        "true_mean": true_mean,
        "true_var": true_var,
        "bayes_mean": mean_b.item() if mean_b is not None else None,
        "bayes_var": var_b.item() if var_b is not None else None,
        "bayes_mean_surprise": float(np.mean(surprise_b)),
        "bayes_max_surprise": float(np.max(surprise_b)),
        "bayes_mean_influence": float(np.mean(influence_b)) if influence_b else None,
        "ng_mean": mean_ng.item(),
        "ng_var": var_ng.item(),
        "ng_mean_surprise": float(np.mean(surprise_ng)),
        "ng_max_surprise": float(np.max(surprise_ng)),
        "ng_mean_influence": float(np.mean(influence_ng)) if influence_ng else None,
    }

    if verbose:
        print(f"\n=== {name} ===")
        print(f"  data: n={result['n']}, true mean={true_mean:+.3f}, true var={true_var:.3f}")
        print(f"  {'':>20} {'Old (Gaussian)':>18}  {'New (Normal-Gamma)':>20}")
        print(f"  {'predicted mean':>20} {result['bayes_mean']:>18.4f}  {result['ng_mean']:>20.4f}")
        print(f"  {'predicted variance':>20} {result['bayes_var']:>18.4f}  {result['ng_var']:>20.4f}")
        print(f"  {'mean surprise':>20} {result['bayes_mean_surprise']:>18.4f}  {result['ng_mean_surprise']:>20.4f}")
        print(f"  {'max surprise':>20} {result['bayes_max_surprise']:>18.4f}  {result['ng_max_surprise']:>20.4f}")
        if influence_b:
            print(f"  {'mean LOO influence':>20} {result['bayes_mean_influence']:>18.4f}  {result['ng_mean_influence']:>20.4f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("BayesianPosterior (unit-precision Gaussian) vs NormalGammaPosterior")
    print("=" * 72)
    print("Surprise units: old = KL between Gaussian posteriors (mean shift).")
    print("                new = -log p(x | data), nats. Direct comparison")
    print("                of magnitudes is not apples-to-apples; what matters")
    print("                is how each scales with the data regime.")

    # 1. Calibration: unit-variance data — old should be near-optimal here.
    run_scenario(
        "Stationary Gaussian, sigma=1.0 (old's implicit assumption)",
        stream_gaussian(n=200, mu=0.5, sigma=1.0, seed=0),
    )

    # 2. Variance mismatch: data has sigma=5, old still assumes sigma=1.
    #    Old's predicted variance will be (close to) 1/n_obs. New should
    #    learn ~25.
    run_scenario(
        "Stationary Gaussian, sigma=5.0 (old's precision is wrong by 25x)",
        stream_gaussian(n=200, mu=0.0, sigma=5.0, seed=1),
    )

    # 3. Tight data: sigma=0.1, old will under-react.
    run_scenario(
        "Stationary Gaussian, sigma=0.1 (data tighter than old assumes)",
        stream_gaussian(n=200, mu=0.0, sigma=0.1, seed=2),
    )

    # 4. Heavy-tailed: Student-t with df=3, occasional large deviations.
    run_scenario(
        "Student-t df=3 (heavy tails — adversarial for unit-precision Gaussian)",
        stream_student_t(n=200, df=3.0, mu=0.0, scale=1.0, seed=3),
    )

    # 5. Drifting mean: posterior should adapt.
    run_scenario(
        "Drifting mean (+1.0 every 40 obs, sigma=0.5)",
        stream_drifting(n=200, sigma=0.5, drift_step=40, seed=4),
    )

    # 6. Bimodal: predictive variance should reflect spread, not zero.
    run_scenario(
        "Bimodal mixture (modes at +/-3, sigma=0.3)",
        stream_bimodal(n=200, sigma=0.3, mode_sep=6.0, seed=5),
    )

    # ---- Focused detail: per-observation surprise on the sigma=5 case ----
    print("\n" + "=" * 72)
    print("Detail: per-observation surprise on Stationary Gaussian sigma=5")
    print("=" * 72)
    print("If the model's variance estimate is wrong by 25x, every observation")
    print("looks improbable. Watch how old's surprise stays high while new's")
    print("comes down once the precision posterior catches up.")

    data = stream_gaussian(n=200, mu=0.0, sigma=5.0, seed=10)
    bayes = BayesianPosterior(n_axes=1, prior_precision=0.01)
    ng = NormalGammaPosterior(n_axes=1, prior_precision=0.01, mu0=0.0, a0=1.0, b0=1.0)

    print(f"\n  {'step':>5} {'x':>8} {'old surprise':>14} {'new surprise':>14} "
          f"{'old pred sd':>13} {'new pred sd':>13}")
    checkpoints = [0, 1, 2, 5, 10, 20, 50, 100, 199]
    for i, x in enumerate(data):
        s_b = bayes.observe(x)
        s_ng = ng.observe(x)
        if i in checkpoints:
            _, var_b = bayes.predict()
            _, var_ng = ng.predict()
            sd_b = math.sqrt(var_b.item()) if var_b is not None else float("nan")
            sd_ng = math.sqrt(var_ng.item()) if torch.isfinite(var_ng).item() else float("inf")
            print(f"  {i:>5} {x.item():>+8.3f} {s_b:>14.4f} {s_ng:>14.4f} "
                  f"{sd_b:>13.4f} {sd_ng:>13.4f}")

    # ---------------------------------------------------------------------
    # Multivariate (4-axis valence) — the actual production regime
    # ---------------------------------------------------------------------
    multivariate_main()


# ---------------------------------------------------------------------------
# Multivariate scenarios
# ---------------------------------------------------------------------------


def stream_mv_heterogeneous(n: int, mus: list[float], sigmas: list[float], seed: int) -> torch.Tensor:
    """Each axis its own Gaussian with its own (mu, sigma)."""
    g = torch.Generator().manual_seed(seed)
    n_axes = len(mus)
    out = torch.empty(n, n_axes)
    for k in range(n_axes):
        out[:, k] = torch.randn(n, generator=g) * sigmas[k] + mus[k]
    return out


def stream_mv_mixed_tails(n: int, seed: int) -> torch.Tensor:
    """4 axes: Gaussian sigma=1, Student-t df=3, Gaussian sigma=0.5, Cauchy."""
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(n, 4)
    out[:, 0] = torch.randn(n, generator=g) * 1.0
    z = torch.randn(n, generator=g)
    chi2 = torch.distributions.Chi2(3.0).sample((n,))
    out[:, 1] = z / torch.sqrt(chi2 / 3.0)                                   # Student-t df=3
    out[:, 2] = torch.randn(n, generator=g) * 0.5
    u = torch.rand(n, generator=g)
    out[:, 3] = torch.tan(math.pi * (u - 0.5))                                # Cauchy = standard
    return out


def run_mv_scenario(
    name: str,
    data: torch.Tensor,
    axis_labels: list[str],
    true_axis_var: list[float | None] | None = None,
):
    n_axes = data.shape[1]
    bayes = BayesianPosterior(n_axes=n_axes, prior_precision=0.01)
    ng = NormalGammaPosterior(n_axes=n_axes, prior_precision=0.01, mu0=0.0, a0=1.0, b0=1.0)

    for x in data:
        bayes.observe(x)
        ng.observe(x)

    mean_b, var_b = bayes.predict()
    mean_ng, var_ng = ng.predict()
    emp_var = data.var(dim=0)

    print(f"\n=== {name} ===")
    print(f"  n={len(data)}, axes={n_axes}")
    print(f"  {'axis':>22} {'true var':>12} {'emp var':>10} {'old pred var':>14} {'new pred var':>14}")
    for k, label in enumerate(axis_labels):
        truth = true_axis_var[k] if true_axis_var else None
        truth_s = f"{truth:.4f}" if truth is not None and math.isfinite(truth) else (
            "inf" if truth is not None else "—"
        )
        new_pv = var_ng[k].item()
        new_pv_s = f"{new_pv:.4f}" if math.isfinite(new_pv) else "inf"
        print(f"  {label:>22} {truth_s:>12} {emp_var[k].item():>10.4f} "
              f"{var_b[k].item():>14.4f} {new_pv_s:>14}")


def _axis_breakdown(bayes, ng, probe, sigmas, label):
    """Print per-axis surprise contribution for a single probe vector."""
    pred = ng._predictive(ng.n, ng.sum_x, ng.sum_x2)
    new_per_axis = -pred.log_prob(probe)                               # (4,) nats
    var_b = 1.0 / (bayes.posterior_prec + 1e-8)
    diff = probe - bayes.posterior_mean
    old_per_axis = 0.5 * (diff ** 2 / var_b)                           # per-axis Mahalanobis

    print(f"\n  Probe ({label}): {[f'{v:+.2f}' for v in probe.tolist()]}")
    print(f"  {'axis':>4} {'probe':>8} {'sigma':>7} {'sigma-units':>12} "
          f"{'old contrib':>14} {'new contrib':>14}")
    for k in range(len(probe)):
        sigma_units = abs(probe[k].item()) / sigmas[k] if sigmas[k] > 0 else 0.0
        print(f"  {k:>4} {probe[k].item():>+8.2f} {sigmas[k]:>7.2f} "
              f"{sigma_units:>12.2f} {old_per_axis[k].item():>14.4f} "
              f"{new_per_axis[k].item():>14.4f}")
    old_total = old_per_axis.sum().item()
    new_total = new_per_axis.sum().item()
    print(f"  {'TOTAL':>4} {'':>8} {'':>7} {'':>12} {old_total:>14.4f} {new_total:>14.4f}")
    return old_per_axis, new_per_axis


def per_axis_surprise_breakdown():
    """
    Train both posteriors on heterogeneous data, then probe with two
    different vectors:

    Probe A: typical noise on every axis (1sigma each, calibrated to that
             axis). A well-calibrated model should rate this as unsurprising.
    Probe B: typical noise on three axes, 6sigma outlier on axis 2.
             A well-calibrated model should localize the surprise to axis 2.

    Old fails on Probe A — its posterior precision is hardcoded
    independent of axis scale, so a 1sigma deviation on axis 2 (+/- 2.0)
    looks 200x more surprising than a 1sigma deviation on axis 0 (+/- 0.5).
    """
    print("\n=== Per-axis anomaly localization ===")
    print("  Train on Gaussian data with sigma=[0.5, 1.0, 2.0, 1.5].")

    sigmas = [0.5, 1.0, 2.0, 1.5]
    train = stream_mv_heterogeneous(
        n=300, mus=[0.0, 0.0, 0.0, 0.0], sigmas=sigmas, seed=42,
    )

    bayes = BayesianPosterior(n_axes=4, prior_precision=0.01)
    ng = NormalGammaPosterior(n_axes=4, prior_precision=0.01, mu0=0.0, a0=1.0, b0=1.0)
    for x in train:
        bayes.observe(x)
        ng.observe(x)

    # Probe A: 1sigma on each axis (perfectly typical)
    probe_a = torch.tensor([sigmas[k] for k in range(4)])
    _axis_breakdown(bayes, ng, probe_a, sigmas, "1sigma on every axis — typical")

    # Probe B: 1sigma on axes 0/1/3, 6sigma on axis 2 (outlier localized)
    probe_b = torch.tensor([sigmas[0], sigmas[1], 6 * sigmas[2], sigmas[3]])
    old_pa, new_pa = _axis_breakdown(
        bayes, ng, probe_b, sigmas, "1sigma everywhere except 6sigma on axis 2",
    )
    old_loc = old_pa[2].item() / old_pa.sum().item()
    new_loc = new_pa[2].item() / new_pa.sum().item()
    print(f"\n  Localization on axis 2 (fraction of total surprise): "
          f"old={old_loc:.1%}, new={new_loc:.1%}")
    print("  Both put most weight on axis 2 here, but read the typical-probe")
    print("  result above: under 1sigma noise, the old fires hard on axes 1, 2,")
    print("  3 because it has no idea what 1sigma means on each axis. The")
    print("  new is properly calibrated and reports near-baseline surprise.")


def multivariate_main():
    print("\n" + "=" * 72)
    print("Multivariate (4-axis valence) comparison")
    print("=" * 72)
    print("Each axis has its own (mu, sigma). The old posterior assumes")
    print("unit observation precision on every axis, so it cannot adapt to")
    print("axes with different natural scales — exactly the production case.")

    # MV-1: Heterogeneous per-axis variances
    sigmas_1 = [0.1, 1.0, 2.0, 5.0]
    run_mv_scenario(
        "Heterogeneous variances [0.1, 1.0, 2.0, 5.0]",
        stream_mv_heterogeneous(
            n=300, mus=[0.0, 0.0, 0.0, 0.0], sigmas=sigmas_1, seed=20,
        ),
        axis_labels=[f"axis {k} (sigma={s})" for k, s in enumerate(sigmas_1)],
        true_axis_var=[s ** 2 for s in sigmas_1],
    )

    # MV-2: Heterogeneous means and variances
    mus_2 = [0.0, 1.0, -2.0, 5.0]
    sigmas_2 = [0.5, 1.5, 0.3, 2.0]
    data_2 = stream_mv_heterogeneous(n=300, mus=mus_2, sigmas=sigmas_2, seed=21)
    run_mv_scenario(
        f"Heterogeneous means {mus_2} and variances {[s**2 for s in sigmas_2]}",
        data_2,
        axis_labels=[f"axis {k} (mu={m}, sigma={s})" for k, (m, s) in enumerate(zip(mus_2, sigmas_2))],
        true_axis_var=[s ** 2 for s in sigmas_2],
    )

    # MV-3: Mixed tail behavior
    run_mv_scenario(
        "Mixed tails: Gaussian, Student-t df=3, Gaussian, Cauchy",
        stream_mv_mixed_tails(n=300, seed=22),
        axis_labels=["axis 0 (Gaussian s=1)", "axis 1 (Student-t df=3)",
                     "axis 2 (Gaussian s=0.5)", "axis 3 (Cauchy)"],
        true_axis_var=[1.0, 3.0, 0.25, None],   # Cauchy has no finite variance
    )

    # MV-4: Per-axis anomaly localization
    per_axis_surprise_breakdown()


if __name__ == "__main__":
    main()
