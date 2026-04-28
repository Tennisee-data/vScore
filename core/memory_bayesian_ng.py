"""
Normal-Gamma posterior — parallel implementation to BayesianPosterior.

Closes three gaps in the unit-precision Gaussian:

1. sum_sq_obs is USED.
   Observation precision tau is a random variable with a Gamma marginal,
   updated from the residual sum of squares. The sufficient statistics
   (n, sum_x, sum_x2) collected today finally feed into the posterior.

2. Posterior predictive is Student-t (heavy-tailed).
   With few observations the model is appropriately humble about tail
   events; with many observations the Student-t collapses onto Gaussian.

3. Surprise and LOO influence are unified Bayesian quantities:
       surprise(x)   = -log p(x | data)
       influence(x)  = -log p(x | data \\ {x})
   Same units, same scale. No ad-hoc importance_weight to mix in.

Per-axis independence retained. Full Normal-Wishart (cross-axis covariance)
is a separate change.

References:
    PRML §2.3.6 (Bishop 2006)        — derivation of Normal-Gamma update
    PRML eq 2.150-2.152              — posterior hyperparameters
    PRML eq 2.160                    — Student-t posterior predictive
    Murphy MLPP §4.6.3.7 (2012)      — equivalent derivation, different notation
    Gelman BDA3 §3.3 (2013)          — alternate parameterization

Numerical verification in tests below (run as a script).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributions as D


class NormalGammaPosterior:
    """
    Per-axis Normal-Gamma conjugate posterior.

    Each axis tracks four hyperparameters (mu, beta, a, b) representing
    the joint posterior over its (mean, precision) pair:
        p(mu, tau) = N(mu | mu_n, (beta_n * tau)^-1) * Gamma(tau | a_n, b_n)

    Closed-form updates from sufficient statistics (n, sum_x, sum_x2),
    so observe / influence_of / surprise_against_current are all O(n_axes).

    External API matches BayesianPosterior so this can drop into
    BayesianMemory without changes to the memory class itself:
        observe(x) -> float                       (surprise)
        influence_of(x) -> float                  (LOO log-score)
        surprise_against_current(x) -> float
        importance_weight(x) -> float             (1.0 — folded into surprise)
        predict() -> (mean, variance)
        confidence -> float
        n_obs -> int
    """

    def __init__(
        self,
        n_axes: int,
        prior_precision: float = 0.01,   # kept for API parity; maps to beta0
        mu0: float = 0.0,
        a0: float = 1.0,
        b0: float = 1.0,
    ):
        self.n_axes = n_axes
        self.mu0 = mu0
        self.beta0 = prior_precision     # weight of prior on the mean
        self.a0 = a0                     # Gamma shape — strength of precision prior
        self.b0 = b0                     # Gamma rate  — E[tau] = a0/b0

        # Sufficient statistics (per axis)
        self.n = 0
        self.sum_x = torch.zeros(n_axes)
        self.sum_x2 = torch.zeros(n_axes)

    @property
    def n_obs(self) -> int:
        return self.n

    def _params(
        self,
        n: int,
        sum_x: torch.Tensor,
        sum_x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Closed-form Normal-Gamma posterior from sufficient statistics.

        From PRML eq 2.150-2.152, derived in §2.3.6:
            beta_n = beta_0 + n
            mu_n   = (beta_0 * mu_0 + sum_x) / beta_n
            a_n    = a_0 + n / 2
            b_n    = b_0 + 0.5 * sum_sq_resid
                          + (beta_0 * n * (x_bar - mu_0)^2) / (2 * beta_n)
        where sum_sq_resid = sum (x_i - x_bar)^2 = sum_x2 - n * x_bar^2.

        The b_n update has two terms:
          - half the within-data scatter (data informs precision)
          - prior-vs-data mean disagreement, scaled by their relative weights.
        """
        beta_n = torch.tensor(float(self.beta0 + n))
        mu_n = (self.beta0 * self.mu0 + sum_x) / beta_n
        a_n = torch.tensor(float(self.a0 + n / 2.0))
        if n > 0:
            x_bar = sum_x / n
            ss_resid = sum_x2 - n * x_bar ** 2                            # >= 0
            ss_resid = ss_resid.clamp(min=0.0)                            # numerical safety
            mean_correction = (self.beta0 * n * (x_bar - self.mu0) ** 2) / (2.0 * beta_n)
        else:
            ss_resid = torch.zeros_like(sum_x)
            mean_correction = torch.zeros_like(sum_x)
        b_n = self.b0 + 0.5 * ss_resid + mean_correction
        return mu_n, beta_n, a_n, b_n

    def _predictive(
        self,
        n: int,
        sum_x: torch.Tensor,
        sum_x2: torch.Tensor,
    ) -> D.StudentT:
        """
        Posterior predictive density: per-axis Student-t.

        From PRML eq 2.160:
            p(x_new | data) = St(x_new | mu_n, lambda_pred, nu_pred)
        with
            nu_pred     = 2 * a_n
            lambda_pred = (a_n * beta_n) / (b_n * (beta_n + 1))
        where lambda is Student-t precision. PyTorch's StudentT takes a
        scale parameter, so:
            scale = sqrt(1 / lambda_pred) = sqrt(b_n * (beta_n + 1) / (a_n * beta_n))
        """
        mu_n, beta_n, a_n, b_n = self._params(n, sum_x, sum_x2)
        df = 2.0 * a_n
        scale = torch.sqrt(b_n * (beta_n + 1.0) / (a_n * beta_n))
        return D.StudentT(df=df, loc=mu_n, scale=scale)

    def observe(self, x: torch.Tensor) -> float:
        """
        Observe a value and return surprise = -log p(x | old data).

        Surprise is computed BEFORE the update, against the predictive
        from the current sufficient statistics. Then the statistics are
        incremented.
        """
        pred = self._predictive(self.n, self.sum_x, self.sum_x2)
        surprise = -pred.log_prob(x).sum().item()
        self.sum_x = self.sum_x + x
        self.sum_x2 = self.sum_x2 + x ** 2
        self.n += 1
        return surprise

    def influence_of(self, x: torch.Tensor) -> float:
        """
        LOO log-predictive: -log p(x | data \\ {x}).

        Identical to surprise except the sufficient statistics have x
        subtracted out. High = the rest of the data predicts x poorly,
        i.e. x carries information the rest lacks.

        Same units (nats per joint observation across axes) as surprise.
        """
        if self.n <= 1:
            return float('inf')
        loo = self._predictive(self.n - 1, self.sum_x - x, self.sum_x2 - x ** 2)
        return -loo.log_prob(x).sum().item()

    def surprise_against_current(self, x: torch.Tensor) -> float:
        """-log p(x | current data). Used by replay against the current state."""
        pred = self._predictive(self.n, self.sum_x, self.sum_x2)
        return -pred.log_prob(x).sum().item()

    def importance_weight(self, x: torch.Tensor) -> float:
        """
        API parity stub — under Normal-Gamma the importance weight is
        already absorbed into surprise_against_current, so this returns 1.0
        and the multiplication in BayesianMemory._replay becomes a no-op.
        """
        return 1.0

    def predict(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Posterior predictive mean and variance.

        Student-t variance (defined when df > 2, i.e. a_n > 1):
            var = scale^2 * df / (df - 2)
                = (b_n (beta_n + 1) / (a_n beta_n)) * (2 a_n) / (2 a_n - 2)
                = b_n (beta_n + 1) / (beta_n (a_n - 1))
        For a_n <= 1 the predictive variance is undefined (heavy tail);
        we report +inf in that regime.
        """
        mu_n, beta_n, a_n, b_n = self._params(self.n, self.sum_x, self.sum_x2)
        if a_n.item() > 1.0:
            var = b_n * (beta_n + 1.0) / (beta_n * (a_n - 1.0))
        else:
            var = torch.full_like(b_n, float('inf'))
        return mu_n.clone(), var

    @property
    def confidence(self) -> float:
        """
        Posterior expected log-precision under the Gamma marginal:
            E[log tau] = digamma(a_n) - log(b_n)
        Higher = more confident about the mean. Per-axis mean reported.
        """
        _, _, a_n, b_n = self._params(self.n, self.sum_x, self.sum_x2)
        log_tau = torch.special.digamma(a_n) - torch.log(b_n.clamp(min=1e-12))
        return log_tau.mean().item()

    def expected_surprise(self) -> float:
        """
        Differential entropy of the predictive Student-t (summed across
        independent axes). This is the expected value of -log p(x|data)
        for x drawn from the predictive itself — i.e. the surprise of a
        typical observation.

        Used by BayesianMemory._should_store to subtract this baseline
        before applying gate thresholds. The threshold then represents
        "additional information beyond what's typical," giving the same
        semantics as the Gaussian/KL gate (which has natural zero
        baseline).
        """
        pred = self._predictive(self.n, self.sum_x, self.sum_x2)
        return pred.entropy().sum().item()


# ---------------------------------------------------------------------------
# Numerical verification — runnable as `python -m vScore.core.memory_bayesian_ng`
# ---------------------------------------------------------------------------

def _verify_loo_consistency(verbose: bool = True) -> None:
    """
    LOO computed from sufficient-stat subtraction must equal LOO computed
    by re-fitting a fresh posterior on n-1 observations. This is the
    cleanest test that the closed-form math is right: both paths must
    produce identical predictive densities for the held-out point.
    """
    torch.manual_seed(0)
    n_axes = 3
    n_obs = 30
    data = torch.randn(n_obs, n_axes) * 0.7 + torch.tensor([0.5, -0.2, 1.0])

    post = NormalGammaPosterior(n_axes)
    for x in data:
        post.observe(x)

    for i in range(n_obs):
        x = data[i]
        # Path A: subtract x from sufficient stats
        loo_fast = post.influence_of(x)

        # Path B: re-fit a fresh posterior on the other n-1 points
        fresh = NormalGammaPosterior(
            n_axes,
            prior_precision=post.beta0, mu0=post.mu0, a0=post.a0, b0=post.b0,
        )
        for j, y in enumerate(data):
            if j != i:
                fresh.observe(y)
        loo_ref = -fresh._predictive(fresh.n, fresh.sum_x, fresh.sum_x2).log_prob(x).sum().item()

        err = abs(loo_fast - loo_ref)
        assert err < 1e-4, f"LOO mismatch at i={i}: fast={loo_fast}, ref={loo_ref}, err={err}"
    if verbose:
        print("[ok] LOO via sufficient-stat subtraction matches re-fit (max err < 1e-4)")


def _verify_predictive_against_monte_carlo(verbose: bool = True) -> None:
    """
    The closed-form Student-t predictive must match the distribution
    obtained by hierarchical sampling:
        tau ~ Gamma(a_n, b_n)
        mu | tau ~ N(mu_n, (beta_n * tau)^-1)
        x | mu, tau ~ N(mu, 1/tau)

    Verified by matching the first two moments AND by Kolmogorov-Smirnov:
    the Monte Carlo CDF at a grid of points should match the closed-form
    Student-t CDF to within sampling noise (O(1/sqrt(N))).
    """
    torch.manual_seed(1)
    n_axes = 1
    n_obs = 20
    data = torch.randn(n_obs, n_axes) * 1.5 + 0.3

    post = NormalGammaPosterior(n_axes)
    for x in data:
        post.observe(x)

    mu_n, beta_n, a_n, b_n = post._params(post.n, post.sum_x, post.sum_x2)

    # Hierarchical sampling — this IS the definition of the predictive
    n_samples = 1_000_000
    tau_samples = D.Gamma(a_n, b_n).sample((n_samples,))
    mu_samples = D.Normal(mu_n, 1.0 / torch.sqrt(beta_n * tau_samples)).sample()
    x_samples = D.Normal(mu_samples, 1.0 / torch.sqrt(tau_samples)).sample().squeeze(-1)

    # Closed-form moments (mean and variance of the Student-t predictive)
    df = 2.0 * a_n
    scale = torch.sqrt(b_n * (beta_n + 1.0) / (a_n * beta_n))
    closed_mean = mu_n.item()
    # Var[Student-t] = scale^2 * df / (df - 2), valid here since df = 22 > 2
    closed_var = (scale ** 2 * df / (df - 2.0)).item()

    emp_mean = x_samples.mean().item()
    emp_var = x_samples.var().item()

    # Sampling SE on mean: sqrt(Var/N) ~ sqrt(closed_var/N)
    se_mean = math.sqrt(closed_var / n_samples)
    mean_z = abs(emp_mean - closed_mean) / se_mean
    # Sampling SE on variance for Gaussian-like data ~ sqrt(2 * Var^2 / N)
    se_var = math.sqrt(2 * closed_var ** 2 / n_samples)
    var_z = abs(emp_var - closed_var) / se_var

    assert mean_z < 5, f"Mean z-score too high: {mean_z:.2f} (emp={emp_mean}, closed={closed_mean})"
    assert var_z < 5, f"Variance z-score too high: {var_z:.2f} (emp={emp_var}, closed={closed_var})"

    # Kolmogorov-Smirnov via scipy (PyTorch's StudentT has no CDF)
    from scipy import stats as _stats
    df_val = df.item()
    loc_val = mu_n.item()
    scale_val = scale.item()
    ks_stat, ks_p = _stats.kstest(
        x_samples.numpy(),
        lambda q: _stats.t.cdf(q, df=df_val, loc=loc_val, scale=scale_val),
    )
    # With N=1e6 a true match has KS stat ~ 1/sqrt(N) ~ 1e-3
    assert ks_stat < 0.01, f"KS statistic too large: {ks_stat:.4f} (p={ks_p:.2e})"

    if verbose:
        print(f"[ok] MC vs Student-t: mean z={mean_z:.2f}, var z={var_z:.2f}, "
              f"KS stat={ks_stat:.4f} (n={n_samples:,})")


def _verify_known_posterior(verbose: bool = True) -> None:
    """
    With a known prior and known data, verify the posterior matches the
    canonical PRML §2.3.6 formulas computed by hand.
    """
    n_axes = 1
    # Prior: mu0=0, beta0=2, a0=3, b0=4
    post = NormalGammaPosterior(n_axes, prior_precision=2.0, mu0=0.0, a0=3.0, b0=4.0)

    # Observe three points: 1.0, 2.0, 3.0
    xs = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    for x in xs:
        post.observe(x)

    # Hand calculation:
    # n = 3, sum_x = 6, sum_x2 = 14
    # beta_n = 2 + 3 = 5
    # mu_n = (2*0 + 6) / 5 = 1.2
    # a_n = 3 + 3/2 = 4.5
    # x_bar = 2.0
    # ss_resid = 14 - 3*4 = 2.0
    # mean_correction = (2 * 3 * (2 - 0)^2) / (2 * 5) = 24/10 = 2.4
    # b_n = 4 + 0.5*2 + 2.4 = 4 + 1 + 2.4 = 7.4
    mu_n, beta_n, a_n, b_n = post._params(post.n, post.sum_x, post.sum_x2)
    tol = 1e-6
    assert abs(beta_n.item() - 5.0) < tol, beta_n
    assert abs(mu_n.item() - 1.2) < tol, mu_n
    assert abs(a_n.item() - 4.5) < tol, a_n
    assert abs(b_n.item() - 7.4) < tol, b_n

    # Predictive variance: b_n (beta_n + 1) / (beta_n (a_n - 1))
    #                    = 7.4 * 6 / (5 * 3.5) = 44.4 / 17.5 = 2.5371...
    _, var = post.predict()
    expected_var = 7.4 * 6.0 / (5.0 * 3.5)
    assert abs(var.item() - expected_var) < 1e-6, var

    if verbose:
        print(f"[ok] Hand-computed posterior matches: "
              f"(mu_n,beta_n,a_n,b_n) = (1.2, 5.0, 4.5, 7.4), pred_var = {expected_var:.4f}")


def _verify_axis_independence(verbose: bool = True) -> None:
    """
    Multivariate diagonal posterior must equal n independent univariate
    posteriors with the same data. This guards against tensor-shape bugs
    in the closed-form update.
    """
    torch.manual_seed(2)
    n_axes = 4
    n_obs = 25
    data = torch.randn(n_obs, n_axes)

    multi = NormalGammaPosterior(n_axes)
    uni = [NormalGammaPosterior(1) for _ in range(n_axes)]

    for x in data:
        multi.observe(x)
        for k in range(n_axes):
            uni[k].observe(x[k:k+1])

    # Compare predictives at a test point
    test_x = torch.tensor([0.3, -0.5, 1.1, 0.0])
    multi_logp_per_axis = multi._predictive(
        multi.n, multi.sum_x, multi.sum_x2,
    ).log_prob(test_x)
    uni_logp = torch.stack([
        uni[k]._predictive(uni[k].n, uni[k].sum_x, uni[k].sum_x2).log_prob(test_x[k:k+1]).squeeze()
        for k in range(n_axes)
    ])

    diff = (multi_logp_per_axis - uni_logp).abs().max().item()
    assert diff < 1e-5, f"Multi vs univariate mismatch: {diff}"
    if verbose:
        print(f"[ok] Multivariate diagonal == n independent univariates "
              f"(max log-prob diff: {diff:.2e})")


def run_verifications():
    print("Running Normal-Gamma numerical verifications...")
    _verify_known_posterior()
    _verify_axis_independence()
    _verify_loo_consistency()
    _verify_predictive_against_monte_carlo()
    print("All verifications passed.")


if __name__ == "__main__":
    run_verifications()
