from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset


def static_quantilize(
    samples: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized quantiles and location-scale features from 1D samples.

    Args:
        samples: [b] -- samples from a 1D distribution.
        K: number of quantile points.

    Returns:
        quantiles: [K] normalized to [-1, 1].
        loc_scale: [3] = [log1p(|mean|), sign(mean), log(sigma + eps)].
    """
    samples_f = samples.to(torch.float32)
    probs = torch.linspace(0.0, 1.0, K, device=samples.device, dtype=torch.float32)
    q = torch.quantile(samples_f, probs)

    q_min = q[0]
    q_max = q[-1]
    span = (q_max - q_min).clamp_min(1e-8)
    q_norm = 2.0 * (q - q_min) / span - 1.0

    mu = samples_f.mean()
    sigma = samples_f.std(unbiased=False)
    eps = 1e-6
    loc_scale = torch.stack([
        torch.log1p(mu.abs()),
        mu.sign(),
        torch.log(sigma + eps),
    ])

    return q_norm, loc_scale


def static_quantilize_batch(
    samples: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched version: samples [B, b] -> quantiles [B, K], loc_scale [B, 3]."""
    B, b = samples.shape
    samples_f = samples.to(torch.float32)
    probs = torch.linspace(0.0, 1.0, K, device=samples.device, dtype=torch.float32)
    q = torch.quantile(samples_f, probs, dim=1)  # [K, B]
    q = q.T  # [B, K]

    q_min = q[:, 0:1]
    q_max = q[:, -1:]
    span = (q_max - q_min).clamp_min(1e-8)
    q_norm = 2.0 * (q - q_min) / span - 1.0

    mu = samples_f.mean(dim=1)
    sigma = samples_f.std(dim=1, unbiased=False)
    eps = 1e-6
    loc_scale = torch.stack([
        torch.log1p(mu.abs()),
        mu.sign(),
        torch.log(sigma + eps),
    ], dim=1)

    return q_norm, loc_scale


def _approx_normal_ppf(p: np.ndarray) -> np.ndarray:
    """Approximate standard normal inverse CDF (no scipy needed).

    Uses rational approximation (Abramowitz & Stegun 26.2.23), accurate to ~4.5e-4.
    """
    p = np.asarray(p, dtype=np.float64)
    # Work with the lower tail
    mask = p > 0.5
    pp = np.where(mask, 1.0 - p, p)
    t = np.sqrt(-2.0 * np.log(np.clip(pp, 1e-15, None)))
    # Rational approximation constants
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return np.where(mask, x, -x).astype(np.float64)


def _try_import_scipy_stats():
    """Try to import scipy.stats, return None if unavailable."""
    try:
        import scipy.stats
        return scipy.stats
    except ImportError:
        return None


def _sample_n_modes(rng: np.random.Generator, max_modes: int, lam: float = 2.5) -> int:
    """Sample number of modes from truncated Poisson(lam) on [1, max_modes].

    With lam=2.5 and max_modes=4:
        P(1) ≈ 0.25, P(2) ≈ 0.32, P(3) ≈ 0.26, P(4) ≈ 0.17
    So 2-3 modes are the most common scenario, 4 is rare but non-zero.
    """
    # Precompute unnormalized probabilities: lam^k / k! for k in [1, max_modes]
    probs = np.empty(max_modes, dtype=np.float64)
    log_factorial = 0.0
    for k in range(1, max_modes + 1):
        log_factorial += np.log(k)
        probs[k - 1] = np.exp(k * np.log(lam) - log_factorial)
    probs /= probs.sum()
    return int(rng.choice(np.arange(1, max_modes + 1), p=probs))


# ---------------------------------------------------------------------------
# Default strategy weight table (name -> default probability)
# ---------------------------------------------------------------------------
DEFAULT_STRATEGY_WEIGHTS: dict[str, float] = {
    "mixture_gauss_t_skew": 0.13,
    "laplace": 0.05,
    "sinh_arcsinh": 0.07,
    "piecewise_linear_qf": 0.05,
    "sparse_dense": 0.07,
    "uniform_segments": 0.04,
    "signed_lognormal": 0.05,
    "beta_transformed": 0.05,
    "alpha_stable": 0.04,
    "quantile_warping": 0.05,
    "contaminated_normal": 0.05,
    # --- 10 new strategies ---
    "generalized_gaussian": 0.04,
    "cauchy_mixture": 0.03,
    "triangular_mixture": 0.04,
    "bimodal_gap": 0.04,
    "truncated_normal": 0.03,
    "folded_normal": 0.03,
    "logistic_mixture": 0.04,
    "raised_cosine": 0.03,
    "staircase_quantized": 0.04,
    "signed_exponential": 0.03,
}


# ---------------------------------------------------------------------------
# Individual strategy samplers
# Each returns np.ndarray of shape (n_samples,) as float32.
# For QF-based strategies (piecewise_linear_qf, quantile_warping) the returned
# array has exactly K elements and represents *quantile values*, not raw samples.
# The caller handles this distinction.
# ---------------------------------------------------------------------------

def _strat_mixture_gauss_t_skew(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    df_range: tuple[float, float] = (2.5, 30.0),
    skew_range: tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """Strategy 0: mixture of Gaussian, Student-t, skew-normal."""
    scipy_stats = _try_import_scipy_stats()

    n_modes = _sample_n_modes(rng, max_modes)
    if scipy_stats is not None:
        comp_probs = np.array([0.4, 0.35, 0.25])
        comp_types = rng.choice(3, size=n_modes, p=comp_probs)
    else:
        # No scipy -> only Gaussian (0) and Student-t (1)
        comp_probs = np.array([0.55, 0.45])
        comp_types = rng.choice(2, size=n_modes, p=comp_probs)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        loc_std = rng.uniform(*loc_std_range)
        mu = rng.normal(0.0, loc_std)
        sigma = rng.uniform(*scale_range)

        ctype = comp_types[i]
        if ctype == 0:
            part = rng.normal(mu, sigma, size=n_i)
        elif ctype == 1:
            df = rng.uniform(*df_range)
            part = mu + sigma * rng.standard_t(df, size=n_i)
        else:
            alpha = rng.uniform(*skew_range)
            part = scipy_stats.skewnorm.rvs(alpha, loc=mu, scale=sigma, size=n_i, random_state=rng)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_laplace(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 1: mixture of Laplace components (L1-regularized weights)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        scale = rng.uniform(*scale_range)
        part = rng.laplace(mu, scale, size=n_i)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.laplace(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_sinh_arcsinh(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 2: mixture of sinh-arcsinh transformed normals.

    Each component has independent skew (epsilon) and tail weight (delta).
    """
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        sigma = rng.uniform(*scale_range)
        epsilon = rng.uniform(-1.5, 1.5)
        delta = rng.uniform(0.3, 2.0)
        z = rng.normal(size=n_i)
        part = mu + sigma * np.sinh((np.arcsinh(z) + epsilon) / delta)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_piecewise_linear_qf(
    rng: np.random.Generator,
    K: int,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 3: directly build a piecewise-linear quantile function.

    Returns K quantile values (NOT raw samples).
    """
    n_knots = rng.integers(5, 16)
    knots_p = np.sort(rng.uniform(0.0, 1.0, size=n_knots))
    knots_p[0] = 0.0
    knots_p[-1] = 1.0

    # Generate sorted knot values (monotonic)
    sigma = rng.uniform(*scale_range)
    raw_v = rng.normal(0.0, sigma, size=n_knots)
    knots_v = np.sort(raw_v)

    # Shift to desired location
    mu = rng.normal(0.0, rng.uniform(*loc_std_range))
    knots_v = knots_v - knots_v.mean() + mu

    probs = np.linspace(0.0, 1.0, K)
    quantiles = np.interp(probs, knots_p, knots_v)
    return quantiles.astype(np.float32)


def _strat_sparse_dense(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 4: sparse spike near zero + mixture of dense Gaussian clusters."""
    p_sparse = rng.uniform(0.05, 0.50)
    n_sparse = max(1, int(p_sparse * n_samples))
    n_dense = n_samples - n_sparse

    sparse_part = rng.normal(0.0, 1e-6, size=n_sparse)

    # Dense part: mixture of n_modes Gaussian clusters
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_dense, weights)

    dense_parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        sigma = rng.uniform(*scale_range)
        dense_parts.append(rng.normal(mu, sigma, size=n_i).astype(np.float32))

    if not dense_parts:
        dense_parts.append(rng.normal(0.0, 0.05, size=n_dense).astype(np.float32))

    return np.concatenate([sparse_part.astype(np.float32)] + dense_parts)


def _strat_uniform_segments(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 5: piecewise-constant density from non-overlapping uniform segments."""
    n_seg = _sample_n_modes(rng, max(max_modes, 2))
    overall_scale = rng.uniform(*scale_range) * 5  # wider for segments

    # Generate non-overlapping intervals
    boundaries = np.sort(rng.uniform(-overall_scale, overall_scale, size=2 * n_seg))
    weights = rng.dirichlet(np.ones(n_seg))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_seg):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        lo = boundaries[2 * i]
        hi = boundaries[2 * i + 1]
        if hi <= lo:
            hi = lo + 1e-6
        part = rng.uniform(lo, hi, size=n_i)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.uniform(-overall_scale, overall_scale, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_signed_lognormal(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 6: mixture of signed lognormal components."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        sigma_log = rng.uniform(0.3, 1.5)
        mu_log = rng.uniform(-3.0, -1.0)
        overall_scale = rng.uniform(*scale_range) * 3
        center = rng.normal(0.0, rng.uniform(*loc_std_range))

        signs = rng.choice([-1.0, 1.0], size=n_i)
        magnitudes = rng.lognormal(mu_log, sigma_log, size=n_i)
        part = center + signs * magnitudes * overall_scale
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_beta_transformed(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 7: mixture of rescaled Beta distributions (bounded, flexible shape)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        alpha = rng.uniform(0.3, 5.0)
        beta_param = rng.uniform(0.3, 5.0)
        width = rng.uniform(*scale_range) * 10
        center = rng.normal(0.0, rng.uniform(*loc_std_range))
        raw = rng.beta(alpha, beta_param, size=n_i)
        part = center - width / 2 + width * raw
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.beta(2.0, 2.0, size=n_samples).astype(np.float32) * 0.1
    return np.concatenate(parts)


def _strat_alpha_stable(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 8: mixture of alpha-stable distributions (power-law tails)."""
    scipy_stats = _try_import_scipy_stats()

    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        sigma = rng.uniform(*scale_range)

        if scipy_stats is not None:
            alpha = rng.uniform(1.2, 2.0)
            beta_skew = rng.uniform(-0.5, 0.5)
            part = scipy_stats.levy_stable.rvs(
                alpha, beta_skew, loc=mu, scale=sigma,
                size=n_i, random_state=rng,
            )
        else:
            df = rng.uniform(1.5, 3.0)
            part = mu + sigma * rng.standard_t(df, size=n_i)

        parts.append(np.clip(part, -2.0, 2.0).astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_quantile_warping(
    rng: np.random.Generator,
    K: int,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 9: warp a base Gaussian QF with random monotonic transforms.

    Returns K quantile values (NOT raw samples).
    """
    probs = np.linspace(0.01, 0.99, K)
    # Approximate Gaussian PPF without scipy using Beasley-Springer-Moro
    base_q = _approx_normal_ppf(probs)

    # Power warp: q -> sign(q) * |q|^gamma
    gamma = rng.uniform(0.3, 3.0)
    warped = np.sign(base_q) * np.abs(base_q) ** gamma

    # Sigmoid compression
    a = rng.uniform(0.5, 5.0)
    warped = np.tanh(a * warped)

    # Scale and shift to realistic range
    sigma = rng.uniform(*scale_range)
    mu = rng.normal(0.0, rng.uniform(*loc_std_range))
    warped = mu + sigma * warped / (np.abs(warped).max() + 1e-8) * 3.0

    return warped.astype(np.float32)


def _strat_contaminated_normal(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 10: mixture of Gaussians with outlier injection per component.

    Each of 1-max_modes base Gaussians gets 1-10% outliers from a wider distribution,
    possibly at a shifted location (creating distinct outlier clusters).
    """
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        sigma = rng.uniform(*scale_range)

        p_outlier = rng.uniform(0.01, 0.10)
        n_outlier = max(1, int(p_outlier * n_i))
        n_clean = n_i - n_outlier

        clean = rng.normal(mu, sigma, size=n_clean)
        outlier_scale = sigma * rng.uniform(5.0, 20.0)
        # Outliers can be at a shifted location
        outlier_shift = rng.normal(0.0, rng.uniform(*loc_std_range))
        outliers = rng.normal(mu + outlier_shift, outlier_scale, size=n_outlier)
        parts.append(np.concatenate([clean, outliers]).astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Strategies 11-20 (new batch)
# ---------------------------------------------------------------------------

def _strat_generalized_gaussian(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 11: mixture of Generalized Gaussian (exponential power) distributions.

    Shape parameter beta: beta=1 -> Laplace, beta=2 -> Gaussian,
    beta<1 -> super-heavy, beta>2 -> sub-Gaussian (lighter tails).
    Sampling via: sign * Gamma(1/beta, 1)^(1/beta).
    """
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        sigma = rng.uniform(*scale_range)
        beta_shape = rng.uniform(0.5, 4.0)
        inv_beta = 1.0 / beta_shape
        gamma_samples = rng.gamma(inv_beta, 1.0, size=n_i)
        signs = rng.choice([-1.0, 1.0], size=n_i)
        part = mu + sigma * signs * np.power(gamma_samples, inv_beta)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_cauchy_mixture(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 12: mixture of Cauchy components (undefined mean/variance)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        scale = rng.uniform(*scale_range)
        part = mu + scale * rng.standard_cauchy(size=n_i)
        parts.append(part.astype(np.float32))

    if not parts:
        return (rng.standard_cauchy(size=n_samples) * 0.05).astype(np.float32)
    samples = np.concatenate(parts)
    # Clip extreme outliers to stay in realistic range
    return np.clip(samples, -2.0, 2.0).astype(np.float32)


def _strat_triangular_mixture(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 13: mixture of triangular distributions (piecewise-linear density)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        center = rng.normal(0.0, rng.uniform(*loc_std_range))
        half_width = rng.uniform(*scale_range) * 3
        left = center - half_width * rng.uniform(0.5, 1.5)
        right = center + half_width * rng.uniform(0.5, 1.5)
        # Mode can be anywhere between left and right
        mode = rng.uniform(left, right)
        part = rng.triangular(left, mode, right, size=n_i)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.triangular(-0.1, 0.0, 0.1, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_bimodal_gap(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 14: n_modes tight clusters with gaps between them (quantized weights)."""
    n_modes = _sample_n_modes(rng, max_modes)
    # Ensure at least 2 clusters for the "gap" effect
    n_modes = max(n_modes, 2)

    center = rng.normal(0.0, rng.uniform(*loc_std_range))
    total_span = rng.uniform(0.05, 0.3)
    cluster_width = rng.uniform(0.001, 0.02)

    # Place cluster centers evenly across the span
    if n_modes == 1:
        cluster_centers = np.array([center])
    else:
        cluster_centers = np.linspace(
            center - total_span / 2, center + total_span / 2, n_modes
        )

    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        part = rng.normal(cluster_centers[i], cluster_width, size=n_i)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(center, 0.01, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_truncated_normal(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 15: mixture of truncated Gaussians (weight/gradient clipping)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        sigma = rng.uniform(*scale_range)
        n_sigma = rng.uniform(1.5, 3.5)
        a = mu - n_sigma * sigma
        b = mu + n_sigma * sigma

        accepted: list[np.ndarray] = []
        collected = 0
        while collected < n_i:
            batch = rng.normal(mu, sigma, size=n_i * 2)
            ok = batch[(batch >= a) & (batch <= b)]
            accepted.append(ok)
            collected += len(ok)
        part = np.concatenate(accepted)[:n_i]
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_folded_normal(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 16: mixture of folded normals |N(mu, sigma)| with random sign."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        fold_mu = rng.uniform(0.0, 0.1)
        sigma = rng.uniform(*scale_range)
        p_neg = rng.uniform(0.3, 0.7)
        raw = np.abs(rng.normal(fold_mu, sigma, size=n_i))
        signs = rng.choice([-1.0, 1.0], size=n_i, p=[p_neg, 1.0 - p_neg])
        center = rng.normal(0.0, rng.uniform(*loc_std_range))
        part = center + signs * raw
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_logistic_mixture(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 17: mixture of Logistic distributions (slightly heavier than Gaussian)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        scale = rng.uniform(*scale_range)
        part = rng.logistic(mu, scale, size=n_i)
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.logistic(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _invert_raised_cosine_cdf(u: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Numerically invert the raised cosine CDF via bisection.

    CDF: F(x) = 0.5 * (1 + (x-mu)/s + sin(pi*(x-mu)/s) / pi)  for x in [mu-s, mu+s].
    """
    lo = np.full_like(u, mu - s)
    hi = np.full_like(u, mu + s)

    for _ in range(40):  # 40 bisection steps -> ~1e-12 precision
        mid = (lo + hi) / 2.0
        t = (mid - mu) / s
        cdf_mid = 0.5 * (1.0 + t + np.sin(np.pi * t) / np.pi)
        lo = np.where(cdf_mid < u, mid, lo)
        hi = np.where(cdf_mid < u, hi, mid)

    return (lo + hi) / 2.0


def _strat_raised_cosine(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 18: mixture of raised cosine distributions (smooth bounded bells)."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        mu = rng.normal(0.0, rng.uniform(*loc_std_range))
        s = rng.uniform(*scale_range) * 5
        u = rng.uniform(0.0, 1.0, size=n_i)
        part = _invert_raised_cosine_cdf(u, mu, s)
        parts.append(part.astype(np.float32))

    if not parts:
        u = rng.uniform(0.0, 1.0, size=n_samples)
        return _invert_raised_cosine_cdf(u, 0.0, 0.1).astype(np.float32)
    return np.concatenate(parts)


def _strat_staircase_quantized(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 19: mixture of k-bit quantized staircase groups + jitter.

    Each mode is a separate staircase (own levels, own center), modeling
    multi-group quantized weights (e.g. per-channel quantization).
    """
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        k_bits = rng.integers(2, 5)
        n_levels = 2 ** k_bits
        scale = rng.uniform(*scale_range) * 5
        center = rng.normal(0.0, rng.uniform(*loc_std_range))

        levels = np.linspace(center - scale, center + scale, n_levels)
        level_probs = rng.dirichlet(np.ones(n_levels) * 2.0)
        chosen = rng.choice(levels, size=n_i, p=level_probs)

        level_spacing = 2 * scale / (n_levels - 1) if n_levels > 1 else scale
        jitter_std = level_spacing * rng.uniform(0.05, 0.25)
        jitter = rng.normal(0.0, jitter_std, size=n_i)
        parts.append((chosen + jitter).astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


def _strat_signed_exponential(
    rng: np.random.Generator,
    n_samples: int,
    max_modes: int = 5,
    loc_std_range: tuple[float, float] = (0.01, 0.3),
    scale_range: tuple[float, float] = (0.001, 0.15),
    **_kw,
) -> np.ndarray:
    """Strategy 20: mixture of signed exponential pairs with asymmetric rates."""
    n_modes = _sample_n_modes(rng, max_modes)
    weights = rng.dirichlet(np.ones(n_modes))
    counts = rng.multinomial(n_samples, weights)

    parts: list[np.ndarray] = []
    for i in range(n_modes):
        n_i = int(counts[i])
        if n_i == 0:
            continue
        scale = rng.uniform(*scale_range)
        rate_pos = rng.uniform(5.0, 50.0)
        rate_neg = rng.uniform(5.0, 50.0)
        p_pos = rng.uniform(0.3, 0.7)
        n_pos = max(1, int(p_pos * n_i))
        n_neg = n_i - n_pos
        pos_part = rng.exponential(scale / rate_pos, size=n_pos)
        neg_part = -rng.exponential(scale / rate_neg, size=n_neg)
        center = rng.normal(0.0, rng.uniform(*loc_std_range))
        part = np.concatenate([pos_part, neg_part]) + center
        parts.append(part.astype(np.float32))

    if not parts:
        return rng.normal(0.0, 0.05, size=n_samples).astype(np.float32)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Dispatcher: pick a strategy and sample
# ---------------------------------------------------------------------------
_SAMPLE_STRATEGIES = {
    "mixture_gauss_t_skew": _strat_mixture_gauss_t_skew,
    "laplace": _strat_laplace,
    "sinh_arcsinh": _strat_sinh_arcsinh,
    "piecewise_linear_qf": _strat_piecewise_linear_qf,
    "sparse_dense": _strat_sparse_dense,
    "uniform_segments": _strat_uniform_segments,
    "signed_lognormal": _strat_signed_lognormal,
    "beta_transformed": _strat_beta_transformed,
    "alpha_stable": _strat_alpha_stable,
    "quantile_warping": _strat_quantile_warping,
    "contaminated_normal": _strat_contaminated_normal,
    # --- 10 new strategies ---
    "generalized_gaussian": _strat_generalized_gaussian,
    "cauchy_mixture": _strat_cauchy_mixture,
    "triangular_mixture": _strat_triangular_mixture,
    "bimodal_gap": _strat_bimodal_gap,
    "truncated_normal": _strat_truncated_normal,
    "folded_normal": _strat_folded_normal,
    "logistic_mixture": _strat_logistic_mixture,
    "raised_cosine": _strat_raised_cosine,
    "staircase_quantized": _strat_staircase_quantized,
    "signed_exponential": _strat_signed_exponential,
}

# Strategies that return quantile vectors directly (not raw samples)
_QF_STRATEGIES = {"piecewise_linear_qf", "quantile_warping"}


def sample_synthetic_distribution(
    rng: np.random.Generator,
    n_samples: int,
    K: int,
    strategy_weights: dict[str, float] | None = None,
    strategy: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample from a random synthetic distribution.

    Args:
        rng: numpy random generator.
        n_samples: number of raw samples to draw (ignored for QF strategies).
        K: number of quantile points.
        strategy_weights: name->weight dict for random strategy selection.
        strategy: if set, force this specific strategy.
        **kwargs: passed to the strategy function (loc_std_range, scale_range, etc.).

    Returns:
        (raw_samples, quantiles_override)
        - For sample-based strategies: raw_samples shape (n_samples,), quantiles_override is None
        - For QF strategies: raw_samples is None, quantiles_override shape (K,)
    """
    if strategy is None:
        weights_dict = strategy_weights or DEFAULT_STRATEGY_WEIGHTS
        names = list(weights_dict.keys())
        probs = np.array([weights_dict[n] for n in names], dtype=np.float64)
        probs /= probs.sum()
        strategy = rng.choice(names, p=probs)

    fn = _SAMPLE_STRATEGIES[strategy]

    if strategy in _QF_STRATEGIES:
        quantiles = fn(rng, K=K, **kwargs)
        return None, quantiles
    else:
        raw = fn(rng, n_samples=n_samples, **kwargs)
        return raw, None


class SyntheticDistributionDataset(IterableDataset):
    """Infinite iterable dataset of synthetic 1D distribution samples.

    Supports 21 distribution generation strategies, randomly selected per sample.

    Each item is a dict with:
        quantiles: [K] -- normalized quantiles in [-1, 1]
        loc_scale: [3] -- [log1p(|mean|), sign(mean), log(sigma+eps)]
        real_samples: [n_real] -- actual samples from the distribution (for critic)
    """

    def __init__(
        self,
        K: int = 64,
        b: int = 1024,
        n_real: int = 64,
        max_modes: int = 5,
        loc_std_range: tuple[float, float] = (0.01, 0.3),
        scale_range: tuple[float, float] = (0.001, 0.15),
        df_range: tuple[float, float] = (2.5, 30.0),
        skew_range: tuple[float, float] = (-5.0, 5.0),
        strategy_weights: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.K = K
        self.b = b
        self.n_real = n_real
        self.max_modes = max_modes
        self.loc_std_range = loc_std_range
        self.scale_range = scale_range
        self.df_range = df_range
        self.skew_range = skew_range
        self.strategy_weights = strategy_weights
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = self.seed + worker_info.id
        else:
            seed = self.seed
        rng = np.random.default_rng(seed)

        common_kwargs = dict(
            max_modes=self.max_modes,
            loc_std_range=self.loc_std_range,
            scale_range=self.scale_range,
            df_range=self.df_range,
            skew_range=self.skew_range,
        )

        while True:
            total_samples = self.b + self.n_real
            raw, qf_override = sample_synthetic_distribution(
                rng,
                n_samples=total_samples,
                K=self.K,
                strategy_weights=self.strategy_weights,
                **common_kwargs,
            )

            if qf_override is not None:
                # QF strategy: quantiles are given directly, generate real_samples
                # by inverse-transform sampling from the quantile function
                q_values = qf_override  # [K]
                probs_grid = np.linspace(0.0, 1.0, self.K)

                # Normalize quantiles to [-1, 1]
                q_min = q_values[0]
                q_max = q_values[-1]
                span = max(q_max - q_min, 1e-8)
                q_norm = 2.0 * (q_values - q_min) / span - 1.0

                mu = float(q_values.mean())
                sigma = float(q_values.std())
                eps = 1e-6
                loc_scale_np = np.array([
                    np.log1p(abs(mu)),
                    np.sign(mu) if mu != 0 else 0.0,
                    np.log(sigma + eps),
                ], dtype=np.float32)

                # Generate real samples via inverse-transform sampling
                u = rng.uniform(0.0, 1.0, size=self.n_real)
                real_samples_np = np.interp(u, probs_grid, q_values).astype(np.float32)

                yield {
                    "quantiles": torch.from_numpy(q_norm.astype(np.float32)),
                    "loc_scale": torch.from_numpy(loc_scale_np),
                    "real_samples": torch.from_numpy(real_samples_np),
                }
            else:
                # Sample-based strategy
                raw_t = torch.from_numpy(raw)
                quantile_samples = raw_t[: self.b]
                real_samples = raw_t[self.b :]

                quantiles, loc_scale = static_quantilize(quantile_samples, self.K)

                yield {
                    "quantiles": quantiles,
                    "loc_scale": loc_scale,
                    "real_samples": real_samples,
                }
