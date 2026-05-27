from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=None)
def p2l_bound(k: int, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    k = int(max(0, k))
    if k >= n:
        return 1.0

    log_m_choose_k = np.array(
        [
            math.lgamma(m + 1) - math.lgamma(m - k + 1) - math.lgamma(k + 1)
            for m in range(k, n)
        ],
        dtype=np.float64,
    )
    log_n_choose_k = math.lgamma(n + 1) - math.lgamma(n - k + 1) - math.lgamma(k + 1)
    coeffs = log_m_choose_k - log_n_choose_k
    m_vec = np.arange(k, n, dtype=np.float64)

    t_low = 0.0
    t_high = 1.0
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        while t_high - t_low > 1e-10:
            t = (t_low + t_high) / 2.0
            log_t = math.log(max(t, 1e-15))
            terms = np.exp(coeffs - (n - m_vec) * log_t)
            value = 1.0 - (delta / n) * float(np.sum(terms))
            if value > 0:
                t_high = t
            else:
                t_low = t
    return float(1.0 - t_low)


def binary_kl(q: float, p: float) -> float:
    eps = 1e-12
    q = min(max(float(q), eps), 1.0 - eps)
    p = min(max(float(p), eps), 1.0 - eps)
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def inv_kl_upper(q: float, c: float) -> float:
    q = min(max(float(q), 0.0), 1.0)
    if q >= 1.0:
        return 1.0
    if c <= 0:
        return q
    low = q
    high = 1.0 - 1e-12
    for _ in range(80):
        mid = (low + high) / 2.0
        if binary_kl(q, mid) > c:
            high = mid
        else:
            low = mid
    return float(min(max(high, q), 1.0))


def gaussian_kl_isotropic(
    posterior_mean: torch.Tensor,
    prior_mean: torch.Tensor,
    posterior_sigma: float,
    prior_sigma: float,
) -> float:
    if posterior_sigma <= 0 or prior_sigma <= 0:
        raise ValueError("PAC-Bayes Gaussian sigmas must be positive.")
    post = posterior_mean.detach().cpu().double()
    prior = prior_mean.detach().cpu().double()
    if post.numel() != prior.numel():
        raise ValueError("Posterior and prior vectors must have the same number of parameters.")
    diff_sq = torch.sum((post - prior) ** 2).item()
    dim = post.numel()
    variance_ratio = (posterior_sigma / prior_sigma) ** 2
    log_ratio = 2.0 * math.log(prior_sigma / posterior_sigma)
    return float(0.5 * (dim * (variance_ratio - 1.0 + log_ratio) + diff_sq / (prior_sigma**2)))


def pac_bayes_bound(
    empirical_risk: float,
    kl: float,
    n: int,
    delta: float,
    mc_samples: int,
    delta_test: float,
) -> tuple[float, float]:
    if n <= 0:
        return 1.0, 1.0
    empirical_risk = min(max(float(empirical_risk), 0.0), 1.0)
    mc_samples = max(1, int(mc_samples))
    mc_upper = inv_kl_upper(empirical_risk, math.log(2.0 / delta_test) / mc_samples)
    complexity = (float(kl) + math.log((2.0 * math.sqrt(n)) / delta)) / n
    return inv_kl_upper(mc_upper, complexity), mc_upper
