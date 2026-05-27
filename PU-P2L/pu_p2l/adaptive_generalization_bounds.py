from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist

import numpy as np


_NORMAL = NormalDist()


@dataclass(frozen=True)
class SelfSelectedBoundResult:
    bound: float
    raw_bound: float
    empirical_risk: float
    initial_gap: float
    reciprocal_gap: float
    total_gap: float
    lipschitz_sample: float
    log_cover_constant: float
    concentration_rate_constant: float
    data_diameter: float
    dimension: float
    wasserstein_p: float


@dataclass(frozen=True)
class AdaBoundResult:
    bound: float
    raw_bound: float
    empirical_risk: float
    alpha: float
    sigma: float
    beta: float
    epsilon: float
    delta_sum: float
    gamma_star: float
    query_count: int
    initial_size: int
    final_size: int


def clipped_risk_bound(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def geometric_lipschitz_sum(lipschitz: float, steps: int) -> float:
    steps = max(0, int(steps))
    if steps == 0:
        return 0.0
    if abs(lipschitz - 1.0) < 1e-12:
        return float(steps)
    return float((lipschitz**steps - 1.0) / (lipschitz - 1.0))


def self_selected_generalization_bound(
    empirical_risk: float,
    n: int,
    delta: float,
    steps: int,
    *,
    loss_lipschitz: float,
    data_diameter: float,
    dimension: float,
    wasserstein_p: float = 1.0,
    changed_per_iter: int = 1,
    sample_lipschitz: float | None = None,
    log_cover_constant: float | None = None,
    concentration_rate_constant: float | None = None,
) -> SelfSelectedBoundResult:
    """Anytime self-selected-data generalization gap from Rodemann and Bailie.

    Implements Theorem 15 in "Generalization Bounds and Stopping Rules for
    Learning with Self-Selected Data":

        R(P, theta_T) <= R(P_T, theta_T)
            + L_l (log(C_a / delta) / (C_b n))^(p / d)
            + L_l ((L_s^T - 1) / (L_s - 1)) (m / n)^(1/p) D_Z.

    The constants are explicit inputs because the paper's bound depends on
    problem-specific Lipschitz, covering, Wasserstein, and diameter constants.
    """

    if n <= 0:
        return SelfSelectedBoundResult(
            bound=1.0,
            raw_bound=1.0,
            empirical_risk=float(empirical_risk),
            initial_gap=1.0,
            reciprocal_gap=0.0,
            total_gap=1.0,
            lipschitz_sample=1.0,
            log_cover_constant=0.0,
            concentration_rate_constant=1.0,
            data_diameter=float(data_diameter),
            dimension=float(dimension),
            wasserstein_p=float(wasserstein_p),
        )
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    if loss_lipschitz < 0.0:
        raise ValueError("loss_lipschitz must be nonnegative.")
    if data_diameter <= 0.0:
        raise ValueError("data_diameter must be positive.")
    if dimension <= 0.0:
        raise ValueError("dimension must be positive.")
    if wasserstein_p <= 0.0:
        raise ValueError("wasserstein_p must be positive.")
    if changed_per_iter < 0:
        raise ValueError("changed_per_iter must be nonnegative.")

    if sample_lipschitz is None:
        sample_lipschitz = max(0.0, (float(n) - 1.0) / float(n))
    if log_cover_constant is None:
        log_cover_constant = (dimension / wasserstein_p) * math.log(2.0)
    if concentration_rate_constant is None:
        concentration_rate_constant = 1.0 / (4.0 * data_diameter * data_diameter)
    if concentration_rate_constant <= 0.0:
        raise ValueError("concentration_rate_constant must be positive.")

    initial_inside = (log_cover_constant + math.log(1.0 / delta)) / (
        concentration_rate_constant * float(n)
    )
    initial_gap = loss_lipschitz * max(initial_inside, 0.0) ** (wasserstein_p / dimension)

    distortion = geometric_lipschitz_sum(sample_lipschitz, steps)
    reciprocal_gap = (
        loss_lipschitz
        * distortion
        * (float(changed_per_iter) / float(n)) ** (1.0 / wasserstein_p)
        * data_diameter
    )
    total_gap = initial_gap + reciprocal_gap
    raw_bound = float(empirical_risk) + total_gap
    return SelfSelectedBoundResult(
        bound=clipped_risk_bound(raw_bound),
        raw_bound=float(raw_bound),
        empirical_risk=float(empirical_risk),
        initial_gap=float(initial_gap),
        reciprocal_gap=float(reciprocal_gap),
        total_gap=float(total_gap),
        lipschitz_sample=float(sample_lipschitz),
        log_cover_constant=float(log_cover_constant),
        concentration_rate_constant=float(concentration_rate_constant),
        data_diameter=float(data_diameter),
        dimension=float(dimension),
        wasserstein_p=float(wasserstein_p),
    )


def erfcinv(value: float) -> float:
    """Inverse complementary error function via the normal quantile."""

    clipped = min(max(float(value), 1e-300), 2.0 - 1e-15)
    return float(-_NORMAL.inv_cdf(clipped / 2.0) / math.sqrt(2.0))


def _log_psi(gamma: float, rho: float, epsilon: float) -> float:
    if gamma <= 1.0:
        return math.inf
    return (
        (gamma - 1.0) * (gamma * rho - epsilon)
        + gamma * math.log1p(-1.0 / gamma)
        - math.log(gamma - 1.0)
    )


def _minimize_log_psi(rho: float, epsilon: float) -> tuple[float, float]:
    """Minimize the zCDP-to-DP conversion term over gamma > 1."""

    lo = -12.0
    hi = 12.0
    for _ in range(90):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        gamma_left = 1.0 + math.exp(left)
        gamma_right = 1.0 + math.exp(right)
        if _log_psi(gamma_left, rho, epsilon) < _log_psi(gamma_right, rho, epsilon):
            hi = right
        else:
            lo = left
    mid = (lo + hi) / 2.0
    gamma = 1.0 + math.exp(mid)
    return gamma, _log_psi(gamma, rho, epsilon)


@lru_cache(maxsize=4096)
def ada_clipped_gaussian_alpha(
    final_size: int,
    initial_size: int,
    query_count: int,
    beta_prime: float,
    sigma_min: float,
    sigma_max: float,
    sigma_grid_size: int,
    beta_min: float,
    beta_max: float,
    beta_grid_size: int,
    epsilon_max: float,
    epsilon_grid_size: int,
) -> tuple[float, float, float, float, float, float]:
    """Theorem 4.4 clipped-Gaussian ADA distributional accuracy bound.

    We instantiate the growing-data theorem with all k statistical queries
    released at the final snapshot. This covers the static setting and keeps the
    comparison conservative for the MNIST boundary experiment, where the data is
    not actually arriving online.
    """

    n = max(1, int(final_size))
    n0 = max(1, min(int(initial_size), n))
    k = max(1, int(query_count))
    if not (0.0 < beta_prime < 1.0):
        raise ValueError("beta_prime must be in (0, 1).")
    if sigma_min <= 0.0 or sigma_max <= 0.0 or sigma_min > sigma_max:
        raise ValueError("invalid sigma grid range.")
    if beta_min <= 0.0 or beta_max <= 0.0 or beta_min > beta_max:
        raise ValueError("invalid beta grid range.")
    if epsilon_max < 0.0:
        raise ValueError("epsilon_max must be nonnegative.")

    sigma_values = np.geomspace(sigma_min, sigma_max, max(1, int(sigma_grid_size)))
    beta_values = np.geomspace(beta_min, min(beta_max, 1.0 - 1e-12), max(1, int(beta_grid_size)))
    epsilon_values = np.linspace(0.0, epsilon_max, max(1, int(epsilon_grid_size)))

    best_alpha = math.inf
    best_sigma = float(sigma_values[0])
    best_beta = float(beta_values[0])
    best_epsilon = float(epsilon_values[0])
    best_delta_sum = math.inf
    best_gamma_star = 1.0

    for sigma in sigma_values:
        sigma_f = float(sigma)
        rho = k / (2.0 * sigma_f * sigma_f * n * n)
        snapshot_scale = math.sqrt(2.0) * sigma_f
        for epsilon in epsilon_values:
            epsilon_f = float(epsilon)
            gamma_star, log_delta = _minimize_log_psi(rho, epsilon_f)
            delta_value = 1.0 if log_delta >= 0.0 else math.exp(log_delta)
            delta_sum = n * min(max(delta_value, 0.0), 1.0)
            privacy_term = math.expm1(epsilon_f) + (2.0 * delta_sum) / (n0 * beta_prime)
            for beta in beta_values:
                beta_f = float(beta)
                alpha = (
                    snapshot_scale * erfcinv(beta_f / k)
                    + privacy_term
                    + beta_f / beta_prime
                    + (2.0 / beta_prime) * math.sqrt((2.0 * beta_f * delta_sum) / n0)
                )
                if alpha < best_alpha:
                    best_alpha = float(alpha)
                    best_sigma = sigma_f
                    best_beta = beta_f
                    best_epsilon = epsilon_f
                    best_delta_sum = float(delta_sum)
                    best_gamma_star = float(gamma_star)

    return best_alpha, best_sigma, best_beta, best_epsilon, best_delta_sum, best_gamma_star


def ada_clipped_gaussian_bound(
    empirical_risk: float,
    final_size: int,
    initial_size: int,
    query_count: int,
    beta_prime: float,
    *,
    sigma_min: float = 1e-4,
    sigma_max: float = 1.0,
    sigma_grid_size: int = 14,
    beta_min: float = 1e-12,
    beta_max: float = 0.5,
    beta_grid_size: int = 14,
    epsilon_max: float = 2.0,
    epsilon_grid_size: int = 14,
) -> AdaBoundResult:
    alpha, sigma, beta, epsilon, delta_sum, gamma_star = ada_clipped_gaussian_alpha(
        int(final_size),
        int(initial_size),
        int(query_count),
        float(beta_prime),
        float(sigma_min),
        float(sigma_max),
        int(sigma_grid_size),
        float(beta_min),
        float(beta_max),
        int(beta_grid_size),
        float(epsilon_max),
        int(epsilon_grid_size),
    )
    raw_bound = float(empirical_risk) + alpha
    return AdaBoundResult(
        bound=clipped_risk_bound(raw_bound),
        raw_bound=float(raw_bound),
        empirical_risk=float(empirical_risk),
        alpha=float(alpha),
        sigma=float(sigma),
        beta=float(beta),
        epsilon=float(epsilon),
        delta_sum=float(delta_sum),
        gamma_star=float(gamma_star),
        query_count=max(1, int(query_count)),
        initial_size=max(1, int(initial_size)),
        final_size=max(1, int(final_size)),
    )
