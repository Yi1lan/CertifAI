from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


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
