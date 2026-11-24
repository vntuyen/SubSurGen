"""Feature selection.

The CancerSubtypes paper found Cox-model feature selection (FSbyCox) gave the
most prognostically meaningful subtypes, so SubSurGen uses the same step before
running SNF: keep genes whose univariate Cox association with survival passes a
p-value cutoff.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _cox_univariate(x: np.ndarray, time: np.ndarray, event: np.ndarray,
                    max_iter: int = 20, tol: float = 1e-6) -> tuple:
    """Univariate Cox PH via Newton-Raphson (Breslow ties). Returns (beta, p)."""
    order = np.argsort(-time)          # descending time -> cumulative risk sets
    x = x[order]
    e = event[order]
    t = time[order]

    beta = 0.0
    for _ in range(max_iter):
        eta = beta * x
        eta = np.clip(eta, -30, 30)
        w = np.exp(eta)

        # risk set aggregates via cumulative sums over descending time
        S0 = np.cumsum(w)
        S1 = np.cumsum(w * x)
        S2 = np.cumsum(w * x * x)

        # handle ties: all rows with equal time share the same risk set
        uniq_t, first_idx = np.unique(t, return_index=True)
        last_of_group = np.zeros(len(t), dtype=int)
        for gi in range(len(uniq_t)):
            lo = first_idx[gi]
            hi = first_idx[gi + 1] if gi + 1 < len(uniq_t) else len(t)
            last_of_group[lo:hi] = hi - 1

        S0e = S0[last_of_group]
        S1e = S1[last_of_group]
        S2e = S2[last_of_group]

        mask = e == 1
        if mask.sum() == 0:
            return 0.0, 1.0

        grad = np.sum(x[mask] - S1e[mask] / (S0e[mask] + 1e-12))
        hess = -np.sum(S2e[mask] / (S0e[mask] + 1e-12)
                       - (S1e[mask] / (S0e[mask] + 1e-12)) ** 2)

        if abs(hess) < 1e-12:
            return beta, 1.0
        step = grad / hess
        beta_new = beta - step
        if not np.isfinite(beta_new):
            return beta, 1.0
        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    se = np.sqrt(1.0 / (abs(hess) + 1e-12))
    z = beta / (se + 1e-12)
    p = float(2 * stats.norm.sf(abs(z)))
    return float(beta), p


def fs_by_cox(X: np.ndarray, time: np.ndarray, event: np.ndarray,
              cutoff: float = 0.05, min_keep: int = 10) -> np.ndarray:
    """Return indices of genes with univariate Cox p-value < cutoff."""
    p_vals = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        xj = X[:, j]
        if xj.std() < 1e-8:
            continue
        xj = (xj - xj.mean()) / (xj.std() + 1e-8)
        _, p_vals[j] = _cox_univariate(xj, time, event)

    keep = np.where(p_vals < cutoff)[0]
    if len(keep) < min_keep:                     # fall back to the strongest genes
        keep = np.argsort(p_vals)[:min_keep]
    return np.sort(keep), p_vals


def fs_by_variance(X: np.ndarray, topk: int = 40) -> np.ndarray:
    v = X.var(axis=0)
    return np.sort(np.argsort(-v)[:min(topk, X.shape[1])])
