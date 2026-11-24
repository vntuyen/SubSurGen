"""Downstream survival model (for train-on-synthetic / test-on-real) and baselines."""
from __future__ import annotations

import numpy as np
import torch

from .metrics import concordance_index
from .model import cox_ph_loss


def fit_cox(X: np.ndarray, time: np.ndarray, event: np.ndarray,
            l2: float = 1e-2, epochs: int = 250, lr: float = 0.05, seed: int = 0):
    """Ridge-penalised Cox PH fitted by gradient descent."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(time, dtype=torch.float32)
    ee = torch.tensor(event, dtype=torch.float32)
    if ee.sum() < 2:
        return np.zeros(X.shape[1])

    beta = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([beta], lr=lr)
    for _ in range(epochs):
        risk = Xt @ beta
        loss = cox_ph_loss(risk, tt, ee) + l2 * (beta ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return beta.detach().numpy()


def cox_cindex(beta: np.ndarray, X: np.ndarray, time: np.ndarray,
               event: np.ndarray) -> float:
    return concordance_index(time, event, X @ beta)


# ------------------------------------------------------------------ baselines --
# All baselines share the signature
#     fn(X, time, event, n, rng, labels=None) -> (X_syn, time_syn, event_syn)
# so they are interchangeable in the evaluation harness. `labels` carries the SNF
# subtype assignment; subtype-agnostic baselines simply ignore it.

def _allocate(labels: np.ndarray, n: int, rng) -> tuple:
    """Split n synthetic samples across subtypes in proportion to real sizes."""
    uniq, counts = np.unique(labels, return_counts=True)
    props = counts / counts.sum()
    alloc = np.floor(props * n).astype(int)
    remainder = n - alloc.sum()
    if remainder > 0:
        for k in rng.choice(len(uniq), size=remainder, p=props):
            alloc[k] += 1
    return uniq, alloc


def baseline_marginal(X, time, event, n, rng, labels=None):
    """Independent per-gene resampling: correct marginals, destroys correlation."""
    Xs = np.column_stack([rng.choice(X[:, j], size=n, replace=True)
                          for j in range(X.shape[1])])
    idx = rng.integers(0, len(time), size=n)
    return Xs, time[idx], event[idx]


def baseline_bootstrap(X, time, event, n, rng, labels=None):
    """Plain bootstrap resampling of real patients."""
    idx = rng.integers(0, len(time), size=n)
    return X[idx], time[idx], event[idx]


def baseline_gaussian(X, time, event, n, rng, labels=None):
    """Multivariate Gaussian fitted to expression; survival bootstrapped."""
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    Xs = np.clip(rng.multivariate_normal(mu, cov, size=n), 0, 1)
    idx = rng.integers(0, len(time), size=n)
    return Xs, time[idx], event[idx]


def baseline_smote(X, time, event, n, rng, labels=None, k: int = 5):
    """SMOTE-style interpolation between nearest neighbours (subtype-agnostic)."""
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    kk = min(k, max(1, len(X) - 1))
    nn = np.argsort(D, axis=1)[:, :kk]
    i = rng.integers(0, len(X), size=n)
    j = nn[i, rng.integers(0, kk, size=n)]
    w = rng.uniform(0, 1, size=(n, 1))
    return X[i] * (1 - w) + X[j] * w, time[i], event[i]


# --------------------------------------------------- added Tier-1 competitors --
def baseline_mixup(X, time, event, n, rng, labels=None):
    """Unconstrained mixup: interpolate RANDOM pairs, ignoring subtype structure.

    Isolates whether SubSurGen's benefit comes from interpolation in general or
    specifically from interpolating *within* a subtype on the SNF manifold.
    """
    i = rng.integers(0, len(X), size=n)
    j = rng.integers(0, len(X), size=n)
    w = rng.uniform(0, 1, size=(n, 1))
    Xs = X[i] * (1 - w) + X[j] * w
    # survival is taken from whichever parent dominates the mix
    from_i = w.ravel() < 0.5
    ts = np.where(from_i, time[i], time[j])
    es = np.where(from_i, event[i], event[j])
    return Xs, ts, es


def baseline_subtype_smote(X, time, event, n, rng, labels=None, k: int = 5):
    """SMOTE applied WITHIN each SNF subtype, then pooled.

    A strong, cheap competitor: if this recovers most of SubSurGen's subtype
    consistency, the learned generator is not earning its complexity.
    """
    if labels is None:
        return baseline_smote(X, time, event, n, rng, k=k)
    from scipy.spatial.distance import cdist

    uniq, alloc = _allocate(labels, n, rng)
    Xs, ts, es = [], [], []
    for lab, m in zip(uniq, alloc):
        if m == 0:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:                                   # degenerate cluster
            pick = rng.choice(idx, size=m, replace=True)
            Xs.append(X[pick]); ts.append(time[pick]); es.append(event[pick])
            continue
        Xg, tg, eg = X[idx], time[idx], event[idx]
        D = cdist(Xg, Xg)
        np.fill_diagonal(D, np.inf)
        kk = min(k, len(idx) - 1)
        nn = np.argsort(D, axis=1)[:, :kk]
        i = rng.integers(0, len(idx), size=m)
        j = nn[i, rng.integers(0, kk, size=m)]
        w = rng.uniform(0, 1, size=(m, 1))
        Xs.append(Xg[i] * (1 - w) + Xg[j] * w)
        ts.append(tg[i]); es.append(eg[i])
    return np.vstack(Xs), np.concatenate(ts), np.concatenate(es)


def baseline_subtype_gaussian(X, time, event, n, rng, labels=None):
    """A separate multivariate Gaussian fitted within each SNF subtype."""
    if labels is None:
        return baseline_gaussian(X, time, event, n, rng)

    uniq, alloc = _allocate(labels, n, rng)
    Xs, ts, es = [], [], []
    for lab, m in zip(uniq, alloc):
        if m == 0:
            continue
        idx = np.where(labels == lab)[0]
        Xg, tg, eg = X[idx], time[idx], event[idx]
        if len(idx) < 3:
            pick = rng.choice(idx, size=m, replace=True)
            Xs.append(X[pick]); ts.append(time[pick]); es.append(event[pick])
            continue
        mu = Xg.mean(0)
        # ridge keeps the covariance invertible when a subtype is small
        cov = np.cov(Xg, rowvar=False) + np.eye(Xg.shape[1]) * 1e-4
        Xs.append(np.clip(rng.multivariate_normal(mu, cov, size=m), 0, 1))
        pick = rng.integers(0, len(idx), size=m)
        ts.append(tg[pick]); es.append(eg[pick])
    return np.vstack(Xs), np.concatenate(ts), np.concatenate(es)


#: name -> callable, in the order they should appear in result tables
BASELINES = {
    "Marginal": baseline_marginal,
    "Gaussian": baseline_gaussian,
    "Bootstrap": baseline_bootstrap,
    "SMOTE": baseline_smote,
    "Mixup": baseline_mixup,
    "SubtypeSMOTE": baseline_subtype_smote,
    "SubtypeGaussian": baseline_subtype_gaussian,
}

#: subset used for the downstream C-index comparison (step 4)
DOWNSTREAM_BASELINES = ["SMOTE", "Gaussian", "Bootstrap", "Mixup",
                        "SubtypeSMOTE", "SubtypeGaussian"]


# --------------------------------------- survival curve prediction (for IBS) --
def breslow_baseline_survival(X, time, event, beta):
    """Breslow estimator of the baseline cumulative hazard / survival.

        H0(t) = sum_{t_i <= t, d_i=1}  d_i / sum_{j in R(t_i)} exp(beta' x_j)
        S0(t) = exp(-H0(t))

    Returns (event_times, S0) evaluated at each distinct event time.
    """
    eta = np.clip(X @ beta, -30, 30)
    w = np.exp(eta)
    order = np.argsort(time)
    t_s, e_s, w_s = time[order], event[order], w[order]

    event_times, H, cum = [], [], 0.0
    n = len(t_s)
    i = 0
    while i < n:
        ti = t_s[i]
        j = i
        d = 0
        while j < n and t_s[j] == ti:
            d += int(e_s[j] == 1)
            j += 1
        if d > 0:
            risk_sum = w_s[i:].sum()
            if risk_sum > 1e-12:
                cum += d / risk_sum
                event_times.append(ti)
                H.append(cum)
        i = j

    if not event_times:
        return np.array([0.0]), np.array([1.0])
    return np.asarray(event_times), np.exp(-np.asarray(H))


def predict_survival_curves(beta, X_train, t_train, e_train, X_test, grid):
    """Predicted survival S(t|x) = S0(t) ** exp(beta'x) on `grid`.

    Returns an (n_test, len(grid)) matrix suitable for brier_score().
    """
    et, S0 = breslow_baseline_survival(X_train, t_train, e_train, beta)
    idx = np.clip(np.searchsorted(et, grid, side="right") - 1, 0, len(S0) - 1)
    S0_grid = np.where(grid < et[0], 1.0, S0[idx])
    risk = np.exp(np.clip(X_test @ beta, -30, 30))
    return np.clip(S0_grid[None, :] ** risk[:, None], 0.0, 1.0)
