"""Evaluation metrics for SubSurGen.

Covers three families:
  * survival utility      -> Harrell C-index, log-rank test
  * subtype fidelity      -> ARI / NMI between intended and recovered subtypes
  * generative fidelity   -> MMD, Kaplan-Meier divergence, at-risk balance,
                             optimism and short-sightedness (SurvivalGAN metrics)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ----------------------------------------------------------------- survival --
def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index (vectorised). `risk` is higher = worse prognosis."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    risk = np.asarray(risk, float)

    ev = event == 1
    if ev.sum() == 0:
        return 0.5
    # comparable pairs: i had an event, j survived longer
    valid = ev[:, None] & (time[None, :] > time[:, None])
    den = valid.sum()
    if den == 0:
        return 0.5
    gt = (risk[:, None] > risk[None, :]) & valid
    eq = (risk[:, None] == risk[None, :]) & valid
    return float((gt.sum() + 0.5 * eq.sum()) / den)


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple:
    """Multi-group log-rank test. Returns (chi2, p_value, df)."""
    groups = np.unique(group)
    g = len(groups)
    if g < 2:
        return 0.0, 1.0, 0

    times = np.unique(time[event == 1])
    O = np.zeros(g)
    E = np.zeros(g)
    V = np.zeros((g, g))

    for t in times:
        at_risk = time >= t
        n_total = at_risk.sum()
        d_total = ((time == t) & (event == 1)).sum()
        if n_total <= 1 or d_total == 0:
            continue
        n_k = np.array([(at_risk & (group == gg)).sum() for gg in groups], float)
        d_k = np.array([((time == t) & (event == 1) & (group == gg)).sum()
                        for gg in groups], float)
        e_k = n_k * d_total / n_total
        O += d_k
        E += e_k
        f = d_total * (n_total - d_total) / (n_total ** 2 * (n_total - 1))
        for a in range(g):
            for b in range(g):
                if a == b:
                    V[a, b] += f * n_k[a] * (n_total - n_k[a])
                else:
                    V[a, b] -= f * n_k[a] * n_k[b]

    d = (O - E)[:-1]
    Vr = V[:-1, :-1]
    try:
        chi2 = float(d @ np.linalg.pinv(Vr) @ d)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, g - 1
    df = g - 1
    p = float(stats.chi2.sf(chi2, df))
    return chi2, p, df


def kaplan_meier(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """KM survival curve evaluated on `grid`."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    n = len(t)
    surv = 1.0
    curve_t, curve_s = [0.0], [1.0]
    i = 0
    while i < n:
        ti = t[i]
        d = 0
        j = i
        while j < n and t[j] == ti:
            d += int(e[j] == 1)
            j += 1
        at_risk = n - i
        if d > 0 and at_risk > 0:
            surv *= (1.0 - d / at_risk)
        curve_t.append(ti)
        curve_s.append(surv)
        i = j
    curve_t = np.array(curve_t)
    curve_s = np.array(curve_s)
    idx = np.searchsorted(curve_t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(curve_s) - 1)
    return curve_s[idx]


# ------------------------------------------------- generative survival gaps --
def km_divergence(t_real, e_real, t_syn, e_syn, n_grid: int = 100) -> float:
    """Mean absolute gap between real and synthetic KM curves."""
    hi = max(t_real.max(), t_syn.max())
    grid = np.linspace(0, hi, n_grid)
    return float(np.mean(np.abs(kaplan_meier(t_real, e_real, grid)
                                - kaplan_meier(t_syn, e_syn, grid))))


def optimism(t_real, e_real, t_syn, e_syn, n_grid: int = 100) -> float:
    """Signed area between curves. >0 = synthetic too optimistic."""
    hi = max(t_real.max(), t_syn.max())
    grid = np.linspace(0, hi, n_grid)
    return float(np.mean(kaplan_meier(t_syn, e_syn, grid)
                         - kaplan_meier(t_real, e_real, grid)))


def short_sightedness(t_real: np.ndarray, t_syn: np.ndarray) -> float:
    """1 - (synthetic horizon / real horizon). >0 = synthetic stops too early."""
    return float(1.0 - (t_syn.max() / (t_real.max() + 1e-10)))


def at_risk_balance(t_real, t_syn, n_grid: int = 50) -> float:
    """Mean |at-risk fraction difference| across the time grid."""
    hi = max(t_real.max(), t_syn.max())
    grid = np.linspace(0, hi, n_grid)
    ar_r = np.array([(t_real >= g).mean() for g in grid])
    ar_s = np.array([(t_syn >= g).mean() for g in grid])
    return float(np.mean(np.abs(ar_r - ar_s)))


# --------------------------------------------------------- covariate quality --
def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Maximum mean discrepancy with an RBF kernel."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)

    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def correlation_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean |difference| between real and synthetic gene-gene correlation matrices."""
    cr = np.corrcoef(X, rowvar=False)
    cs = np.corrcoef(Y, rowvar=False)
    cr = np.nan_to_num(cr)
    cs = np.nan_to_num(cs)
    iu = np.triu_indices_from(cr, k=1)
    return float(np.mean(np.abs(cr[iu] - cs[iu])))


# ------------------------------------------------- calibration: Brier score --
def _km_censoring(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier estimate of the CENSORING distribution G(t) = P(C > t).

    Obtained by running the KM estimator with the event indicator flipped, so
    censorings are treated as the events of interest. Used for IPCW weights.
    """
    return kaplan_meier(time, 1.0 - event, grid)


def brier_score(time_train, event_train, time_test, event_test,
                surv_pred: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Time-dependent Brier score with IPCW censoring correction (Graf et al. 1999).

        BS(t) = 1/n sum_i [ S(t|x_i)^2 * 1{T_i <= t, d_i=1} / G(T_i)
                          + (1 - S(t|x_i))^2 * 1{T_i > t} / G(t) ]

    Parameters
    ----------
    surv_pred : (n_test, len(grid)) predicted survival probabilities
    grid      : evaluation times

    Returns
    -------
    (len(grid),) Brier score at each time. Lower is better; 0.25 is the score of
    an uninformative model that always predicts S = 0.5.
    """
    n = len(time_test)
    # censoring distribution estimated on the TRAINING data
    G_grid = _km_censoring(time_train, event_train, grid)
    G_Ti = _km_censoring(time_train, event_train, time_test)
    G_grid = np.clip(G_grid, 1e-3, None)
    G_Ti = np.clip(G_Ti, 1e-3, None)

    bs = np.zeros(len(grid))
    for m, t in enumerate(grid):
        S = np.clip(surv_pred[:, m], 0.0, 1.0)
        died_before = (time_test <= t) & (event_test == 1)
        still_alive = time_test > t
        term1 = np.where(died_before, (S ** 2) / G_Ti, 0.0)
        term2 = np.where(still_alive, ((1.0 - S) ** 2) / G_grid[m], 0.0)
        bs[m] = float((term1 + term2).sum() / n)
    return bs


def integrated_brier_score(time_train, event_train, time_test, event_test,
                           surv_pred: np.ndarray, grid: np.ndarray) -> float:
    """IBS = (1 / (t_max - t_min)) * integral of BS(t) dt, by the trapezoid rule."""
    bs = brier_score(time_train, event_train, time_test, event_test,
                     surv_pred, grid)
    span = grid[-1] - grid[0]
    if span <= 0:
        return float(bs.mean())
    return float(np.trapezoid(bs, grid) / span) if hasattr(np, "trapezoid") \
        else float(np.trapz(bs, grid) / span)


# ------------------------------------------------------------ detection AUC --
def detection_auc(X_real: np.ndarray, X_syn: np.ndarray, seed: int = 0,
                  n_splits: int = 5) -> float:
    """Can a classifier tell real from synthetic? (a.k.a. propensity / detection score)

    Trains a gradient-boosted classifier with cross-validation to separate real from
    synthetic rows and reports mean ROC-AUC.

        0.5 -> indistinguishable (ideal)
        1.0 -> trivially separable (poor synthetic data)

    Values below 0.5 are clipped back to 0.5, since a classifier performing worse
    than chance still implies the two sets are indistinguishable.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = np.vstack([X_real, X_syn])
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_syn))])
    n_splits = int(min(n_splits, np.bincount(y.astype(int)).min()))
    if n_splits < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], prob))
    return float(max(0.5, np.mean(aucs))) if aucs else 0.5


# =============================================================================
# The five  headline metrics for the SubSurGen benchmark
# =============================================================================

def mean_wasserstein(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """[1] Covariate fidelity. Mean 1-D Wasserstein distance across genes (lower better).

        W_j = W1( real_j , syn_j )          per-gene 1-D optimal transport cost
        score = (1/p) * sum_j W_j

    Marginal-level fidelity: does each gene's expression distribution match?
    Insensitive to gene-gene dependence, which metric [2] covers.
    """
    from scipy.stats import wasserstein_distance
    p = X_real.shape[1]
    return float(np.mean([wasserstein_distance(X_real[:, j], X_syn[:, j])
                          for j in range(p)]))


def gene_correlation_mae(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """[2] Molecular dependency. MAE between gene-gene correlation matrices (lower better).

        score = mean_{i<j} | C_real[i,j] - C_syn[i,j] |

    Only the strict upper triangle is used (the matrix is symmetric with unit
    diagonal). Exposes generators that reproduce marginals while destroying
    co-expression structure.
    """
    return correlation_error(X_real, X_syn)


def _centroids(X: np.ndarray, labels: np.ndarray, center: bool = False):
    """Mean expression profile per subtype, ordered by label value."""
    ks = np.unique(labels)
    C = np.vstack([X[labels == k].mean(axis=0) for k in ks])
    if center:                       # remove the shared global profile
        C = C - X.mean(axis=0, keepdims=True)
    return C, ks


def match_subtypes(X_real, s_real, X_syn, s_syn, center: bool = True):
    """Optimally pair real and synthetic subtypes by centroid correlation.

    Matching is done on MOLECULAR profiles only (never on survival), so the
    subtype-survival metric that follows is not circular.

    Uses the Hungarian algorithm on the negated correlation matrix. Returns
    (pairs, corr_matrix, ks_real, ks_syn) where pairs is a list of
    (real_label, syn_label) tuples.
    """
    from scipy.optimize import linear_sum_assignment

    Cr, kr = _centroids(X_real, s_real, center=center)
    Cs, ks = _centroids(X_syn, s_syn, center=center)

    n_r, n_s = len(Cr), len(Cs)
    corr = np.zeros((n_r, n_s))
    for i in range(n_r):
        for j in range(n_s):
            a, b = Cr[i], Cs[j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(a, b)[0, 1])
    corr = np.nan_to_num(corr)

    rows, cols = linear_sum_assignment(-corr)
    pairs = [(int(kr[i]), int(ks[j])) for i, j in zip(rows, cols)]
    return pairs, corr, kr, ks


def subtype_centroid_correlation(X_real, s_real, X_syn, s_syn,
                                 center: bool = True) -> dict:
    """[3] Subtype fidelity. Correlation of matched subtype centroids (higher better).

        1. compute the mean expression profile of every real and synthetic subtype
        2. match them optimally (Hungarian on centroid correlation)
        3. report the mean Pearson r over matched pairs

    `center=True` subtracts the cohort-mean profile before correlating. This is the
    reported value: without centering, every centroid shares the dominant global
    expression profile and correlations are near 1 for any generator, which makes
    the metric uninformative. The uncentered value is returned as well for
    transparency.
    """
    out = {}
    for tag, ctr in (("centered", True), ("raw", False)):
        pairs, corr, _, _ = match_subtypes(X_real, s_real, X_syn, s_syn, center=ctr)
        vals = [corr[list(np.unique(s_real)).index(a),
                     list(np.unique(s_syn)).index(b)] for a, b in pairs]
        out[tag] = float(np.mean(vals)) if vals else 0.0
    pairs, _, _, _ = match_subtypes(X_real, s_real, X_syn, s_syn, center=center)
    out["value"] = out["centered"] if center else out["raw"]
    out["pairs"] = pairs
    return out


def subtype_km_divergence(t_real, e_real, s_real, t_syn, e_syn, s_syn,
                          pairs, n_grid: int = 100, min_n: int = 5) -> dict:
    """[4] Subtype-survival fidelity. Mean KM divergence over matched subtypes (lower better).

        for each matched pair (a, b):
            D_ab = mean_t | S_real,a(t) - S_syn,b(t) |
        score = mean over pairs

    This is the metric that most directly tests SubSurGen's central claim: it is
    not enough for the pooled survival curve to match, each SUBTYPE must carry the
    right prognosis. A generator can match the overall KM curve perfectly while
    assigning survival to the wrong subtypes; only this metric detects that.

    Pairs come from molecular centroid matching, so survival plays no part in the
    alignment. Subtypes with fewer than `min_n` patients on either side are skipped.
    """
    hi = max(float(np.max(t_real)), float(np.max(t_syn)))
    grid = np.linspace(0, hi, n_grid)

    per_pair, used = [], []
    for a, b in pairs:
        ia, ib = (s_real == a), (s_syn == b)
        if ia.sum() < min_n or ib.sum() < min_n:
            continue
        d = float(np.mean(np.abs(kaplan_meier(t_real[ia], e_real[ia], grid)
                                 - kaplan_meier(t_syn[ib], e_syn[ib], grid))))
        per_pair.append(d)
        used.append((int(a), int(b), int(ia.sum()), int(ib.sum()), d))

    return {
        "value": float(np.mean(per_pair)) if per_pair else float("nan"),
        "n_pairs_used": len(per_pair),
        "per_pair": used,
    }


def within_subtype_dispersion(X_real, s_real, X_syn, s_syn) -> dict:
    """Ratio of synthetic to real within-subtype spread (1.0 = matched).

        spread(X, s) = mean over subtypes of the mean per-gene SD within a subtype
        ratio        = spread(synthetic) / spread(real)

    Diagnostic for interpreting subtype-centroid correlation, which rewards tight,
    well-separated clusters and therefore cannot by itself distinguish

        ratio << 1  collapse onto subtype centroids -- high centroid r, but the
                    synthetic cohort has lost the within-subtype heterogeneity
                    that real patients have
        ratio ~ 1   dispersion matched to the real cohort
        ratio >> 1  over-dispersed -- extra within-subtype spread blurs the
                    centroids and depresses centroid r even when the subtype
                    structure itself is correct

    Report alongside metric [3]; a centroid-correlation win at ratio far from 1
    should not be read as better subtype preservation.
    """
    def spread(X, s):
        vals = [X[s == k].std(axis=0).mean()
                for k in np.unique(s) if (s == k).sum() > 1]
        return float(np.mean(vals)) if vals else float("nan")

    r = spread(X_real, s_real)
    y = spread(X_syn, s_syn)
    return {"real": r, "synthetic": y,
            "ratio": float(y / r) if r and np.isfinite(r) and r > 0 else float("nan")}
