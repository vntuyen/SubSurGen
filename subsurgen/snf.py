"""Similarity Network Fusion (Wang et al., 2014) implemented in NumPy.

The uploaded cohorts contain a single omics type (gene expression), so a plain
SNF run would have nothing to fuse. We therefore build multiple *views* from the
single expression matrix by partitioning genes into co-expression modules, and
fuse the resulting patient networks. This keeps the SNF machinery (local scaling,
KNN sparsification, iterative diffusion) meaningful for single-omics input.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def euclidean_dist(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(X, metric="euclidean"))


def affinity_matrix(D: np.ndarray, K: int = 20, mu: float = 0.5) -> np.ndarray:
    """Scaled exponential similarity kernel with local scaling (SNF eq. 1)."""
    n = D.shape[0]
    K = min(K, n - 1)
    D = D.copy()
    np.fill_diagonal(D, 0.0)

    # mean distance to the K nearest neighbours, per patient
    order = np.argsort(D, axis=1)
    knn_idx = order[:, 1:K + 1]
    mean_knn = np.mean(np.take_along_axis(D, knn_idx, axis=1), axis=1)

    eps = (mean_knn[:, None] + mean_knn[None, :] + D) / 3.0 + 1e-10
    W = np.exp(-(D ** 2) / (mu * eps + 1e-10))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 1.0)
    return W


def _full_kernel(W: np.ndarray) -> np.ndarray:
    """Row-normalised full matrix P with P(i,i) = 1/2 (SNF eq. 2)."""
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    row = W.sum(axis=1, keepdims=True) + 1e-10
    P = W / (2.0 * row)
    np.fill_diagonal(P, 0.5)
    return P


def _knn_kernel(W: np.ndarray, K: int = 20) -> np.ndarray:
    """Sparse local kernel S keeping only K nearest neighbours (SNF eq. 3)."""
    n = W.shape[0]
    K = min(K, n - 1)
    S = np.zeros_like(W)
    W0 = W.copy()
    np.fill_diagonal(W0, -np.inf)
    idx = np.argsort(-W0, axis=1)[:, :K]
    for i in range(n):
        vals = W[i, idx[i]]
        s = vals.sum() + 1e-10
        S[i, idx[i]] = vals / s
    return S


def snf_fuse(Ws: list[np.ndarray], K: int = 20, t: int = 20) -> np.ndarray:
    """Iterative cross-diffusion across views (SNF eq. 4)."""
    m = len(Ws)
    if m == 1:
        return _full_kernel(Ws[0])

    Ps = [_full_kernel(W) for W in Ws]
    Ss = [_knn_kernel(W, K=K) for W in Ws]

    for _ in range(t):
        new_Ps = []
        for v in range(m):
            others = np.mean([Ps[u] for u in range(m) if u != v], axis=0)
            Pv = Ss[v] @ others @ Ss[v].T
            Pv = _full_kernel(Pv)
            new_Ps.append(Pv)
        Ps = new_Ps

    P = np.mean(Ps, axis=0)
    P = (P + P.T) / 2.0
    return P


def build_views(X: np.ndarray, n_views: int = 3, seed: int = 0) -> list[np.ndarray]:
    """Partition genes into co-expression modules -> one view per module."""
    p = X.shape[1]
    n_views = max(1, min(n_views, p // 5))
    if n_views == 1:
        return [X]

    # cluster genes by their correlation profile across patients
    Xg = X.T  # (p, n)
    Xg = (Xg - Xg.mean(axis=1, keepdims=True)) / (Xg.std(axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=n_views, n_init=4, random_state=seed)
    labels = km.fit_predict(Xg)

    views = []
    for k in range(n_views):
        cols = np.where(labels == k)[0]
        if len(cols) >= 3:
            views.append(X[:, cols])
    if not views:
        views = [X]
    return views


def spectral_embedding(P: np.ndarray, dim: int = 8) -> np.ndarray:
    """Normalised-Laplacian spectral embedding of the fused network."""
    n = P.shape[0]
    W = (P + P.T) / 2.0
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1) + 1e-10
    Dm = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - Dm @ W @ Dm
    vals, vecs = eigh(L)
    dim = min(dim, n - 1)
    emb = vecs[:, 1:dim + 1]  # skip the trivial constant eigenvector
    # row-normalise (Ng-Jordan-Weiss style)
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    return emb / norm


def spectral_cluster(P: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    emb = spectral_embedding(P, dim=max(k, 4))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(emb)


def run_snf(X: np.ndarray, k_subtypes: int = 3, K: int = 20, t: int = 20,
            n_views: int = 3, emb_dim: int = 8, seed: int = 0):
    """Full SNF pass on one dataset: fused network, subtypes, spectral embedding."""
    views = build_views(X, n_views=n_views, seed=seed)
    Ws = [affinity_matrix(euclidean_dist(V), K=K) for V in views]
    P = snf_fuse(Ws, K=K, t=t)
    emb = spectral_embedding(P, dim=emb_dim)
    labels = spectral_cluster(P, k=k_subtypes, seed=seed)
    return {"P": P, "embedding": emb, "labels": labels, "n_views": len(views)}


# ------------------------------------------------- choosing the subtype count --
def select_k(P: np.ndarray, k_range=(2, 7), n_boot: int = 20,
             subsample: float = 0.8, seed: int = 0) -> dict:
    """Choose the number of subtypes K from the fused network alone.

    Three unsupervised criteria are computed; none uses survival, so the choice
    of K cannot inflate the subtype-survival metric.

    eigengap
        Eigenvalues of the normalised Laplacian, ascending. A well-separated
        K-cluster graph has K eigenvalues near zero followed by a jump, so the
        index of the largest gap lambda_{K+1} - lambda_K suggests K.

    silhouette
        Mean silhouette width of the spectral embedding at each K. Higher means
        patients sit closer to their own subtype than to the nearest other one.

    stability
        Cluster `n_boot` random 80% subsamples and measure the adjusted Rand
        index against the labels those same patients received in the full
        clustering. High stability means K is not an artefact of who was sampled.

    Returns per-K scores plus a `recommended` K.

    Selection uses STABILITY as the primary criterion, with the eigengap as an
    independent check. Silhouette is reported but deliberately not used to pick K:
    on spectral embeddings it tends to increase monotonically with K (more, tighter
    clusters always look better by that measure), so it cannot distinguish a real
    cluster count from over-partitioning. Stability does not have that pathology --
    splitting genuine structure into more pieces makes subsample agreement worse.
    """
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rng = np.random.default_rng(seed)
    n = P.shape[0]
    ks = list(range(k_range[0], k_range[1] + 1))

    W = (P + P.T) / 2.0
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1) + 1e-10
    Dm = np.diag(1.0 / np.sqrt(d))
    vals = np.sort(eigh(np.eye(n) - Dm @ W @ Dm, eigvals_only=True))
    gaps = np.diff(vals[:max(ks) + 2])
    eigengap_k = int(np.argmax(gaps[1:len(ks) + 1]) + k_range[0])

    out = {"k_values": ks, "eigengap_k": eigengap_k,
           "eigenvalues": vals[:max(ks) + 2].tolist(),
           "silhouette": {}, "stability": {}}

    emb_full = spectral_embedding(P, dim=max(max(ks), 4))
    for k in ks:
        labels = spectral_cluster(P, k=k, seed=seed)
        try:
            out["silhouette"][k] = float(silhouette_score(emb_full, labels))
        except Exception:                                    # noqa: BLE001
            out["silhouette"][k] = float("nan")

        aris = []
        for _ in range(n_boot):
            idx = np.sort(rng.choice(n, size=int(subsample * n), replace=False))
            sub = spectral_cluster(P[np.ix_(idx, idx)], k=k, seed=seed)
            aris.append(adjusted_rand_score(labels[idx], sub))
        out["stability"][k] = float(np.mean(aris))

    stab = np.array([out["stability"][k] for k in ks], float)
    out["recommended"] = int(ks[int(np.nanargmax(stab))])
    out["agrees_with_eigengap"] = bool(out["recommended"] == eigengap_k)
    out["silhouette_is_monotone"] = bool(
        np.all(np.diff([out["silhouette"][k] for k in ks]) > -1e-9))
    return out
