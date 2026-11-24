"""Cohort loading for SubSurGen.

Each cohort is kept as a SEPARATE dataset. A cohort carries its own
feature space (gene symbols), its own patients, and its own survival outcomes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cohort:
    name: str
    X: np.ndarray          # (n, p) expression, already in [0, 1]
    genes: list            # length p, gene symbols -> defines the feature space
    time: np.ndarray       # (n,) observed time
    event: np.ndarray      # (n,) 1 = event, 0 = censored

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def event_rate(self) -> float:
        return float(self.event.mean())

    def subset_genes(self, keep_idx: np.ndarray) -> "Cohort":
        """Return a view of this cohort restricted to a subset of genes.

        Used to simulate heterogeneous feature spaces across episodes.
        """
        return Cohort(
            name=self.name,
            X=self.X[:, keep_idx],
            genes=[self.genes[i] for i in keep_idx],
            time=self.time,
            event=self.event,
        )

    def subset_patients(self, idx: np.ndarray) -> "Cohort":
        return Cohort(
            name=self.name,
            X=self.X[idx],
            genes=list(self.genes),
            time=self.time[idx],
            event=self.event[idx],
        )


def load_cohort(path: str, name: str | None = None) -> Cohort:
    df = pd.read_csv(path)
    if name is None:
        name = os.path.basename(path).replace("_driver_genes.csv", "")
    genes = [c for c in df.columns if c not in ("time", "event")]
    X = df[genes].to_numpy(dtype=np.float64)
    time = df["time"].to_numpy(dtype=np.float64)
    event = df["event"].to_numpy(dtype=np.float64)

    # Non-positive follow-up times are invalid for survival models and cause
    # hard failures in some baselines (synthcity rejects time <= 0 outright, which
    # silently drops a whole cohort from the comparison). Clip to a small positive
    # value and report, so every method is evaluated on the same cohorts.
    n_bad = int((time <= 0).sum())
    if n_bad:
        floor = max(1e-3, float(time[time > 0].min()) * 0.5) if (time > 0).any() else 1e-3
        time = np.where(time <= 0, floor, time)
        print(f"  [{name}] clipped {n_bad} non-positive survival time(s) to {floor:.4g}")

    # guard against degenerate columns (zero variance breaks the SNF kernel)
    keep = X.std(axis=0) > 1e-8
    if not keep.all():
        X = X[:, keep]
        genes = [g for g, k in zip(genes, keep) if k]

    return Cohort(name=name, X=X, genes=genes, time=time, event=event)


def load_all(upload_dir: str) -> list[Cohort]:
    files = sorted(
        f for f in os.listdir(upload_dir) if f.endswith("_driver_genes.csv")
    )
    return [load_cohort(os.path.join(upload_dir, f)) for f in files]


def build_gene_vocab(cohorts: list[Cohort]) -> dict:
    """Global gene symbol -> integer id. Lets the model accept any feature space."""
    vocab: dict = {}
    for c in cohorts:
        for g in c.genes:
            if g not in vocab:
                vocab[g] = len(vocab)
    return vocab
