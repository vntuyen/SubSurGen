"""SubSurGen: subtype- and survival-preserving synthetic data generation.

Public API
----------
    from subsurgen import build_episode, train_episodic, generate, run_snf
"""
from __future__ import annotations

__version__ = "1.0.0"

from .data import Cohort, build_gene_vocab, load_all, load_cohort
from .features import fs_by_cox, fs_by_variance
from .model import Diffusion, SubSurGenDenoiser, cox_ph_loss
from .snf import run_snf, spectral_cluster, spectral_embedding
from .trainer import Episode, build_episode, generate, train_episodic

__all__ = [
    "Cohort", "load_cohort", "load_all", "build_gene_vocab",
    "fs_by_cox", "fs_by_variance",
    "run_snf", "spectral_embedding", "spectral_cluster",
    "SubSurGenDenoiser", "Diffusion", "cox_ph_loss",
    "Episode", "build_episode", "train_episodic", "generate",
]
