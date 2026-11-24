"""Episode construction and episodic training for SubSurGen.

Each cohort is an independent EPISODE: its own feature space, its own SNF graph,
its own survival distribution. The generator is trained across episodes so it
learns a general mechanism for small cohorts rather than memorising one dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Cohort
from .features import fs_by_cox
from .model import TOK_GENE, TOK_TIME, TOK_EVENT, cox_ph_loss
from .snf import run_snf

MAX_SUBTYPES = 6
COND_DIM = 16


@dataclass
class Episode:
    name: str
    cohort: Cohort              # after feature selection
    gene_ids: np.ndarray        # (p,) global vocabulary ids
    emb: np.ndarray             # (n, d) SNF spectral embedding
    labels: np.ndarray          # (n,) SNF subtypes
    P: np.ndarray               # fused similarity network
    x_norm: np.ndarray          # (n, p) expression scaled to [-1, 1]
    t_norm: np.ndarray          # (n,) normalised log-time
    e_norm: np.ndarray          # (n,) event as +/-1
    t_min: float
    t_max: float
    censor_rate: float

    @property
    def n_tokens(self) -> int:
        return self.cohort.p + 2

    def token_spec(self):
        gene_ids = np.concatenate([self.gene_ids, [0, 0]])
        types = np.array([TOK_GENE] * self.cohort.p + [TOK_TIME, TOK_EVENT])
        return gene_ids, types

    def targets(self) -> np.ndarray:
        return np.concatenate(
            [self.x_norm, self.t_norm[:, None], self.e_norm[:, None]], axis=1
        )

    def conditioning(self, emb=None, labels=None) -> np.ndarray:
        """cond = [spectral position | subtype one-hot | censor rate, log horizon]."""
        emb = self.emb if emb is None else emb
        labels = self.labels if labels is None else labels
        n = emb.shape[0]

        c_emb = np.zeros((n, COND_DIM))
        d = min(COND_DIM, emb.shape[1])
        c_emb[:, :d] = emb[:, :d]

        onehot = np.zeros((n, MAX_SUBTYPES))
        onehot[np.arange(n), np.clip(labels, 0, MAX_SUBTYPES - 1)] = 1.0

        meta = np.tile([[self.censor_rate, np.log1p(self.t_max)]], (n, 1))
        return np.concatenate([c_emb, onehot, meta], axis=1)

    def denormalise(self, gen: np.ndarray):
        """Map generated tokens back to expression / time / event."""
        p = self.cohort.p
        x = np.clip((gen[:, :p] + 1) / 2, 0, 1)
        tn = np.clip(gen[:, p], -1, 1)
        log_t = (tn + 1) / 2 * (np.log1p(self.t_max) - np.log1p(self.t_min)) \
            + np.log1p(self.t_min)
        time = np.expm1(np.clip(log_t, 0, None))
        event = (gen[:, p + 1] > 0).astype(float)
        return x, time, event


def build_episode(cohort: Cohort, gene_vocab: dict, k_subtypes: int = 3,
                  cox_cutoff: float = 0.05, seed: int = 0,
                  snf_X=None) -> Episode:
    """Build one training episode.

    `snf_X` optionally supplies a DIFFERENT matrix for subtype discovery than the
    one that is generated. This separates two decisions that are usually
    conflated: which genes define the subtypes, and which genes the model has to
    reproduce. Passing a survival-informed subset as `snf_X` yields prognostic
    subtypes while the generator still emits the complete input panel, so the
    synthetic cohort keeps exactly the genes the real one has.
    """
    if cox_cutoff is not None and cox_cutoff <= 1.0:
        keep, _ = fs_by_cox(cohort.X, cohort.time, cohort.event, cutoff=cox_cutoff)
        c = cohort.subset_genes(keep)
    else:
        c = cohort

    snf_input = c.X if snf_X is None else snf_X
    snf = run_snf(snf_input, k_subtypes=k_subtypes, emb_dim=COND_DIM, seed=seed)

    x_norm = c.X * 2 - 1
    t_min, t_max = float(c.time.min()), float(c.time.max())
    log_t = np.log1p(c.time)
    lo, hi = np.log1p(t_min), np.log1p(t_max)
    t_norm = (log_t - lo) / (hi - lo + 1e-10) * 2 - 1
    e_norm = c.event * 2 - 1

    gene_ids = np.array([gene_vocab[g] for g in c.genes])

    return Episode(
        name=c.name, cohort=c, gene_ids=gene_ids, emb=snf["embedding"],
        labels=snf["labels"], P=snf["P"], x_norm=x_norm, t_norm=t_norm,
        e_norm=e_norm, t_min=t_min, t_max=t_max,
        censor_rate=float(1 - c.event.mean()),
    )


def train_episodic(model, diffusion, episodes: list[Episode], steps: int = 3000,
                   batch_size: int = 64, lr: float = 2e-3, lambda_cox: float = 0.2,
                   feature_dropout: float = 0.3, cond_dropout: float = 0.15,
                   seed: int = 0, log_every: int = 500,
                   device: str = "cpu", verbose: bool = True):
    """Train across episodes. Feature dropout randomly hides genes each step so the
    model must cope with heterogeneous feature spaces at inference time."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []

    for step in range(steps):
        ep = episodes[rng.integers(len(episodes))]
        p = ep.cohort.p

        # --- random feature subset: simulates a different feature space ---
        if feature_dropout > 0 and p > 6:
            n_keep = max(5, int(p * (1 - rng.uniform(0, feature_dropout))))
            cols = np.sort(rng.choice(p, size=n_keep, replace=False))
        else:
            cols = np.arange(p)

        idx = rng.choice(ep.cohort.n, size=min(batch_size, ep.cohort.n), replace=False)

        tgt_full = ep.targets()
        tgt = np.concatenate([tgt_full[idx][:, cols],
                              tgt_full[idx][:, p:p + 2]], axis=1)
        cond = ep.conditioning()[idx]

        gene_ids = torch.tensor(
            np.concatenate([ep.gene_ids[cols], [0, 0]]), dtype=torch.long, device=device
        )
        types = torch.tensor(
            [TOK_GENE] * len(cols) + [TOK_TIME, TOK_EVENT],
            dtype=torch.long, device=device
        )
        x0 = torch.tensor(tgt, dtype=torch.float32, device=device)
        c = torch.tensor(cond, dtype=torch.float32, device=device)
        if cond_dropout > 0:                 # unconditional passes for CFG
            drop = torch.rand(c.shape[0], device=device) < cond_dropout
            c = c * (~drop).float()[:, None]

        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        eps_pred, risk = model(x_t, gene_ids, types, t, c, return_risk=True)
        loss_diff = torch.nn.functional.mse_loss(eps_pred, noise)

        tt = torch.tensor(ep.cohort.time[idx], dtype=torch.float32, device=device)
        ee = torch.tensor(ep.cohort.event[idx], dtype=torch.float32, device=device)
        loss_cox = cox_ph_loss(risk, tt, ee) if ee.sum() > 1 else torch.tensor(0.0)

        loss = loss_diff + lambda_cox * loss_cox

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % log_every == 0:
            history.append((step + 1, float(loss_diff.detach()), float(loss_cox.detach())))
            if verbose:
                print(f"  step {step+1:5d} | diff {float(loss_diff.detach()):.4f} "
                      f"| cox {float(loss_cox.detach()):.4f}")
    return history


@torch.no_grad()
def generate(model, diffusion, ep: Episode, n_samples: int, seed: int = 0,
             interpolate: bool = True, guidance: float = 2.0, device: str = "cpu"):
    """Sample synthetic patients by drawing positions on the cohort's SNF manifold.

    `interpolate=True` mixes pairs of within-subtype neighbours, producing NEW
    positions on the manifold rather than replaying real patients' positions.
    """
    rng = np.random.default_rng(seed)
    # The reverse diffusion draws its initial noise from torch's GLOBAL RNG, so
    # without seeding it here the output depends on whatever consumed that RNG
    # earlier in the process. That made generation irreproducible and caused the
    # same configuration to score differently in step 2 (fresh process) and
    # step 4 (immediately after training). Seed it explicitly and restore the
    # previous state afterwards so callers are unaffected.
    _torch_state = torch.get_rng_state()
    torch.manual_seed(seed)
    model.eval()

    emb, labels = ep.emb, ep.labels
    picks = rng.integers(0, len(emb), size=n_samples)
    new_emb = emb[picks].copy()
    new_lab = labels[picks].copy()

    if interpolate:
        for i, pi in enumerate(picks):
            same = np.where(labels == labels[pi])[0]
            if len(same) > 1:
                pj = rng.choice(same)
                w = rng.uniform(0.0, 0.5)
                new_emb[i] = (1 - w) * emb[pi] + w * emb[pj]

    cond = torch.tensor(ep.conditioning(new_emb, new_lab),
                        dtype=torch.float32, device=device)
    gene_ids_np, types_np = ep.token_spec()
    gene_ids = torch.tensor(gene_ids_np, dtype=torch.long, device=device)
    types = torch.tensor(types_np, dtype=torch.long, device=device)

    gen = diffusion.sample(model, (n_samples, ep.n_tokens), gene_ids, types,
                           cond, device=device, guidance=guidance).cpu().numpy()
    x, time, event = ep.denormalise(gen)
    model.train()
    torch.set_rng_state(_torch_state)
    return x, time, event, new_lab
