"""SubSurGen generative model.

Design notes
------------
* Every gene is a TOKEN carrying a learned identity embedding keyed by gene
  symbol, so the model accepts datasets with different feature spaces and
  different numbers of features without retraining.
* Survival is generated JOINTLY with expression: `time` and `event` are two extra
  tokens denoised in the same reverse process (no separate covariate/survival
  stage, which is the limitation SurvDiff identifies in SurvivalGAN).
* Conditioning = SNF spectral position + subtype one-hot + dataset-level survival
  summary. The spectral position is what ties generation to the patient-similarity
  graph rather than to a discrete label alone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOK_GENE, TOK_TIME, TOK_EVENT = 0, 1, 2


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class SubSurGenDenoiser(nn.Module):
    def __init__(self, n_genes_vocab: int, d_model: int = 64, n_layers: int = 3,
                 n_heads: int = 4, cond_dim: int = 16, max_subtypes: int = 6):
        super().__init__()
        self.d_model = d_model
        self.gene_emb = nn.Embedding(n_genes_vocab, d_model)
        self.type_emb = nn.Embedding(3, d_model)
        self.value_proj = nn.Linear(1, d_model)

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # cond = spectral embedding + subtype one-hot + [censor rate, log horizon]
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + max_subtypes + 2, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        # auxiliary Cox risk head, reads the pooled token representation
        self.risk_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.SiLU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_noisy, gene_ids, tok_types, t, cond, return_risk=False):
        """x_noisy (B,L) ; gene_ids (L,) ; tok_types (L,) ; t (B,) ; cond (B,C)."""
        B, L = x_noisy.shape
        h = self.value_proj(x_noisy.unsqueeze(-1))                      # B,L,D
        h = h + self.gene_emb(gene_ids)[None] + self.type_emb(tok_types)[None]
        ctx = self.t_mlp(timestep_embedding(t, self.d_model)) + self.cond_mlp(cond)
        h = h + ctx[:, None, :]
        h = self.encoder(h)
        eps = self.out(h).squeeze(-1)                                   # B,L
        if return_risk:
            gene_mask = (tok_types == TOK_GENE).float()[None, :, None]
            pooled = (h * gene_mask).sum(1) / gene_mask.sum()
            return eps, self.risk_head(pooled).squeeze(-1)
        return eps


class Diffusion:
    """Standard DDPM with a cosine schedule."""

    def __init__(self, n_steps: int = 200, device: str = "cpu"):
        self.T = n_steps
        s = 0.008
        steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
        alphas_cum = torch.cos(((steps + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cum = alphas_cum / alphas_cum[0]
        betas = torch.clip(1 - alphas_cum[1:] / alphas_cum[:-1], 0, 0.999)
        self.betas = betas.float().to(device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cum = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a = self.alphas_cum[t][:, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, shape, gene_ids, tok_types, cond, device="cpu",
               guidance: float = 1.0):
        """Reverse diffusion. `guidance` > 1 applies classifier-free guidance,
        sharpening how strongly samples adhere to their SNF conditioning."""
        x = torch.randn(shape, device=device)
        null_cond = torch.zeros_like(cond)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, gene_ids, tok_types, t, cond)
            if guidance != 1.0:
                eps_u = model(x, gene_ids, tok_types, t, null_cond)
                eps = eps_u + guidance * (eps - eps_u)
            a_t = self.alphas[i]
            ac_t = self.alphas_cum[i]
            mean = (x - (1 - a_t) / (1 - ac_t).sqrt() * eps) / a_t.sqrt()
            if i > 0:
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
    """Negative Cox partial log-likelihood (Breslow)."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    ll = (risk - log_cum) * event
    denom = event.sum().clamp(min=1.0)
    return -ll.sum() / denom
