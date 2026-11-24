"""Step 0 -- Data preparation.

For EACH cohort independently:
  1. load expression + survival
  2. univariate Cox feature selection
  3. SNF -> fused similarity network, subtypes, spectral embedding
  4. package as an Episode and cache to results/episodes.pkl

Usage:
    python scripts/step0_prepare_data.py [--quick]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from common import (banner, load_episodes, parse_common_args, save_episodes,
                    write_json)
import config as C
from subsurgen.data import build_gene_vocab, load_all
from subsurgen.features import fs_by_cox
from subsurgen.metrics import logrank_test
from subsurgen.snf import run_snf, select_k
from subsurgen.trainer import build_episode

warnings.filterwarnings("ignore")


def main() -> dict:
    p = argparse.ArgumentParser(description="Prepare cohorts and build episodes")
    parse_common_args(p)

    banner("STEP 0 -- DATA PREPARATION")
    cohorts = load_all(C.DATA_DIR)
    if not cohorts:
        raise SystemExit(
            f"No *_driver_genes.csv files found in {C.DATA_DIR}.\n"
            f"Put your cohort CSVs there, or set SURVGEN_DATA=/path/to/data"
        )

    vocab = build_gene_vocab(cohorts)
    print(f"Loaded {len(cohorts)} cohorts | gene vocabulary: {len(vocab)} symbols\n")

    mode = C.FEATURE_SELECTION
    if mode == "none":
        print("feature selection: NONE -- full panel used for both subtype\n"
              "  discovery and generation.\n")
    elif mode == "subtypes_only":
        print(f"feature selection: SUBTYPES ONLY (Cox p<{C.COX_CUTOFF}, "
              f"min {C.MIN_KEEP} genes)\n"
              "  Subtypes are defined on survival-informed genes, but the\n"
              "  generator emits the FULL input panel, so synthetic cohorts keep\n"
              "  exactly the same genes as the real ones.\n")
    else:
        print(f"feature selection: FULL (Cox p<{C.COX_CUTOFF}, "
              f"min {C.MIN_KEEP} genes)\n"
              "  WARNING: synthetic cohorts will have FEWER genes than the real\n"
              "  input, so they are not a drop-in replacement.\n")

    auto_k = isinstance(C.K_SUBTYPES, str) and C.K_SUBTYPES.lower() == "auto"
    if auto_k:
        print(f"subtype count: AUTO (stability over K in {C.K_RANGE}, "
              f"eigengap cross-check)\n")
    else:
        print(f"subtype count: fixed K={C.K_SUBTYPES}\n")

    episodes, summary = {}, []
    for c in cohorts:
        n_snf_genes = c.p
        if mode == "none":
            snf_X, gen_cohort = c.X, c
        else:
            keep, _ = fs_by_cox(c.X, c.time, c.event,
                                cutoff=C.COX_CUTOFF, min_keep=C.MIN_KEEP)
            n_snf_genes = len(keep)
            snf_X = c.X[:, keep]
            # "subtypes_only" generates every gene; "full" also reduces the panel
            gen_cohort = c if mode == "subtypes_only" else c.subset_genes(keep)
            if mode == "full":
                snf_X = None            # already reduced, use the cohort itself

        # choose K from the fused network, without touching survival
        k_diag = None
        k_use = C.K_SUBTYPES
        probe = snf_X if snf_X is not None else gen_cohort.X
        P_probe = run_snf(probe, k_subtypes=2, K=C.SNF_K, t=C.SNF_T,
                          n_views=C.SNF_VIEWS, emb_dim=C.COND_DIM,
                          seed=C.SEED)["P"]
        k_diag = select_k(P_probe, k_range=C.K_RANGE,
                          n_boot=C.K_BOOTSTRAPS, seed=C.SEED)
        if auto_k:
            k_use = k_diag["recommended"]

        ep = build_episode(gen_cohort, vocab, k_subtypes=k_use,
                           cox_cutoff=1.01, seed=C.SEED, snf_X=snf_X)
        episodes[c.name] = ep

        _, p_real, _ = logrank_test(ep.cohort.time, ep.cohort.event, ep.labels)
        sizes = np.bincount(ep.labels, minlength=k_use).tolist()
        summary.append({
            "cohort": c.name,
            "n_patients": int(c.n),
            "n_genes_all": int(c.p),
            "n_genes_generated": int(ep.cohort.p),
            "n_genes_for_subtypes": int(n_snf_genes),
            "feature_selection": mode,
            "k_subtypes": int(k_use),
            "k_selected_by_stability": int(k_diag["recommended"]),
            "k_eigengap": int(k_diag["eigengap_k"]),
            "k_stability": {str(k): round(v, 4)
                            for k, v in k_diag["stability"].items()},
            "k_silhouette": {str(k): round(v, 4)
                             for k, v in k_diag["silhouette"].items()},
            "selected_genes": list(ep.cohort.genes),
            "event_rate": round(float(c.event_rate), 4),
            "time_min": round(float(c.time.min()), 3),
            "time_max": round(float(c.time.max()), 3),
            "subtype_sizes": sizes,
            "logrank_p_real": float(p_real),
            # Metric [3] asks whether subtype-specific prognosis is preserved.
            # Where the real subtypes carry no survival signal there is nothing
            # to preserve, so those cohorts are reported as a separate subgroup.
            "subtypes_prognostic": bool(p_real < C.PROGNOSTIC_ALPHA),
        })
        k_note = (f"K={k_use}" if not auto_k
                  else f"K={k_use}(auto)")
        agree = "" if k_diag["recommended"] == k_diag["eigengap_k"] else \
                f" [eigengap suggests {k_diag['eigengap_k']}]"
        print(f"  {c.name:10s} n={c.n:4d}  generate {ep.cohort.p:3d} genes, "
              f"subtypes from {n_snf_genes:3d}  {k_note}  "
              f"events={c.event_rate:.2f}  sizes={sizes}  "
              f"log-rank p={p_real:.4g}{agree}")

    n_prog = sum(1 for r in summary if r["subtypes_prognostic"])
    if n_prog < len(summary):
        weak = [r["cohort"] for r in summary if not r["subtypes_prognostic"]]
        print(f"\n  {n_prog}/{len(summary)} cohorts have prognostic subtypes "
              f"(log-rank p < {C.PROGNOSTIC_ALPHA}).")
        print(f"  Without a survival difference between real subtypes there is")
        print(f"  nothing for metric [3] to preserve, so these are reported as a")
        print(f"  separate subgroup: {', '.join(weak)}")

    save_episodes(episodes, vocab)
    write_json("step0_data_summary.json", summary)
    print(f"\nEpisodes cached to {C.EPISODES_PKL}")
    return {"summary": summary}


if __name__ == "__main__":
    main()
