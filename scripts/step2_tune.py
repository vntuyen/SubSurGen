"""Step 2 -- Automatic guidance tuning.

Sweeps the classifier-free guidance weight on the already-trained models, scores
every candidate on all five benchmark metrics plus the dispersion diagnostic, and
**writes the selected value to results/tuning.json**. Steps 3 and 5 read it from
there, so the operating point is chosen from data rather than set by hand.

No retraining happens here: the models are fixed and only sampling changes, so a
sweep costs minutes.

Selection ranks candidates by weighted mean rank across the five metrics, with an
optional penalty on |dispersion - 1|. The penalty matters because subtype-centroid
correlation can be inflated by over- or under-dispersing the synthetic cohort;
without it the tuner could pick a setting that wins metric [2] by distorting the
within-subtype spread.

Usage:
    python scripts/step2_tune.py                 # sweep and persist the choice
    python scripts/step2_tune.py --values 1 2 3  # custom candidates
    python scripts/step2_tune.py --show          # print the current selection
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from common import (banner, describe_device, get_device, load_episodes,
                    load_tuning, load_trained_model, model_for,
                    parse_common_args, save_tuning, write_json)
import config as C
from subsurgen.trainer import generate

warnings.filterwarnings("ignore")

LOWER = ["wasserstein", "subtype_km_div", "km_div"]
HIGHER = ["centroid_corr", "tstr_cindex"]


def main() -> dict:
    p = argparse.ArgumentParser(description="Auto-tune the guidance weight")
    p.add_argument("--values", type=float, nargs="+", default=None)
    p.add_argument("--show", action="store_true",
                   help="print the persisted selection and exit")
    args = parse_common_args(p)

    if args.show:
        t = load_tuning()
        banner("STEP 2 -- CURRENT TUNING")
        if not t:
            print("nothing tuned yet -- run scripts/step2_tune.py")
        else:
            for k, v in t.items():
                print(f"  {k}: {v}")
        return t

    from step5_ablation import score          # shared scorer, all five metrics

    banner("STEP 2 -- GUIDANCE TUNING")
    values = args.values if args.values else C.GUIDANCE_SWEEP
    models, diffusion, episodes, _ = load_trained_model()
    print(f"candidates: {values}")
    print(f"cohorts:    {len(episodes)}")
    print(f"device:     {describe_device()}\n")

    sweep = []
    for g in values:
        per = []
        for name, ep in episodes.items():
            Xs, ts, es, _ = generate(model_for(models, name), diffusion, ep,
                                     ep.cohort.n, seed=C.SEED, guidance=g,
                                     device=get_device())
            per.append(score(ep, Xs, ts, es))
        row = {"guidance": float(g)}
        row.update({k: float(np.nanmean([r[k] for r in per])) for k in per[0]})
        sweep.append(row)
        print(f"  w={g:4.1f}  W1={row['wasserstein']:.4f}  "
              f"centroid_r={row['centroid_corr']:+.3f}  "
              f"subKM={row['subtype_km_div']:.3f}  KM={row['km_div']:.3f}  "
              f"TSTR={row['tstr_cindex']:.3f}  "
              f"disp={row['dispersion_ratio']:.2f}", flush=True)

    # ---- rank the candidates -------------------------------------------------
    names = [r["guidance"] for r in sweep]
    ranks = {g: [] for g in names}
    weights = []
    print("\n  best per metric:")
    for k in LOWER + HIGHER:
        vals = np.array([r[k] for r in sweep], float)
        sign = 1.0 if k in LOWER else -1.0
        keyed = np.where(np.isnan(vals), np.inf, sign * vals)
        order = np.argsort(keyed)
        for pos, i in enumerate(order, 1):
            ranks[names[i]].append(pos * C.TUNE_WEIGHTS.get(k, 1.0))
        weights.append(C.TUNE_WEIGHTS.get(k, 1.0))
        print(f"    {k:16s} w={names[int(order[0])]}")

    wsum = sum(weights) or 1.0
    scores = {}
    for g in names:
        row = [r for r in sweep if r["guidance"] == g][0]
        base = sum(ranks[g]) / wsum
        pen = C.DISPERSION_PENALTY * abs(row["dispersion_ratio"] - 1.0)
        scores[g] = base + pen

    best = min(scores, key=scores.get)
    print("\n  ranking (weighted mean rank + dispersion penalty, lower is better):")
    for g in sorted(scores, key=scores.get):
        row = [r for r in sweep if r["guidance"] == g][0]
        mark = "  <-- selected" if g == best else ""
        print(f"    w={g:4.1f}  score {scores[g]:.2f}  "
              f"(dispersion {row['dispersion_ratio']:.2f}){mark}")

    chosen = [r for r in sweep if r["guidance"] == best][0]
    tuning = {
        "guidance": float(best),
        "n_candidates": len(values),
        "candidates": [float(v) for v in values],
        "selected_metrics": {k: chosen[k] for k in LOWER + HIGHER
                             + ["dispersion_ratio"]},
        "dispersion_penalty": C.DISPERSION_PENALTY,
        "weights": C.TUNE_WEIGHTS,
    }
    save_tuning(tuning)
    write_json("step2_guidance_sweep.json", sweep)

    print(f"\n  SELECTED guidance = {best}")
    print("  written to results/tuning.json -- steps 3 and 5 will use it "
          "automatically.")
    if abs(chosen["dispersion_ratio"] - 1.0) > 0.25:
        print(f"  NOTE: dispersion {chosen['dispersion_ratio']:.2f} is far from "
              f"1.00, so subtype-centroid correlation at this setting reflects "
              f"altered within-subtype spread as well as subtype structure.")
    return tuning


if __name__ == "__main__":
    main()
