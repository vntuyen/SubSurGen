"""Step 4 -- The four-metric benchmark.

Scores every cached synthetic cohort on the four metrics that test SubSurGen's
core claims. Metrics [2] and [3] require re-running SNF on the synthetic data;
everything else is computed directly from the cached arrays.

    #  Dimension                  Metric                          Better
    1  Covariate fidelity         mean Wasserstein distance         down
    2  Subtype fidelity           subtype-centroid correlation      up
    3  Subtype-survival fidelity  subtype-specific KM divergence    down
    4  Downstream utility         TSTR C-index                      up


Usage:
    python scripts/step4_evaluate.py [--methods SubSurGen CTGAN] [--quick]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from common import (banner, list_synthetic_methods, list_synthetic_seeds,
                    load_episodes, load_synthetic, parse_common_args, read_json,
                    write_json)

# Which cohorts have prognostic real subtypes (set in step 0). Metric [3] is only
# interpretable where the real subtypes differ in survival.
PROGNOSTIC: dict = {}


def load_prognostic():
    for r in (read_json("step0_data_summary.json") or []):
        PROGNOSTIC[r["cohort"]] = bool(r.get("subtypes_prognostic", True))
import config as C
from subsurgen.baselines import BASELINES
from subsurgen.downstream import cox_cindex, fit_cox
from subsurgen.metrics import (km_divergence, mean_wasserstein,
                               subtype_centroid_correlation,
                               subtype_km_divergence, within_subtype_dispersion)
from subsurgen.snf import run_snf

warnings.filterwarnings("ignore")

FAMILY = {m: f for m, (_, f) in BASELINES.items()}
FAMILY["SubSurGen"] = "Ours"

# Provenance is read back from the generation log so the table reports what was
# ACTUALLY executed, never what merely happened to be installed on disk.
SOURCE: dict = {}

# The Four headline metrics that get a seed-to-seed error bar. step3_generate
# can sample the same method/cohort with several seeds; here that becomes,
# for each metric, both a mean and an sd (stored alongside it as "<metric>_sd").
METRIC_KEYS = ("wasserstein", "centroid_corr", "subtype_km_div", "km_div",
               "tstr_cindex")


def _agg_with_sd(rs: list) -> dict:
    """Mean each metric over `rs` (rows), plus the mean of their per-cohort
    seed-sds -- i.e. how much a metric wobbles from resampling generation,
    averaged over cohorts. The _sd value is NaN wherever fewer than 2 seeds
    were available for every cohort that went into `rs`."""
    out = {}
    for k in METRIC_KEYS:
        out[k] = float(np.nanmean([r[k] for r in rs]))
        out[f"{k}_sd"] = float(np.nanmean(
            [r.get(f"{k}_sd", float("nan")) for r in rs]))
    return out


def load_sources():
    """Where each method's synthetic data actually came from.

    The generation log records what step 3 ran. Methods later imported from an
    official repository override that, since those cohorts were produced by the
    published implementation rather than the in-repo fallback.
    """
    from common import read_json
    log = read_json("step3_generation_log.json") or []
    for r in log:
        SOURCE[r["method"]] = r.get("source", "?")
    for m, meta in (read_json("external_provenance.json") or {}).items():
        SOURCE[m] = meta.get("source", "repo")
    SOURCE.setdefault("SubSurGen", "ours")


def tstr_cindex(Xs, ts, es, ep, rng_seed_base=500) -> tuple:
    """Fit Cox on SYNTHETIC only, evaluate C-index on held-out REAL patients."""
    tstr, trtr = [], []
    for i in range(C.N_REPEATS):
        rng = np.random.default_rng(rng_seed_base + i)
        idx = rng.permutation(ep.cohort.n)
        n_tr = int((1 - C.TEST_FRACTION) * ep.cohort.n)
        tr, te = idx[:n_tr], idx[n_tr:]
        Xtr, ttr, etr = ep.cohort.X[tr], ep.cohort.time[tr], ep.cohort.event[tr]
        Xte, tte, ete = ep.cohort.X[te], ep.cohort.time[te], ep.cohort.event[te]
        if etr.sum() < 2 or ete.sum() < 2:
            continue
        trtr.append(cox_cindex(fit_cox(Xtr, ttr, etr, seed=i), Xte, tte, ete))

        sub = rng.choice(len(Xs), size=min(len(Xs), n_tr), replace=False)
        Xsy, tsy, esy = Xs[sub], ts[sub], es[sub]
        if esy.sum() < 2:
            continue
        tstr.append(cox_cindex(fit_cox(Xsy, tsy, esy, seed=i), Xte, tte, ete))
    return (float(np.mean(tstr)) if tstr else float("nan"),
            float(np.mean(trtr)) if trtr else float("nan"))


def evaluate(method, cohort, ep, Xs, ts, es) -> dict:
    Xr, tr, er, sr = ep.cohort.X, ep.cohort.time, ep.cohort.event, ep.labels

    # A generator that emits no events cannot support a Cox fit, so TSTR is
    # undefined. That is a genuine survival-generation failure, not a bug -- it
    # is recorded explicitly rather than silently returned as NaN.
    n_events = int(es.sum())
    degenerate = n_events < 2

    # [2]/[3] need subtypes discovered on the synthetic cohort, independently
    ss = run_snf(Xs, k_subtypes=C.K_SUBTYPES, K=C.SNF_K, t=C.SNF_T,
                 n_views=C.SNF_VIEWS, emb_dim=C.COND_DIM, seed=C.SEED)["labels"]

    cc = subtype_centroid_correlation(Xr, sr, Xs, ss, center=True)
    disp = within_subtype_dispersion(Xr, sr, Xs, ss)
    skm = subtype_km_divergence(tr, er, sr, ts, es, ss, cc["pairs"])
    if degenerate:
        tstr, trtr = float("nan"), float("nan")
    else:
        tstr, trtr = tstr_cindex(Xs, ts, es, ep)

    return {
        "method": method, "cohort": cohort,
        "family": FAMILY.get(method, "?"),
        "source": SOURCE.get(method, "?"),
        "is_official": SOURCE.get(method) in ("ours", "pip", "repo"),
        # the Four headline metrics
        "wasserstein": mean_wasserstein(Xr, Xs),                       # [1] down
        "centroid_corr": cc["value"],                                  # [2] up
        "subtype_km_div": skm["value"],                                # [3] down
        "km_div": km_divergence(tr, er, ts, es),                       # [4] down
        "tstr_cindex": tstr,                                           # [5] up
        # supporting detail
        "centroid_corr_raw": cc["raw"],
        "dispersion_ratio": disp["ratio"],
        "trtr_cindex": trtr,
        "tstr_ratio": tstr / trtr if trtr and np.isfinite(trtr) else float("nan"),
        "n_subtype_pairs": skm["n_pairs_used"],
        "event_rate_syn": float(es.mean()),
        "event_rate_real": float(er.mean()),
        "n_events_syn": n_events,
        "degenerate_events": bool(degenerate),
    }


def main() -> dict:
    p = argparse.ArgumentParser(description="Four-metric benchmark")
    p.add_argument("--methods", nargs="*", default=None)
    args = parse_common_args(p)

    banner("STEP 4 -- FOUR-METRIC BENCHMARK")
    load_sources()
    load_prognostic()
    episodes, _ = load_episodes()
    n_prog = sum(1 for c in episodes if PROGNOSTIC.get(c, True))
    if n_prog < len(episodes):
        print(f"  {n_prog}/{len(episodes)} cohorts have prognostic subtypes; "
              f"subKM is also reported for that subgroup as subKM*\n")

    # An undertrained SubSurGen is compared against fully-fitted baselines, which
    # is not a comparison at all. Measured on one fixed 26-gene cohort pair,
    # subtype-centroid correlation is 0.260 at 200 steps and 0.717 at 2500 --
    # the metric is dominated by training length, not by method quality.
    hist = read_json("step1_train_history.json") or {}
    trained_steps = hist.get("steps")
    undertrained = trained_steps is not None and trained_steps < 1000
    if undertrained:
        print("!" * 74)
        print("Rerun:  python run_all.py --train-steps 2500")
        print("!" * 74 + "\n")
    methods = args.methods or list_synthetic_methods()
    if not methods:
        raise SystemExit("no cached synthetic data -- run step 2 first")

    rows = []
    for method in methods:
        per, seed_spread = [], []
        for cohort, ep in episodes.items():
            seeds = list_synthetic_seeds(method, cohort)
            if not seeds:
                continue
            reps = []
            for sd in seeds:
                syn = load_synthetic(method, cohort, seed=sd)
                if syn is None:
                    continue
                Xs, ts, es = syn
                if len(Xs) < 10:
                    continue
                reps.append(evaluate(method, cohort, ep, Xs, ts, es))
            if not reps:
                continue
            # average the seeds within a cohort; keep each headline metric's
            # seed-to-seed sd too (NaN with a single seed), so error bars
            # survive up to the aggregate table and figures instead of being
            # collapsed away here.
            merged = dict(reps[0])
            for k, v in merged.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    merged[k] = float(np.nanmean([r[k] for r in reps]))
            for k in METRIC_KEYS:
                merged[f"{k}_sd"] = (float(np.nanstd([r[k] for r in reps]))
                                     if len(reps) > 1 else float("nan"))
            merged["n_seeds"] = len(reps)
            merged["prognostic"] = PROGNOSTIC.get(cohort, True)
            per.append(merged)
            if len(reps) > 1:
                seed_spread.append(merged["centroid_corr_sd"])
        if not per:
            continue
        n_ok_cov = len(per) - sum(r["degenerate_events"] for r in per)
        for r in per:
            r["tstr_cohort_coverage"] = f"{n_ok_cov}/{len(per)}"
        rows.extend(per)
        m = lambda k: np.nanmean([r[k] for r in per])          # noqa: E731
        n_seeds = int(np.max([r.get("n_seeds", 1) for r in per]))
        sd_note = (f"  +/-{np.mean(seed_spread):.3f} over {n_seeds} seeds"
                   if seed_spread else "")
        prog = [r for r in per if r.get("prognostic", True)]
        sub_note = (f"  subKM*={np.nanmean([r['subtype_km_div'] for r in prog]):.3f}"
                    if prog and len(prog) < len(per) else "")
        n_bad = sum(r["degenerate_events"] for r in per)
        n_ok = len(per) - n_bad
        if np.isnan(m("tstr_cindex")):
            tstr_s = "  n/a"
        elif n_bad:
            # averaging over a fraction of the cohorts is not comparable to a
            # full average, so mark it rather than printing a bare number
            tstr_s = f"{m('tstr_cindex'):.3f}*"
        else:
            tstr_s = f"{m('tstr_cindex'):.3f}"
        flag = (f"   <-- no events in {n_bad}/{len(per)} cohorts; TSTR* is a mean "
                f"over only {n_ok}" if n_bad else "")
        print(f"  {method:16s} [{SOURCE.get(method,'?'):4s}] "
              f"W1={m('wasserstein'):.4f}  centroid_r={m('centroid_corr'):+.3f}  "
              f"subKM={m('subtype_km_div'):.3f}  KM={m('km_div'):.3f}  "
              f"TSTR={tstr_s}  disp={m('dispersion_ratio'):.2f}"
              f"{sub_note}{sd_note}{flag}")

    # ---- leaderboard: mean rank across the Four metrics ----
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    agg = {m: _agg_with_sd(rs) for m, rs in by.items()}

    lower_better = ["wasserstein", "subtype_km_div", "km_div"]
    higher_better = ["centroid_corr", "tstr_cindex"]
    names = list(agg)
    ranks = {m: [] for m in names}
    for k in lower_better + higher_better:
        vals = np.array([agg[m][k] for m in names], float)
        # NaN (e.g. TSTR when a generator produced no events) is a failure, so it
        # is pushed to the worst rank rather than dropped from the comparison.
        sign = 1.0 if k in lower_better else -1.0
        keyed = np.where(np.isnan(vals), np.inf, sign * vals)
        for pos, i in enumerate(np.argsort(keyed), start=1):
            ranks[names[i]].append(pos)
    mean_rank = {m: float(np.mean(v)) for m, v in ranks.items()}

    bad = sorted({r["method"] for r in rows if r["degenerate_events"]})
    if bad:
        print(f"\n  NOTE: {', '.join(bad)} produced fewer than 2 events in at "
              f"least one cohort.")
        print("        TSTR is undefined there and is ranked worst. Values marked")
        print("        with * are averaged over the remaining cohorts only and are")
        print("        NOT comparable to other methods' full averages. This is a")
        print("        real failure mode of the generator on imbalanced event")
        print("        columns, not a pipeline error.")

    if undertrained:
        print("\n  leaderboard SUPPRESSED -- the generator is undertrained "
              "(see the warning above).")
    else:
        print("\n  leaderboard (mean rank across the Four metrics, lower is "
              "better):")
        for i, (m, mr) in enumerate(sorted(mean_rank.items(),
                                           key=lambda kv: kv[1]), 1):
            star = " *" if m == "SubSurGen" else ""
            print(f"    {i:2d}. {m:16s} mean rank {mr:.2f}{star}")

    n_seeds_used = int(np.max([r.get("n_seeds", 1) for r in rows])) if rows else 1
    if n_seeds_used == 1:
        print("\n  NOTE: single sampling seed. Run-to-run variation on this "
              "benchmark is")
        print("        substantial for the subtype metrics (observed mean "
              "|delta| ~0.05 and")
        print("        max ~0.11 on subtype-centroid r between two otherwise "
              "identical runs),")
        print("        so differences smaller than that are not meaningful. "
              "Generate with")
        print("        several seeds for error bars:  "
              "python scripts/step2_generate.py --seeds 0 1 2 3 4")

    write_json("step4_benchmark.json",
               {"rows": rows,
                "aggregate": agg,
                "mean_rank": mean_rank,
                "n_seeds": n_seeds_used,
                "prognostic_cohorts": [c for c in episodes
                                       if PROGNOSTIC.get(c, True)],
                "aggregate_prognostic_only": {
                    m: _agg_with_sd([r for r in rows if r["method"] == m
                                     and r.get("prognostic", True)])
                    for m in by},
                "train_steps": trained_steps,
                "undertrained": bool(undertrained)})
    return {"rows": rows}


if __name__ == "__main__":
    main()
