#!/usr/bin/env python
"""
Step 5 -- focused SubSurGen ablation study.

This version trains ONLY two ablation configurations:

    no_interpolation
        SNF/spectral embedding ON
        subtype label ON
        interpolation OFF

    embedding_only
        SNF/spectral embedding ON
        subtype label OFF
        interpolation OFF

The Full/SubSurGen result is read directly from
step4_benchmark.json, expected in the same results directory as step5_ablation.json.

Usage:
    python scripts/step5_ablation.py
    python scripts/step5_ablation.py --seeds 0 1 2 3 4
    python scripts/step5_ablation.py --configs no_interpolation
    python scripts/step5_ablation.py --configs embedding_only
"""
from __future__ import annotations

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np

from common import (
    banner,
    get_device,
    load_episodes,
    make_diffusion,
    make_model,
    parse_common_args,
    resolve_guidance,
    training_mode,
    write_json,
)
import config as C

from subsurgen.downstream import cox_cindex, fit_cox
from subsurgen.metrics import (
    km_divergence,
    mean_wasserstein,
    subtype_centroid_correlation,
    subtype_km_divergence,
    within_subtype_dispersion,
)
from subsurgen.snf import run_snf
from subsurgen.trainer import generate, train_episodic

warnings.filterwarnings("ignore")

_G, _G_SRC = resolve_guidance()

DEFAULT_ABLATION_SEEDS = [0, 1, 2, 3, 4]

METRIC_KEYS = (
    "wasserstein",
    "centroid_corr",
    "subtype_km_div",
    "tstr_cindex",
)

CONFIGS = {
    "no_interpolation": dict(
        lam=C.LAMBDA_COX,
        interp=False,
        emb=True,
        lab=True,
        guid=_G,
    ),
    "embedding_only": dict(
        lam=C.LAMBDA_COX,
        interp=False,
        emb=True,
        lab=False,
        guid=_G,
    ),
}


def _find_step4_benchmark() -> Path:
    """Find step4_benchmark.json in the normal project/results locations."""
    project_root = Path(__file__).resolve().parents[1]

    candidates = [
        project_root / "results" / "step4_benchmark.json",
        Path("results") / "step4_benchmark.json",
        project_root / "step4_benchmark.json",
        Path("step4_benchmark.json"),
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find step4_benchmark.json.\n"
        "Expected it in the same results directory used by Step 5.\n"
        f"Tried:\n{tried}"
    )


def _load_full_from_step4(
    step4_path: Path,
    selected_cohorts: list[str] | None = None,
) -> dict:
    """
    Load the existing Full/SubSurGen result from Step 4.

    If all cohorts are used, copy aggregate['SubSurGen'] directly.
    If --cohorts is supplied, recompute the Full summary from the matching
    Step 4 per-cohort SubSurGen rows.
    """
    with step4_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        r for r in data.get("rows", [])
        if r.get("method") == "SubSurGen"
    ]

    if not rows:
        raise ValueError(
            f"{step4_path} contains no rows with method='SubSurGen'."
        )

    if selected_cohorts:
        wanted = set(selected_cohorts)
        rows_use = [r for r in rows if r["cohort"] in wanted]
        missing = sorted(wanted - {r["cohort"] for r in rows_use})
        if missing:
            raise ValueError(
                f"Missing Step 4 SubSurGen cohort(s): {missing}"
            )

        use_official_aggregate = (
            len(rows_use) == len(rows)
            and {r["cohort"] for r in rows_use}
            == {r["cohort"] for r in rows}
        )
    else:
        rows_use = rows
        use_official_aggregate = True

    if use_official_aggregate:
        agg = data.get("aggregate", {}).get("SubSurGen")
        if agg is None:
            raise ValueError(
                f"{step4_path} has no aggregate['SubSurGen'] entry."
            )
        summary = dict(agg)
    else:
        summary = {}
        for k in METRIC_KEYS:
            vals = np.asarray(
                [r.get(k, np.nan) for r in rows_use],
                dtype=float,
            )
            summary[k] = float(np.nanmean(vals))

            sd_key = f"{k}_sd"
            sd_vals = np.asarray(
                [r.get(sd_key, np.nan) for r in rows_use],
                dtype=float,
            )
            summary[sd_key] = float(np.nanmean(sd_vals))

    disp = np.asarray(
        [r.get("dispersion_ratio", np.nan) for r in rows_use],
        dtype=float,
    )
    summary["dispersion_ratio"] = float(np.nanmean(disp))

    summary.update(
        {
            "config": "full",
            "source": "step4_benchmark.json",
            "source_path": str(step4_path),
            "reused": True,
            "trained_in_step5": False,
            "n_seeds": int(
                data.get("n_seeds", rows_use[0].get("n_seeds", 0))
            ),
            "seeds": "reused_from_step4",
            "guidance": None,
            "guidance_source": "not stored in step4_benchmark.json",
            "training_mode": "reused Step 4 benchmark result",
            "cohorts": [r["cohort"] for r in rows_use],
            "tstr_source": "step4_benchmark.json",
            "tstr_strictly_comparable_to_step4": True,
        }
    )

    return summary


def _tstr(ep, Xs, ts, es):
    """
    Historical Step 5 TSTR implementation.

    Note: the previous Step 5 Full row matched Step 4 on the four fidelity
    metrics but not exactly on TSTR. Therefore Step 4 Full TSTR is treated as
    authoritative, while new ablation TSTR values are kept as diagnostics.
    """
    if es.sum() < 2:
        return float("nan")

    vals = []

    for i in range(C.N_REPEATS):
        rng = np.random.default_rng(700 + i)
        idx = rng.permutation(ep.cohort.n)
        n_tr = int((1 - C.TEST_FRACTION) * ep.cohort.n)

        _, te = idx[:n_tr], idx[n_tr:]

        if ep.cohort.event[te].sum() < 2:
            continue

        sub = rng.choice(
            len(Xs),
            size=min(len(Xs), n_tr),
            replace=False,
        )

        if es[sub].sum() < 2:
            continue

        beta = fit_cox(
            Xs[sub],
            ts[sub],
            es[sub],
            seed=i,
        )

        vals.append(
            cox_cindex(
                beta,
                ep.cohort.X[te],
                ep.cohort.time[te],
                ep.cohort.event[te],
            )
        )

    return float(np.mean(vals)) if vals else float("nan")


def score(ep, Xs, ts, es):
    """Score one generated synthetic cohort."""
    k = (
        ep.n_subtypes
        if hasattr(ep, "n_subtypes")
        else len(np.unique(ep.labels))
    )

    ss = run_snf(
        Xs,
        k_subtypes=k,
        K=C.SNF_K,
        t=C.SNF_T,
        n_views=C.SNF_VIEWS,
        emb_dim=C.COND_DIM,
        seed=C.SEED,
    )["labels"]

    cc = subtype_centroid_correlation(
        ep.cohort.X,
        ep.labels,
        Xs,
        ss,
        center=True,
    )

    skm = subtype_km_divergence(
        ep.cohort.time,
        ep.cohort.event,
        ep.labels,
        ts,
        es,
        ss,
        cc["pairs"],
    )

    disp = within_subtype_dispersion(
        ep.cohort.X,
        ep.labels,
        Xs,
        ss,
    )

    return {
        "wasserstein": mean_wasserstein(ep.cohort.X, Xs),
        "centroid_corr": cc["value"],
        "subtype_km_div": skm["value"],
        "km_div": km_divergence(
            ep.cohort.time,
            ep.cohort.event,
            ts,
            es,
        ),
        "dispersion_ratio": disp["ratio"],
        "tstr_cindex": _tstr(ep, Xs, ts, es),
    }


def _merge_sampling_replicates(reps: list[dict]) -> dict:
    """Average sampling seeds within one cohort."""
    merged = {}

    for key in reps[0]:
        vals = np.asarray(
            [r.get(key, np.nan) for r in reps],
            dtype=float,
        )
        merged[key] = float(np.nanmean(vals))

    for key in METRIC_KEYS:
        vals = np.asarray(
            [r.get(key, np.nan) for r in reps],
            dtype=float,
        )
        merged[f"{key}_sd"] = (
            float(np.nanstd(vals))
            if len(reps) > 1
            else float("nan")
        )

    return merged


def _aggregate_cohorts(
    config_name: str,
    per_cohort: list[dict],
    seeds: list[int],
    cfg: dict,
    mode: str,
    cohort_names: list[str],
) -> dict:
    """Aggregate cohort-level metrics into one summary row."""
    row = {
        "config": config_name,
        "source": "step5_ablation.py",
        "reused": False,
        "trained_in_step5": True,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "guidance": float(cfg["guid"]),
        "guidance_source": _G_SRC,
        "training_mode": mode,
        "cohorts": cohort_names,
        "tstr_source": "historical Step 5 _tstr()",
        "tstr_strictly_comparable_to_step4": False,
    }

    numeric_keys = [
        "wasserstein",
        "centroid_corr",
        "subtype_km_div",
        "km_div",
        "dispersion_ratio",
        "tstr_cindex",
        "wasserstein_sd",
        "centroid_corr_sd",
        "subtype_km_div_sd",
        "km_div_sd",
        "tstr_cindex_sd",
    ]

    for key in numeric_keys:
        vals = np.asarray(
            [r.get(key, np.nan) for r in per_cohort],
            dtype=float,
        )
        row[key] = float(np.nanmean(vals))

    return row


def _print_row(row: dict) -> None:
    tstr = row.get("tstr_cindex", float("nan"))
    tstr_s = "n/a" if np.isnan(tstr) else f"{tstr:.3f}"
    source = "REUSED step4" if row.get("reused") else "trained step5"

    print(
        f"  {row['config']:18s} "
        f"W1={row['wasserstein']:.4f}  "
        f"centroid_r={row['centroid_corr']:+.3f}  "
        f"subKM={row['subtype_km_div']:.3f}  "
        f"KM={row['km_div']:.3f}  "
        f"TSTR={tstr_s}  "
        f"disp={row['dispersion_ratio']:.2f}  "
        f"[{source}]"
    )


def main() -> dict:
    p = argparse.ArgumentParser(
        description=(
            "Focused SubSurGen ablation: reuse Full from Step 4 and train "
            "only no_interpolation + embedding_only."
        )
    )

    p.add_argument(
        "--configs",
        nargs="+",
        default=["no_interpolation", "embedding_only"],
        choices=list(CONFIGS),
        help=(
            "Ablations to retrain. Full is always reused from "
            "step4_benchmark.json."
        ),
    )

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_ABLATION_SEEDS,
        help="Sampling seeds per trained ablation model.",
    )

    p.add_argument(
        "--cohorts",
        nargs="*",
        default=None,
        help=(
            "Optional cohort subset. If supplied, the reused Full row is "
            "recomputed from matching Step 4 per-cohort SubSurGen rows."
        ),
    )

    p.add_argument(
        "--mode",
        choices=["per_cohort", "episodic"],
        default=None,
    )

    args = parse_common_args(p)

    banner("STEP 5 -- ABLATION STUDY")

    step4_path = _find_step4_benchmark()

    print(f"Step 4 benchmark: {step4_path}")
    print("Full will be REUSED from Step 4;")
    print("Training: " + ", ".join(args.configs))
    print(f"Sampling seeds: {args.seeds}")
    print(f"Guidance: {_G} [{_G_SRC}]")

    episodes, vocab = load_episodes()

    if args.cohorts:
        requested = set(args.cohorts)
        episodes = {
            k: v
            for k, v in episodes.items()
            if k in requested
        }

        missing = sorted(requested - set(episodes))
        if missing:
            raise ValueError(
                f"Requested cohort(s) not found: {missing}"
            )

    mode = args.mode or training_mode()
    cohort_names = list(episodes)

    if mode != "per_cohort":
        print(
            "\nWARNING: For comparison with the Step 4 benchmark, "
            "per_cohort mode is preferred.\n"
        )

    full_row = _load_full_from_step4(
        step4_path,
        selected_cohorts=cohort_names if args.cohorts else None,
    )

    results = [full_row]

    print("\nReused Full reference:")
    _print_row(full_row)

    diffusion = make_diffusion()

    for config_name in args.configs:
        cfg = CONFIGS[config_name]

        print(
            f"\nTraining {config_name}: "
            f"emb={cfg['emb']} "
            f"label={cfg['lab']} "
            f"interp={cfg['interp']} "
            f"lambda_cox={cfg['lam']} "
            f"guidance={cfg['guid']}"
        )

        eps_use = {}

        for cohort, ep in episodes.items():
            e = copy.copy(ep)

            if not cfg["emb"]:
                e.emb = np.zeros_like(ep.emb)

            if not cfg["lab"]:
                e.labels = np.zeros_like(ep.labels)

            eps_use[cohort] = e

        def _fit(train_eps):
            model = make_model(len(vocab))

            train_episodic(
                model,
                diffusion,
                train_eps,
                steps=C.TRAIN_STEPS,
                batch_size=C.BATCH_SIZE,
                lr=C.LR,
                lambda_cox=cfg["lam"],
                feature_dropout=C.FEATURE_DROPOUT,
                cond_dropout=C.COND_DROPOUT,
                seed=C.SEED,
                verbose=False,
                log_every=10**9,
                device=get_device(),
            )

            model.eval()
            return model

        shared_model = (
            None
            if mode == "per_cohort"
            else _fit(list(eps_use.values()))
        )

        per_cohort = []

        for cohort, ep in episodes.items():
            model = (
                shared_model
                if shared_model is not None
                else _fit([eps_use[cohort]])
            )

            reps = []

            for sampling_seed in args.seeds:
                Xs, ts, es, _ = generate(
                    model,
                    diffusion,
                    eps_use[cohort],
                    ep.cohort.n,
                    seed=sampling_seed,
                    interpolate=cfg["interp"],
                    guidance=cfg["guid"],
                )

                reps.append(score(ep, Xs, ts, es))

            merged = _merge_sampling_replicates(reps)
            merged["cohort"] = cohort
            per_cohort.append(merged)

            print(
                f"    {cohort:12s} "
                f"W1={merged['wasserstein']:.4f}  "
                f"centroid_r={merged['centroid_corr']:+.3f}  "
                f"subKM={merged['subtype_km_div']:.3f}  "
                f"KM={merged['km_div']:.3f}  "
                f"TSTR={merged['tstr_cindex']:.3f}"
            )

        row = _aggregate_cohorts(
            config_name=config_name,
            per_cohort=per_cohort,
            seeds=args.seeds,
            cfg=cfg,
            mode=mode,
            cohort_names=cohort_names,
        )

        row["per_cohort"] = per_cohort
        results.append(row)

        print("\nAggregate:")
        _print_row(row)

    print("\n" + "-" * 76)
    print("FINAL ABLATION SUMMARY")
    print("-" * 76)

    for row in results:
        _print_row(row)

    print(
        "\nComparisons:"
        "\n  Full vs no_interpolation"
        "\n      -> isolates interpolation."
        "\n  no_interpolation vs embedding_only"
        "\n      -> isolates the discrete subtype-label contribution while "
        "keeping embedding ON and interpolation OFF in both."
    )

    print(
        "\nTSTR caution:"
        "\n  Full TSTR is reused directly from Step 4."
        "\n  New ablation TSTR values use the historical Step 5 evaluator, "
        "which previously did not exactly reproduce Step 4 TSTR."
    )

    write_json("step5_ablation.json", results)

    print("\nWritten: step5_ablation.json")

    return {
        "rows": results,
        "step4_source": str(step4_path),
    }


if __name__ == "__main__":
    main()
