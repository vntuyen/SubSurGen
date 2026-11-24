"""Step 6 -- Report: benchmark tables and figures."""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from common import (banner, list_synthetic_seeds, load_episodes,
                    load_synthetic, read_json, write_json)
import config as C
from subsurgen.snf import run_snf

warnings.filterwarnings("ignore")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

METRICS = [
    ("wasserstein",    "Mean Wasserstein",        "down"),
    ("centroid_corr",  "Subtype-centroid r",      "up"),
    ("subtype_km_div", "Subtype KM div",          "down"),
    ("tstr_cindex",    "TSTR C-index",            "up"),
]


UNDERTRAINED_STEPS = 1000


def _fmt(mean, sd, prec, sign=False):
    """'mean' to `prec` decimals, plus '± sd' when a seed-to-seed sd is
    available (finite and > 0). Falls back to a bare mean when it isn't --
    e.g. only one sampling seed was generated, or this is older data from
    before step4_evaluate.py tracked per-seed spread."""
    txt = f"{mean:+.{prec}f}" if sign else f"{mean:.{prec}f}"
    if sd is not None and np.isfinite(sd) and sd > 0:
        txt += f" \u00b1 {sd:.{prec}f}"
    return txt



def _mean_rank_four_metrics(agg):
    """Recompute mean rank using only METRICS.

    Step 4 JSON may contain a stored mean_rank calculated with an older metric
    set that included overall KM divergence. Step 6 deliberately ignores that
    stored rank so every reported rank/order reflects exactly the four metrics
    listed in METRICS.
    """
    methods = list(agg)
    ranks = {m: [] for m in methods}

    for key, _label, direction in METRICS:
        vals = {
            m: agg[m].get(key, float("nan"))
            for m in methods
            if np.isfinite(agg[m].get(key, float("nan")))
        }
        order = sorted(
            vals,
            key=lambda m: vals[m],
            reverse=(direction == "up"),
        )
        for pos, m in enumerate(order, 1):
            ranks[m].append(pos)

    return {
        m: float(np.mean(ranks[m])) if ranks[m] else float("nan")
        for m in methods
    }


def tables(R):
    L = ["# SubSurGen benchmark results", ""]

    hist = R.get("train_history") or {}
    steps = hist.get("steps")
    mode = hist.get("mode", "?")
    if steps is not None:
        L += [f"Training: `{mode}`, {steps} steps per model.", ""]
        if steps < UNDERTRAINED_STEPS:
            L += ["> # DO NOT PUBLISH THESE NUMBERS",
                  ">",
                  f"> This run used only **{steps} training steps** "
                  f"(a `--quick` smoke test). The generator is badly",
                  "> undertrained while every baseline is fitted normally, so the "
                  "comparison is invalid.",
                  "> Rerun with `python run_all.py --train-steps 2500` before "
                  "reading any table below.",
                  ""]

    if R.get("data"):
        L += ["## Table 1. Cohorts", "",
              "| Cohort | n | genes in | generated | for subtypes | K | K by "
              "stability | event rate | log-rank p | prognostic |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for d in R["data"]:
            # tolerate both the current and the pre-rename schema
            gen = d.get("n_genes_generated", d.get("n_genes_selected", "?"))
            sub = d.get("n_genes_for_subtypes", gen)
            prog = d.get("subtypes_prognostic")
            mark = "yes" if prog else ("**no**" if prog is not None else "-")
            L.append(f"| {d['cohort']} | {d['n_patients']} | {d['n_genes_all']} | "
                     f"{gen} | {sub} | {d.get('k_subtypes','-')} | "
                     f"{d.get('k_selected_by_stability','-')} | "
                     f"{d['event_rate']} | {d['logrank_p_real']:.4g} | {mark} |")
        fs = R["data"][0].get("feature_selection")
        if fs:
            L.append("")
            L.append(f"Feature-selection mode: `{fs}`.")
        L.append("")

    B = R.get("benchmark")
    if B:
        srcs = {r["method"]: r.get("source") for r in B["rows"]}
        if all(v in ("ref", None) for m, v in srcs.items() if m != "SubSurGen"):
            L += ["> # BASELINES ARE NOT THE PUBLISHED MODELS",
                  ">",
                  "> Every baseline in this run used an in-repo reimplementation "
                  "(`source: ref`).",
                  "> No official package or repository was detected. Install "
                  "`sdv` and `synthcity`,",
                  "> run `python scripts/setup_external.py`, and rerun step 2 "
                  "before publishing.", ""]
        agg = B["aggregate"]
        mr = _mean_rank_four_metrics(agg)
        src = {r["method"]: r for r in B["rows"]}
        n_seeds = B.get("n_seeds", 1)
        multi_seed = bool(n_seeds and n_seeds > 1)
        seed_note = f" (mean \u00b1 sd over up to {n_seeds} seeds)" if multi_seed else ""
        L += [f"## Table 2. Four-metric benchmark (mean over cohorts){seed_note}",
              "",
              "| Method | Family | Source | W1 v | centroid r ^ | "
              "subKM v | TSTR ^ | mean rank |",
              "|---|---|---|---|---|---|---|---|"]
        for m in sorted(agg, key=lambda x: mr[x]):
            a, meta = agg[m], src[m]
            b = "**" if m == "SubSurGen" else ""
            L.append(f"| {b}{m}{b} | {meta['family']} | {meta['source']} | "
                     f"{_fmt(a['wasserstein'], a.get('wasserstein_sd'), 4)} | "
                     f"{_fmt(a['centroid_corr'], a.get('centroid_corr_sd'), 3, sign=True)} | "
                     f"{_fmt(a['subtype_km_div'], a.get('subtype_km_div_sd'), 3)} | "
                     f"{_fmt(a['tstr_cindex'], a.get('tstr_cindex_sd'), 3)} | "
                     f"{b}{mr[m]:.2f}{b} |")
        L += ["", "v = lower is better, ^ = higher is better. Source: `ours`, "
              "`pip` (official package), `repo` (official GitHub), `ref` "
              "(in-repo reimplementation -- not the published model)."
              + (f" \u00b1 is the seed-to-seed sd (up to {n_seeds} seeds)."
                 if multi_seed else ""), ""]

        # per-metric winners
        L += ["## Table 3. Best method per metric", "",
              "| Metric | Best | Value | SubSurGen | rank |", "|---|---|---|---|---|"]
        for key, label, direction in METRICS:
            vals = {m: agg[m][key] for m in agg if np.isfinite(agg[m][key])}
            best = (min if direction == "down" else max)(vals, key=vals.get)
            order = sorted(vals, key=lambda m: vals[m],
                           reverse=(direction == "up"))
            rank = order.index("SubSurGen") + 1 if "SubSurGen" in order else None
            best_sd = agg[best].get(f"{key}_sd")
            sub_val = vals.get("SubSurGen", float("nan"))
            sub_sd = agg.get("SubSurGen", {}).get(f"{key}_sd")
            L.append(f"| {label} | {best} | {_fmt(vals[best], best_sd, 4)} | "
                     f"{_fmt(sub_val, sub_sd, 4)} | "
                     f"{rank}/{len(order)} |")
        L.append("")

    if R.get("benchmark") and R["benchmark"].get("aggregate_prognostic_only"):
        ap = R["benchmark"]["aggregate_prognostic_only"]
        pc = R["benchmark"].get("prognostic_cohorts", [])
        agg = R["benchmark"]["aggregate"]
        L += [f"## Prognostic-subtype cohorts only ({len(pc)} of "
              f"{len(R.get('data', []))})", "",
              f"Cohorts: {', '.join(pc)} (log-rank p < 0.05).", "",
              "| Method | subKM (all) | subKM (prognostic only) | centroid r "
              "(prognostic only) |", "|---|---|---|---|"]
        for m in sorted(ap, key=lambda x: ap[x]["subtype_km_div"]):
            b = "**" if m == "SubSurGen" else ""
            L.append(f"| {b}{m}{b} | "
                     f"{_fmt(agg[m]['subtype_km_div'], agg[m].get('subtype_km_div_sd'), 4)} | "
                     f"{_fmt(ap[m]['subtype_km_div'], ap[m].get('subtype_km_div_sd'), 4)} | "
                     f"{_fmt(ap[m]['centroid_corr'], ap[m].get('centroid_corr_sd'), 3, sign=True)} |")
        L.append("")

    if R.get("benchmark") and R["benchmark"].get("rows"):
        rows = R["benchmark"]["rows"]
        cohorts = sorted({r["cohort"] for r in rows})
        L += ["## Per-cohort comparison", ""]
        for metric, label, better in [
                ("centroid_corr", "Subtype-centroid correlation", "higher"),
                ("subtype_km_div", "Subtype-specific KM divergence", "lower"),
                ("tstr_cindex", "TSTR C-index", "higher")]:
            meths = sorted({r["method"] for r in rows})
            L += [f"### {label} ({better} is better)", "",
                  "| Cohort | prognostic | " + " | ".join(meths) + " |",
                  "|---" * (len(meths) + 2) + "|"]
            for c in cohorts:
                vals, sds = {}, {}
                for m in meths:
                    hit = [r for r in rows
                           if r["cohort"] == c and r["method"] == m]
                    vals[m] = hit[0][metric] if hit else float("nan")
                    sds[m] = hit[0].get(f"{metric}_sd") if hit else None
                good = [v for v in vals.values() if np.isfinite(v)]
                best = (max(good) if better == "higher" else min(good)) if good else None
                cells = []
                for m in meths:
                    v = vals[m]
                    txt = "-" if not np.isfinite(v) else _fmt(v, sds[m], 3)
                    if best is not None and np.isfinite(v) and abs(v - best) < 1e-9:
                        txt = f"**{txt}**"
                    cells.append(txt)
                hit = [r for r in rows if r["cohort"] == c]
                pr = "yes" if hit and hit[0].get("prognostic", True) else "no"
                L.append(f"| {c} | {pr} | " + " | ".join(cells) + " |")
            L.append("")

    if R.get("ablation"):
        abl_mode = R["ablation"][0].get("training_mode", "?")
        abl_cohorts = R["ablation"][0].get("cohorts")
        abl_seeds = R["ablation"][0].get("n_seeds", 1)
        abl_multi_seed = bool(abl_seeds and abl_seeds > 1)
        seed_note = f" (mean \u00b1 sd over {abl_seeds} seeds)" if abl_multi_seed else ""
        L += [f"## Table 4. Ablations{seed_note}", ""]
        hist_mode = (R.get("train_history") or {}).get("mode")
        if hist_mode and abl_mode != "?" and abl_mode != hist_mode:
            L += [f"> **Not comparable to Table 2.** These ablations used "
                  f"`{abl_mode}` training while the benchmark used `{hist_mode}`.",
                  "> Rerun step5_ablation.py with `--mode " + str(hist_mode) + "`.", ""]
        else:
            L += [f"Training: `{abl_mode}`"
                  + (f", cohorts: {', '.join(abl_cohorts)}" if abl_cohorts else "")
                  + ".", ""]
        g_src = R["ablation"][0].get("guidance_source")
        if g_src == "config.py fallback":
            L += ["> **Untuned guidance.** No `results/tuning.json` was found; "
                  "`full` used the `config.py` default rather than a value "
                  "`step2_tune.py` selected.", ""]
        if not abl_multi_seed:
            L += ["> **Single sampling seed.** Differences between configs "
                  "here have not been checked against run-to-run noise; pass "
                  "`--seeds 0 1 2 3 4` to `step5_ablation.py` before treating "
                  "any comparison as reliable.", ""]
        L += ["| Config | W1 v | centroid r ^ | subKM v | TSTR ^ | dispersion |",
              "|---|---|---|---|---|---|"]
        for r in R["ablation"]:
            tstr = r.get("tstr_cindex", float("nan"))
            tstr_s = _fmt(tstr, r.get("tstr_cindex_sd"), 3) if np.isfinite(tstr) else "n/a"
            disp = r.get("dispersion_ratio")
            disp_s = "-" if disp is None or not np.isfinite(disp) else f"{disp:.2f}"
            L.append(f"| {r['config']} | "
                     f"{_fmt(r['wasserstein'], r.get('wasserstein_sd'), 4)} | "
                     f"{_fmt(r['centroid_corr'], r.get('centroid_corr_sd'), 3, sign=True)} | "
                     f"{_fmt(r['subtype_km_div'], r.get('subtype_km_div_sd'), 3)} | "
                     f"{tstr_s} | {disp_s} |")
        L.append("")

    if R.get("guidance_sweep"):
        L += ["## Table 5. Guidance sweep", "",
              "| w | W1 v | centroid r ^ | subKM v | TSTR ^ |",
              "|---|---|---|---|---|"]
        for r in R["guidance_sweep"]:
            tstr = r.get("tstr_cindex", float("nan"))
            tstr_s = f"{tstr:.3f}" if np.isfinite(tstr) else "n/a"
            L.append(
                f"| {r['guidance']:.1f} | "
                f"{r['wasserstein']:.4f} | "
                f"{r['centroid_corr']:+.3f} | "
                f"{r['subtype_km_div']:.3f} | "
                f"{tstr_s} |"
            )
        L.append("")
    return "\n".join(L)


def _bold_axis_text(ax, tick_size=None):
    """Apply bold styling to axis labels, tick labels, and subplot title."""
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        if tick_size is not None:
            label.set_fontsize(tick_size)


def _save_figure(fig, stem, dpi=300):
    """Save a figure as both a high-resolution PNG and a vector PDF."""
    os.makedirs(C.FIGURES_DIR, exist_ok=True)

    png_name = f"{stem}.png"
    pdf_name = f"{stem}.pdf"

    fig.savefig(
        os.path.join(C.FIGURES_DIR, png_name),
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(C.FIGURES_DIR, pdf_name),
        format="pdf",
        bbox_inches="tight",
    )
    return [png_name, pdf_name]


def figures(R):
    B = R.get("benchmark")
    if not B:
        return []

    agg = B["aggregate"]
    mr = _mean_rank_four_metrics(agg)
    order = sorted(agg, key=lambda m: mr[m])
    made = []

    # Per-metric bars for the four metrics reported in the paper.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.ravel()

    for ax, (key, label, direction) in zip(axes_flat, METRICS):
        vals = [agg[m][key] for m in order]

        # Use zero error bars when seed-level SD is unavailable.
        errs = [agg[m].get(f"{key}_sd") for m in order]
        errs = [
            e if e is not None and np.isfinite(e) else 0
            for e in errs
        ]

        cols = [
            "#2E5EAA" if m == "SubSurGen" else "#B0B0B0"
            for m in order
        ]

        ax.bar(
            range(len(order)),
            vals,
            yerr=errs,
            capsize=3,
            color=cols,
            edgecolor="black",
            lw=0.5,
            error_kw=dict(ecolor="#444444", lw=1),
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            order,
            rotation=60,
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

        arrow = "lower better" if direction == "down" else "higher better"
        ax.set_title(
            f"{label} ({arrow})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)
        _bold_axis_text(ax)

    fig.suptitle(
        "SubSurGen benchmark: four metrics",
        y=1.01,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    made.extend(_save_figure(fig, "four_metrics", dpi=300))
    plt.close(fig)

    # Subtype fidelity vs subtype-survival fidelity.
    fig, ax = plt.subplots(figsize=(7, 5))

    for m in order:
        x = agg[m]["subtype_km_div"]
        y = agg[m]["centroid_corr"]

        xerr = agg[m].get("subtype_km_div_sd")
        yerr = agg[m].get("centroid_corr_sd")

        xerr = xerr if xerr is not None and np.isfinite(xerr) else 0
        yerr = yerr if yerr is not None and np.isfinite(yerr) else 0

        ours = m == "SubSurGen"
        col = "#2E5EAA" if ours else "#B0B0B0"

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=11 if ours else 7,
            color=col,
            ecolor=col,
            elinewidth=1,
            capsize=3,
            markeredgecolor="black",
            zorder=3 if ours else 2,
        )
        ax.annotate(
            m,
            (x, y),
            fontsize=8,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Subtype-specific KM divergence (lower better)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Subtype-centroid correlation (higher better)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title(
        "Subtype preservation: molecular profile vs prognosis",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    _bold_axis_text(ax)
    fig.tight_layout()
    made.extend(_save_figure(fig, "subtype_plane", dpi=300))
    plt.close(fig)

    return made


def _km_curve(time, event):
    """Kaplan-Meier product-limit estimator. Returns (times, survival) step
    points starting at (0, 1.0), for plotting with ax.step(..., where='post')."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time)
    t, e = time[order], event[order]
    xs, ys = [0.0], [1.0]
    s = 1.0
    for ut in np.unique(t[e == 1]):
        at_risk = int(np.sum(t >= ut))
        d = int(np.sum((t == ut) & (e == 1)))
        if at_risk > 0:
            s *= (1 - d / at_risk)
        xs.append(float(ut))
        ys.append(s)
    return np.array(xs), np.array(ys)


def _match_subtypes(Xr, sr, Xs, ss):
    """Greedy real-to-synthetic subtype pairing by centroid correlation, so
    matched subtypes can share a color in the KM-curve plot. This is separate
    from whatever pairing subtype_centroid_correlation uses internally for the
    reported metric -- it exists only for consistent plot labeling."""
    real_labels = sorted(np.unique(sr))
    remaining = set(np.unique(ss))
    pairs = {}
    for r in real_labels:
        if not remaining:
            pairs[r] = None
            continue
        rc = Xr[sr == r].mean(axis=0)
        best = max(remaining,
                   key=lambda s: np.corrcoef(rc, Xs[ss == s].mean(axis=0))[0, 1])
        pairs[r] = best
        remaining.discard(best)
    return pairs


_SUBTYPE_COLORS = ["#2E5EAA", "#D1495B", "#3CAEA3", "#F4A261", "#8E44AD", "#5A6650"]


def km_curve_figures() -> list:
    """Per-cohort KM-curve diagnostics for SubSurGen: real vs synthetic,
    both whole-cohort and per (matched) subtype -- a visual diagnostic of whole-cohort and subtype-specific
    survival curves. Overall KM divergence is not reported as a benchmark metric. Synthetic curves pool every cached seed for that cohort into
    one larger sample (steadier curve than any single seed); this is a
    complementary visual, not a literal decomposition of the per-seed-
    averaged metric reported elsewhere. Self-contained: loads episodes and
    cached synthetic data directly rather than going through the R dict,
    since none of that patient-level detail is in the JSON summaries.
    """
    try:
        episodes, _ = load_episodes()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (KM-curve figures skipped: no episodes -- {exc})")
        return []

    made = []
    for cohort, ep in episodes.items():
        seeds = list_synthetic_seeds("SubSurGen", cohort)
        if not seeds:
            continue
        Xs_parts, ts_parts, es_parts = [], [], []
        for sd in seeds:
            syn = load_synthetic("SubSurGen", cohort, seed=sd)
            if syn is None:
                continue
            Xb, tb, eb = syn
            Xs_parts.append(Xb)
            ts_parts.append(tb)
            es_parts.append(eb)
        if not Xs_parts:
            continue
        Xs = np.concatenate(Xs_parts, axis=0)
        ts = np.concatenate(ts_parts, axis=0)
        es = np.concatenate(es_parts, axis=0)

        Xr, tr, er, sr = ep.cohort.X, ep.cohort.time, ep.cohort.event, ep.labels
        k = len(np.unique(sr))
        ss = run_snf(Xs, k_subtypes=k, K=C.SNF_K, t=C.SNF_T,
                     n_views=C.SNF_VIEWS, emb_dim=C.COND_DIM,
                     seed=C.SEED)["labels"]
        pairs = _match_subtypes(Xr, sr, Xs, ss)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        rx, ry = _km_curve(tr, er)
        sx, sy = _km_curve(ts, es)
        ax1.step(rx, ry, where="post", color="black", lw=2, label=f"real (n={len(tr)})")
        ax1.step(sx, sy, where="post", color="#2E5EAA", lw=2, ls="--",
                 label=f"SubSurGen (n={len(ts)})")
        ax1.set_title("Whole cohort", fontweight="bold")
        ax1.set_xlabel("Time (years)", fontweight="bold")
        ax1.set_ylabel("Survival Probability", fontweight="bold")
        ax1.set_ylim(0, 1.02)
        ax1.grid(alpha=0.3)
        leg1 = ax1.legend(fontsize=10)
        for txt in leg1.get_texts():
            txt.set_fontweight("bold")
        _bold_axis_text(ax1)

        for i, r in enumerate(sorted(pairs)):
            col = _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)]
            rx, ry = _km_curve(tr[sr == r], er[sr == r])
            ax2.step(rx, ry, where="post", color=col, lw=2,
                     label=f"subtype {r} real (n={int((sr == r).sum())})")
            s_lab = pairs[r]
            if s_lab is not None:
                sx, sy = _km_curve(ts[ss == s_lab], es[ss == s_lab])
                ax2.step(sx, sy, where="post", color=col, lw=2, ls="--",
                         label=f"subtype {r} synthetic "
                               f"(n={int((ss == s_lab).sum())})")
        ax2.set_title(
            "By subtype (matched real vs synthetic)",
            fontweight="bold",
        )
        ax2.set_xlabel("Time (years)", fontweight="bold")
        ax2.set_ylabel("Survival Probability", fontweight="bold")
        ax2.set_ylim(0, 1.02)
        ax2.grid(alpha=0.3)
        leg2 = ax2.legend(fontsize=9, loc="lower left")
        for txt in leg2.get_texts():
            txt.set_fontweight("bold")
        _bold_axis_text(ax2)

        fig.suptitle(
            f"{cohort}: SubSurGen KM curves, real vs synthetic",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        made.extend(_save_figure(fig, f"km_{cohort}", dpi=300))
        plt.close(fig)

    if not made:
        print("  (no cached SubSurGen synthetic data found for KM curves -- "
              "run step 3 first)")
    return made


def main():
    argparse.ArgumentParser(description="Build report").parse_args()
    banner("STEP 6 -- REPORT")
    R = {"data": read_json("step0_data_summary.json"),
         "benchmark": read_json("step4_benchmark.json"),
         "ablation": read_json("step5_ablation.json"),
         "guidance_sweep": read_json("step2_guidance_sweep.json"),
         "generation_log": read_json("step3_generation_log.json"),
         "tuning": read_json("tuning.json"),
         "train_history": read_json("step1_train_history.json")}
    R = {k: v for k, v in R.items() if v}
    print("available:", ", ".join(R))

    path = os.path.join(C.RESULTS_DIR, "RESULTS.md")
    with open(path, "w") as f:
        f.write(tables(R) + "\n")
    print(f"  -> {path}")
    write_json("subsurgen_all_results.json", R)

    if HAVE_MPL:
        for m in figures(R):
            print(f"  -> {C.FIGURES_DIR}/{m}")
        for m in km_curve_figures():
            print(f"  -> {C.FIGURES_DIR}/{m}")
    else:
        print("  (matplotlib missing; figures skipped)")
    return R


if __name__ == "__main__":
    main()