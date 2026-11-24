"""Step 3 -- Generate synthetic cohorts from every method.

Runs SubSurGen and every registered baseline on each cohort and caches the result to
results/synthetic/<Method>/<Cohort>.{npz,csv}. Evaluation happens separately in
step 4, so that generation (slow, sometimes dependent on external repos) and
scoring (fast, pure numpy) can be rerun independently.



Usage:
    python scripts/step3_generate.py                       # SubSurGen + all baselines
    python scripts/step3_generate.py --methods CTGAN TVAE  # subset
    python scripts/step3_generate.py --skip-baselines      # SubSurGen only
"""
from __future__ import annotations

# Set before NumPy/PyTorch/Synthcity imports to reduce macOS native-runtime crashes.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from common import (banner, describe_device, get_device, load_synthetic,
                    load_trained_model, model_for, parse_common_args,
                    resolve_guidance, save_synthetic, training_mode, write_json)
import config as C
from subsurgen.baselines import (BASELINES, IMPORT_ONLY, availability,
                                 available_methods, is_official, last_source,
                                 registry_report, source_of, synthcity_status)
from subsurgen.trainer import generate

warnings.filterwarnings("ignore")


def export(method, cohort, genes, X, t, e, seed=None):
    save_synthetic(method, cohort, X, t, e, genes, seed=seed)
    import os
    d = os.path.join(C.SYNTHETIC_DIR, method)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(X, columns=list(genes))
    df["time"] = t
    df["event"] = e.astype(int)
    tag = "" if seed is None else f"__s{seed}"
    df.to_csv(os.path.join(d, f"{cohort}{tag}.csv"), index=False)


def main() -> dict:
    p = argparse.ArgumentParser(description="Generate synthetic cohorts")
    p.add_argument("--methods", nargs="*", default=None, choices=list(BASELINES))
    p.add_argument("--skip-baselines", action="store_true",
                   help="generate SubSurGen only, leaving baselines untouched")
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="sampling seeds; several enable error bars in step 3")
    p.add_argument("--skip-existing", action="store_true",
                   help="do not regenerate a method/cohort that is already cached")
    args = parse_common_args(p)
    guidance, g_src = resolve_guidance(args.guidance)
    epochs = args.epochs or C.BASELINE_EPOCHS

    banner("STEP 3 -- SYNTHETIC COHORT GENERATION")
    print(f"synthcity : {synthcity_status()}")
    avail = availability()
    print(f"sdv       : {avail['sdv']}")
    print()
    print(registry_report())
    print()
    refs = [m for m in BASELINES
            if m not in IMPORT_ONLY and source_of(m) == "ref"]
    if refs:
        print(f"NOTE: {', '.join(refs)} will run as in-repo reimplementations,")
        print("      not the published models. Install synthcity to replace the")
        print("      diffusion and survival families with official ones:")
        print("          pip install synthcity")
        print("      (if it's installed but unusable, run scripts/setup_external.py")
        print("      --list, or see subsurgen/baselines.py:_torch_supports_synthcity")
        print("      for the actual cause -- it is almost never the torch version)")
        print()

    failures: dict = {}
    models, diffusion, episodes, _ = load_trained_model()
    mode = training_mode()
    log = []

    seeds = args.seeds if args.seeds else [None]
    if len(seeds) > 1:
        print(f"\nsampling seeds: {seeds}  (step 4 will report mean +/- sd)")

    print(f"\nSubSurGen (guidance={guidance} [{g_src}], training={mode}):")
    if g_src == "config.py fallback" and C.AUTO_TUNE_GUIDANCE:
        print("  NOTE: guidance has not been tuned. Run scripts/step2_tune.py")
        print("        to select it from the data instead of using the fallback.")
    if mode == "episodic":
        print("  NOTE: episodic training saw all cohorts; baselines are fitted")
        print("        per cohort. Not a like-for-like comparison.")
    for cohort, ep in episodes.items():
        t0 = time.time()
        # Skip PER SEED, not per cohort: adding seeds 3 and 4 to an existing
        # 0-1-2 run should generate only the two new ones. An all-or-nothing
        # check regenerates everything, which is slow and silently rewrites
        # data that later steps may already have scored.
        todo = [sd for sd in seeds
                if not (args.skip_existing
                        and load_synthetic("SubSurGen", cohort, seed=sd)
                        is not None)]
        if not todo:
            print(f"  {cohort:10s} cached, skipped")
            continue
        intended = None
        for sd in todo:
            Xs, ts, es, intended = generate(model_for(models, cohort), diffusion,
                                            ep, ep.cohort.n,
                                            seed=C.SEED if sd is None else sd,
                                            guidance=guidance,
                                            device=get_device())
            export("SubSurGen", cohort, ep.cohort.genes, Xs, ts, es, seed=sd)
        if intended is not None:
            np.save(f"{C.SYNTHETIC_DIR}/SubSurGen/{cohort}_intended.npy",
                    intended)
        log.append({"method": "SubSurGen", "cohort": cohort, "family": "Ours",
                    "source": "ours", "is_official": True, "training_mode": mode,
                    "guidance": float(guidance), "guidance_source": g_src,
                    "n": int(len(ts)), "seconds": round(time.time() - t0, 1)})
        extra = "" if len(todo) == len(seeds) else f"  ({len(todo)} new seed(s))"
        print(f"  {cohort:10s} n={len(ts):4d}  {time.time()-t0:5.1f}s{extra}")

    if args.skip_baselines:
        write_json("step3_generation_log.json", log)
        return {"log": log}

    # import-only methods come from scripts/import_external.py, not from here
    methods = args.methods or available_methods()
    named_import_only = [m for m in (args.methods or []) if m in IMPORT_ONLY]
    if named_import_only:
        word = "is" if len(named_import_only) == 1 else "are"
        print(f"\nNOTE: {', '.join(named_import_only)} {word} import-only.")
        print("      Run in its own environment (README_external.md), then:")
        print("          python scripts/import_external.py --runs external_runs")
        methods = [m for m in methods if m not in IMPORT_ONLY]

    # Cohorts imported from an official repository are more authoritative than
    # anything this step can produce, so regenerating them with the in-repo
    # fallback would silently downgrade the benchmark. Skip them unless the user
    # names them explicitly.
    from common import read_json
    imported = set(read_json("external_provenance.json") or {})
    if imported and not args.methods:
        clash = sorted(imported & set(methods))
        if clash:
            print(f"\nskipping {', '.join(clash)}: already imported from the "
                  f"official repo")
            print("  (pass --methods explicitly to regenerate them anyway)")
            methods = [m for m in methods if m not in imported]
    elif imported & set(methods):
        print(f"\nWARNING: {', '.join(sorted(imported & set(methods)))} will be "
              f"OVERWRITTEN with the in-repo reference implementation,")
        print("  replacing data imported from the official repository.")

    for method in methods:
        fn, family = BASELINES[method]
        print(f"\n{method} ({family}, {source_of(method)}):")
        for cohort, ep in episodes.items():
            missing = [sd for sd in seeds
                       if not (args.skip_existing
                               and load_synthetic(method, cohort, seed=sd)
                               is not None)]
            if not missing:
                print(f"  {cohort:10s} cached, skipped")
                continue
            t0 = time.time()
            kwargs = dict(labels=ep.labels, epochs=epochs, genes=ep.cohort.genes)
            if method == "CTABGANPlus":
                kwargs["epochs"] = C.CTABGAN_EPOCHS or epochs
                kwargs["batch_size"] = C.CTABGAN_BATCH_SIZE
                kwargs["num_channels"] = C.CTABGAN_NUM_CHANNELS
            # Fit once per MISSING seed. `rng` seeds the baseline itself, so a
            # different seed gives a genuinely different sample -- which is what
            # the error bars in step 4 measure.
            first = missing[0]
            rng = np.random.default_rng(C.SEED if first is None else first)
            try:
                Xs, ts, es = fn(ep.cohort.X, ep.cohort.time, ep.cohort.event,
                                ep.cohort.n, rng, **kwargs)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {cohort:10s} FAILED {type(exc).__name__}: "
                      f"{str(exc)[:90]}")
                # Record it. Without this a failing baseline vanishes from the
                # log and from every downstream table, making an unavailable
                # method indistinguishable from one that was never attempted --
                # which is how a 12-cohort run silently became a 4-baseline
                # comparison.
                log.append({"method": method, "cohort": cohort,
                            "family": family, "source": source_of(method),
                            "is_official": False, "status": "failed",
                            "n": 0, "seconds": 0.0,
                            "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                failures.setdefault(method, []).append(cohort)
                continue
            export(method, cohort, ep.cohort.genes, Xs, ts, es, seed=first)
            for sd in missing[1:]:
                rng2 = np.random.default_rng(sd)
                try:
                    Xb, tb, eb = fn(ep.cohort.X, ep.cohort.time, ep.cohort.event,
                                    ep.cohort.n, rng2, **kwargs)
                    export(method, cohort, ep.cohort.genes, Xb, tb, eb, seed=sd)
                except Exception:                             # noqa: BLE001
                    pass
            log.append({"method": method, "cohort": cohort, "family": family,
                        "source": last_source(method), "status": "ok",
                        "is_official": last_source(method) in ("pip", "repo"),
                        "n": int(len(ts)),
                        "seconds": round(time.time() - t0, 1)})
            extra = ("" if len(missing) == len(seeds)
                     else f"  ({len(missing)} new seed(s))")
            print(f"  {cohort:10s} n={len(ts):4d}  "
                  f"{time.time()-t0:5.1f}s{extra}")
    if failures:
        print()
        print("=" * 66)
        print("BASELINES THAT PRODUCED NO DATA")
        print("=" * 66)
        for _m, _c in sorted(failures.items()):
            _e = next((r.get("error", "") for r in log
                       if r["method"] == _m and r.get("status") == "failed"), "")
            print(f"  {_m:16s} failed on {len(_c)} cohort(s)")
            if _e:
                print(f"    {_e[:110]}")
        print()
        print("  python scripts/check_baselines.py   -- how to enable each one")

    write_json("step3_generation_log.json", log)
    print(f"\nsynthetic cohorts cached under {C.SYNTHETIC_DIR}")
    return {"log": log}


if __name__ == "__main__":
    main()