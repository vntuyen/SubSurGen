"""SubSurGen -- master pipeline runner.

Runs the whole experiment end to end, or any subset of steps.

Examples
--------
    python run_all.py                      # full pipeline (~6-7 hours on CPU)
    python run_all.py --quick              # fast smoke test (~3 min)
    python run_all.py --steps 0 1 2 3 4    # prepare, train, tune, generate, evaluate
    python run_all.py --steps 6            # rebuild tables/figures only
    python run_all.py --skip 5             # skip the slow ablation step
    python run_all.py --steps 3 --seeds 0 1 2   # regenerate step 3, 3 seeds only

Steps
-----
    0  prepare data  load cohorts, Cox feature selection, SNF, choose K, episodes
    1  train         per-cohort training of the generator
    2  tune          sweep guidance, select the operating point, persist it
    3  generate      synthetic cohorts from SubSurGen and every registered baseline
                      (called with --seeds 0 1 2 3 4 by default, for error
                      bars in step 4; --quick reduces this to a single seed)
    4  evaluate      the four-metric benchmark
    5  ablation      component ablations
    6  report        tables (RESULTS.md) + figures

Steps 3 and 5 read the guidance chosen in step 2 from results/tuning.json, so no
hyperparameter has to be edited by hand between steps.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _silence_joblib_core_probe() -> None:
    """.."""
    try:
        from joblib.externals.loky.backend import context as _ctx
        n = os.cpu_count() or 1
        if getattr(_ctx, "physical_cores_cache", None) is None:
            _ctx.physical_cores_cache = n
        if hasattr(_ctx, "_count_physical_cores_win32"):
            _ctx._count_physical_cores_win32 = lambda _n=n: _n
    except Exception:                                        # noqa: BLE001
        pass


_silence_joblib_core_probe()

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import config as C  # noqa: E402

STEPS = {
    0: ("prepare data", "step0_prepare_data"),
    1: ("train",        "step1_train"),
    2: ("tune",         "step2_tune"),
    3: ("generate",     "step3_generate"),
    4: ("evaluate",     "step4_evaluate"),
    5: ("ablation",     "step5_ablation"),
    6: ("report",       "step6_report"),
}

# step3_generate.py already accepts a `--seeds` flag (nargs="+") and loops
# over all of them itself in a single call -- generating and caching each
# cohort's output per seed (results/synthetic/<Method>/<Cohort>__s<seed>.csv)
# so step 4 can report mean +/- sd. run_all.py just forwards the seed list.
GENERATE_STEP = 3
DEFAULT_GENERATE_SEEDS = [0, 1, 2, 3, 4]


def run_setup_external() -> bool:
    """Fetch official baseline repos (currently just CTAB-GAN+) before
    generation runs, so step 3 gets the 'repo' source instead of falling back
    to an in-repo reimplementation. setup_external.py skips repos that are
    already cloned, so this is a cheap no-op on every run after the first.

    """
    print("fetching external baseline repositories (setup_external.py)...")
    saved_argv = sys.argv
    sys.argv = ["setup_external.py"]
    try:
        mod = importlib.import_module("setup_external")
        importlib.reload(mod)
        rc = mod.main()
        if rc != 0:
            print("[setup_external] one or more repos failed to fetch; "
                  "affected baselines will fall back to their reference "
                  "implementation (source 'ref') instead of 'repo'.")
        return rc == 0
    except Exception:                                          # noqa: BLE001
        print("[setup_external] FAILED to run at all -- continuing anyway; "
              "affected baselines will use their reference implementation.")
        traceback.print_exc()
        return False
    finally:
        sys.argv = saved_argv


def check_environment() -> bool:
    ok = True
    if sys.version_info < (3, 9):
        print(f"ERROR: Python {sys.version_info.major}.{sys.version_info.minor} "
              f"detected; 3.9+ required (3.10 recommended)")
        ok = False
    for pkg in ["numpy", "scipy", "sklearn", "pandas", "torch"]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"ERROR: missing required package '{pkg}' "
                  f"-- run: pip install -r requirements.txt")
            ok = False
    try:
        importlib.import_module("matplotlib")
    except ImportError:
        print("NOTE: matplotlib not installed; step 6 will skip figures")

    if not os.path.isdir(C.DATA_DIR):
        print(f"ERROR: data directory not found: {C.DATA_DIR}")
        ok = False
    else:
        csvs = [f for f in os.listdir(C.DATA_DIR) if f.endswith("_driver_genes.csv")]
        if not csvs:
            print(f"ERROR: no *_driver_genes.csv files in {C.DATA_DIR}")
            ok = False
        else:
            print(f"data: {len(csvs)} cohort file(s) in {C.DATA_DIR}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the SubSurGen pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   choices=sorted(STEPS), help="which steps to run")
    p.add_argument("--skip", type=int, nargs="+", default=[],
                   choices=sorted(STEPS), help="steps to skip")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke test with reduced steps")
    p.add_argument("--train-steps", type=int, default=None,
                   help="override the number of training steps")
    p.add_argument("--guidance", type=float, default=None,
                   help="override the sampling guidance strength")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="sampling seeds forwarded to step 3 (generate) as "
                        "--seeds; several seeds give step 4 error bars "
                        "(default: 0 1 2 3 4, or just 0 under --quick)")
    p.add_argument("--check", action="store_true",
                   help="verify the environment and exit")
    p.add_argument("--skip-setup-external", action="store_true",
                   help="don't fetch external baseline repos before generating")
    args = p.parse_args()

    print("=" * 74)
    print("SubSurGen pipeline")
    print(f"python {sys.version.split()[0]} | root {ROOT}")
    print("=" * 74)

    if not check_environment():
        return 1
    if args.check:
        print("\nenvironment OK")
        return 0

    if args.quick:
        C.apply_quick_mode()
        print("[quick mode] reduced training and diffusion steps")
        print("[quick mode] WARNING: this is a PLUMBING TEST ONLY. With "
              f"{C.TRAIN_STEPS} training steps the")
        print("[quick mode] generator is badly undertrained and its metrics are "
              "NOT comparable")
        print("[quick mode] to the baselines. Never draw conclusions from a "
              "--quick run.")
    if args.train_steps:
        C.TRAIN_STEPS = args.train_steps
        print(f"[override] TRAIN_STEPS={C.TRAIN_STEPS}")
    if args.guidance:
        C.GUIDANCE = args.guidance
        print(f"[override] GUIDANCE={C.GUIDANCE}")
    if args.seeds is None:
        args.seeds = [0] if args.quick else DEFAULT_GENERATE_SEEDS

    todo = [s for s in (args.steps if args.steps is not None else sorted(STEPS))
            if s not in args.skip]
    print(f"\nrunning steps: {todo}")
    if GENERATE_STEP in todo:
        print(f"step {GENERATE_STEP} (generate) will use --seeds {args.seeds}")
    print()

    if GENERATE_STEP in todo and not args.skip_setup_external:
        run_setup_external()
        print()

    # step scripts parse their own args; hide ours from them
    saved_argv = sys.argv
    timings, failed = {}, []
    t_all = time.time()

    for s in todo:
        label, module_name = STEPS[s]
        argv = [f"{module_name}.py"]
        if s == GENERATE_STEP:
            argv += ["--seeds", *(str(sd) for sd in args.seeds)]
        sys.argv = argv
        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)
            mod.main()
            timings[s] = time.time() - t0
            print(f"\n[step {s}: {label}] done in {timings[s]:.0f}s")
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(f"\n[step {s}: {label}] FAILED: {exc}")
                failed.append(s)
                break
            timings[s] = time.time() - t0
        except Exception:                                    # noqa: BLE001
            print(f"\n[step {s}: {label}] FAILED")
            traceback.print_exc()
            failed.append(s)
            break
        finally:
            sys.argv = saved_argv

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for s, sec in timings.items():
        print(f"  step {s} ({STEPS[s][0]:<17s}) {sec:7.0f}s")
    print(f"  {'total':<26s} {time.time() - t_all:7.0f}s")
    if failed:
        print(f"\nfailed steps: {failed}")
        return 1
    print(f"\nresults in {C.RESULTS_DIR}")
    print(f"  RESULTS.md, subsurgen_all_results.json, figures/, synthetic/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
