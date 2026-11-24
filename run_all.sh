#!/usr/bin/env bash
#
# SubSurGen -- full experiment on macOS.
#
#     bash run_macos.sh                 # setup (if needed) + steps 0-6
#     bash run_macos.sh --setup-only    # create/refresh .venv, then stop
#     bash run_macos.sh --quick         # ~5 min plumbing test, NOT results
#     bash run_macos.sh --skip-ablation # skip step 5 (the slowest step)
#     bash run_macos.sh --steps 4 6     # rerun specific steps only
#     bash run_macos.sh --rerun-012     # force prepare/train/tune again
#
# Environment overrides:
#     TRAIN_STEPS=4000 bash run_macos.sh
#     SEEDS="0 1 2 3 4" bash run_macos.sh
#     NCPUS=4 bash run_macos.sh
#
# The external baselines (TabSyn, TabDiff, SAVAE, SurvDiff) are NOT run here:
# their dependency pins conflict with each other and with this project, so each
# needs its own environment. TabDDPM and SurvivalDDPM are run through Synthcity
# in this environment. See README_external.md for the import-only baselines, then
#     python scripts/import_external.py --runs external_runs
# and rerun steps 4 and 6.

set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="${SUBSURGEN_VENV:-$ROOT/.venv}"

QUICK=""; SETUP_ONLY=0; SKIP_ABLATION=0; RERUN_012=0; STEPS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)          QUICK="--quick"; shift ;;
        --setup-only)     SETUP_ONLY=1; shift ;;
        --skip-ablation)  SKIP_ABLATION=1; shift ;;
        --rerun-012)      RERUN_012=1; shift ;;
        --steps)          shift; while [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do
                              STEPS="$STEPS $1"; shift; done ;;
        -h|--help)        sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; echo "try --help"; exit 1 ;;
    esac
done

TRAIN_STEPS="${TRAIN_STEPS:-2500}"
SEEDS="${SEEDS:-0 1 2}"

# macOS reports performance+efficiency cores; use a modest CPU count for the
# project generally, but keep Synthcity/PyTorch native thread pools conservative.
if [ -z "${NCPUS:-}" ]; then
    NCPUS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null \
             || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)"
fi

# SurvivalDDPM = Synthcity SurvivalPipeline("ddpm"). On macOS, PyTorch + BLAS +
# XGBoost/OpenMP can crash natively when several thread pools are active at once.
# The fixed baselines.py also enforces this locally; exporting here ensures the
# same settings apply before NumPy/PyTorch/Synthcity are imported.
SYNTHCITY_THREADS="${SYNTHCITY_THREADS:-1}"
export NCPUS
export OMP_NUM_THREADS="$SYNTHCITY_THREADS"
export MKL_NUM_THREADS="$SYNTHCITY_THREADS"
export OPENBLAS_NUM_THREADS="$SYNTHCITY_THREADS"
export NUMEXPR_NUM_THREADS="$SYNTHCITY_THREADS"
export VECLIB_MAXIMUM_THREADS="$SYNTHCITY_THREADS"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Safe defaults for Synthcity's survival pipeline. Override them explicitly if
# required, e.g. SUBSURGEN_SAMPLING_PATIENCE=50 bash run_macos.sh
export SUBSURGEN_SAMPLING_PATIENCE="${SUBSURGEN_SAMPLING_PATIENCE:-25}"
export SUBSURGEN_UNCENSORING_MODEL="${SUBSURGEN_UNCENSORING_MODEL:-cox_ph}"

banner () { printf '\n%s\n%s\n' "==================================================" "$1"; }
fail ()   { echo; echo "FAILED: $1"; echo "see the output above"; exit 1; }

banner "SubSurGen on macOS"
echo "root        : $ROOT"
echo "cores used  : $NCPUS"
echo "train steps : $TRAIN_STEPS"
echo "seeds       : $SEEDS"
echo "synthcity threads : $SYNTHCITY_THREADS"
echo "synthcity sampling patience : $SUBSURGEN_SAMPLING_PATIENCE"
echo "synthcity uncensoring model : $SUBSURGEN_UNCENSORING_MODEL"
[ -n "$QUICK" ] && echo "MODE        : QUICK -- plumbing test only, results are not comparable"

# ------------------------------------------------------------------- python --
PY=""
for c in python3.10 python3.11 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && fail "no python3 found. Install it: brew install python@3.10"
# Only used to CREATE the venv if absent; once activated the venv's own
# interpreter takes over, so a python3.10 from an unrelated venv on PATH here
# is harmless. The prefix printed after activation is the one that matters.
echo "bootstrap py: $("$PY" --version 2>&1) ($(command -v "$PY"))"

# --------------------------------------------------------------------- venv --
if [ ! -d "$VENV" ]; then
    banner "creating virtual environment"
    "$PY" -m venv "$VENV" || fail "could not create $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate" || fail "could not activate $VENV"
echo "active venv : $(python -c 'import sys; print(sys.prefix)')"
echo "active py   : $(python -c 'import sys; print(sys.executable)')"

NEED_INSTALL=0
python -c "import numpy, scipy, sklearn, pandas, torch" 2>/dev/null || NEED_INSTALL=1
if [ "$NEED_INSTALL" = "1" ] || [ "$SETUP_ONLY" = "1" ]; then
    banner "installing dependencies"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt || fail "requirements install"
    # Optional official baselines. GaussianCopula/CTGAN/TVAE come from SDV.
    # Synthcity provides TabDDPM and the survival-family adapters used here.
    # baselines.py performs a functional compatibility probe instead of assuming
    # one particular torch version.
    python -m pip install "sdv>=1.17" \
        || echo "NOTE: sdv unavailable -- those baselines will run as 'ref'"
    python -m pip install synthcity \
        || echo "NOTE: synthcity unavailable -- Synthcity baselines will fall back/skip"
fi

mkdir -p data results logs

banner "environment check"
python scripts/check_env.py || echo "NOTE: some baselines unavailable (see above)"

if [ "$SETUP_ONLY" = "1" ]; then
    echo
    echo "setup complete. Put cohort CSVs in data/ then: bash run_macos.sh"
    exit 0
fi

# --------------------------------------------------------------------- data --
COHORTS=$(ls -1 data/*_driver_genes.csv 2>/dev/null | wc -l | tr -d ' ')
if [ "$COHORTS" = "0" ]; then
    fail "no cohort files in data/. Expected <COHORT>_driver_genes.csv (see data/README.txt)"
fi
echo "cohorts     : $COHORTS"
python run_all.py --check || fail "pipeline pre-flight"

# ----------------------------------------------------------------- pipeline --
START=$(date +%s)

if [ -n "$STEPS" ]; then
    banner "running steps:$STEPS"
    # shellcheck disable=SC2086
    python run_all.py --steps $STEPS $QUICK --train-steps "$TRAIN_STEPS" \
        || fail "steps$STEPS"
else
    # ------------------------------------------------------- steps 0-2 resume
    # A normal rerun should resume from the trained SubSurGen checkpoint and
    # tuned guidance instead of retraining. Verify that the saved training state
    # is loadable; only rerun 0-2 when it is missing/corrupt, or when explicitly
    # requested with --rerun-012.
    HAVE_TRAINED_STATE=0
    if [ "$RERUN_012" = "0" ]; then
        if python - <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, "scripts")
from common import load_trained_model
models, diffusion, episodes, extra = load_trained_model()
assert models is not None
assert diffusion is not None
assert episodes
PY
        then
            HAVE_TRAINED_STATE=1
        fi
    fi

    if [ "$HAVE_TRAINED_STATE" = "1" ]; then
        # Check whether auto-tuned guidance also exists. If training is cached
        # but tuning is missing, rerun only step 2 -- never steps 0 or 1.
        NEED_TUNE=0
        if ! python - <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, "scripts")
import config as C
from common import resolve_guidance
guidance, source = resolve_guidance(None)
if getattr(C, "AUTO_TUNE_GUIDANCE", False):
    assert source != "config.py fallback"
PY
        then
            NEED_TUNE=1
        fi

        if [ "$NEED_TUNE" = "1" ]; then
            banner "steps 0-1: existing trained state found -- skipped"
            banner "step 2: tuned guidance missing -- rerunning tune only"
            python run_all.py --steps 2 $QUICK --train-steps "$TRAIN_STEPS" \
                || fail "step 2"
        else
            banner "steps 0-2: existing prepare/train/tune state found -- skipped"
            echo "Use --rerun-012 to force prepare/train/tune again."
        fi
    else
        banner "steps 0-2: no valid trained state found -- running"
        python run_all.py --steps 0 $QUICK --train-steps "$TRAIN_STEPS" \
            || fail "steps 0-2"
    fi

    # ------------------------------------------------------------- step 3
    banner "step 3a: generate SubSurGen + stable baselines"

    # Do NOT run SurvivalDDPM in the same Python process as every other method.
    # Synthcity can segfault in native code on macOS; SIGSEGV bypasses Python
    # exception handling and would otherwise abort all of Step 3.
    #
    # Explicit method list keeps the native-crash-prone SurvivalDDPM isolated.
    STABLE_METHODS="GaussianCopula CTGAN TVAE CTABGANPlus TabDDPM SurVAE SurvivalGAN SurvivalCTGAN"

    # shellcheck disable=SC2086
    python -X faulthandler scripts/step3_generate.py $QUICK \
        --methods $STABLE_METHODS --seeds $SEEDS --skip-existing \
        || fail "step 3 stable baselines"

    banner "step 3b: SurvivalDDPM (isolated Synthcity process)"
    echo "Running SurvivalDDPM separately so a native Synthcity SIGSEGV cannot"
    echo "discard SubSurGen or the other completed baseline outputs."

    # Run the risky native component in a child process. Existing SurvivalDDPM
    # cohorts/seeds are skipped. If Synthcity still segfaults on this macOS
    # environment, preserve all completed outputs and continue to evaluation;
    # the missing SurvivalDDPM rows remain explicit rather than killing the run.
    python -X faulthandler scripts/step3_generate.py \
        --methods SurvivalDDPM --seeds $SEEDS --skip-existing
    SURVIVAL_DDPM_RC=$?

    if [ "$SURVIVAL_DDPM_RC" -ne 0 ]; then
        echo
        echo "WARNING: SurvivalDDPM exited with code $SURVIVAL_DDPM_RC."
        if [ "$SURVIVAL_DDPM_RC" -eq 139 ]; then
            echo "         This is SIGSEGV (11) from Synthcity/native libraries."
        fi
        echo "         Other synthetic outputs are preserved; continuing."
        echo "         SurvivalDDPM should be run in a dedicated compatible"
        echo "         Synthcity environment if official results are required."
    fi

    banner "step 4: six-metric benchmark"
    python scripts/step4_evaluate.py $QUICK || fail "step 4"

    if [ "$SKIP_ABLATION" = "0" ]; then
        banner "step 5: ablations (slowest step; --skip-ablation to omit)"
        python scripts/step5_ablation.py $QUICK --mode per_cohort \
            || echo "NOTE: ablation failed; continuing to the report"
    fi

    banner "step 6: report"
    python scripts/step6_report.py || fail "step 6"
fi

ELAPSED=$(( $(date +%s) - START ))

banner "done in $((ELAPSED/60))m $((ELAPSED%60))s"
echo "results: $ROOT/results"
ls -1 results/RESULTS.md results/*.json 2>/dev/null | sed 's|^|  |'
[ -d results/figures ] && ls -1 results/figures/*.png 2>/dev/null | sed 's|^|  |'

if [ -n "$QUICK" ]; then
    echo
    echo "QUICK MODE: these numbers are not comparable to the baselines."
    echo "On identical data, subtype-centroid correlation is 0.260 at 200"
    echo "training steps and 0.717 at 2500. Rerun without --quick."
fi

echo
echo "Next: external baselines run in their own environments."
echo "  python scripts/export_for_external.py --out external_runs"
echo "  # see README_external.md, then"
echo "  python scripts/import_external.py --runs external_runs"
echo "  python run_all.py --steps 4 6"
