"""Central configuration for the SubSurGen pipeline.

Every step script imports from here, so changing a hyperparameter in one place
changes it everywhere. Paths are resolved relative to this file, so the pipeline
runs correctly from any working directory.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------- paths --
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SUBSURGEN_DATA", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SYNTHETIC_DIR = os.path.join(RESULTS_DIR, "synthetic")

EPISODES_PKL = os.path.join(RESULTS_DIR, "episodes.pkl")
MODEL_PT = os.path.join(RESULTS_DIR, "model.pt")          # episodic mode
MODELS_DIR = os.path.join(RESULTS_DIR, "models")          # per-cohort mode
MODE_MARKER = os.path.join(RESULTS_DIR, "training_mode.txt")

for _d in (RESULTS_DIR, FIGURES_DIR, SYNTHETIC_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- experiment --
SEED = 0
# Number of molecular subtypes. Either a fixed integer, or "auto" to select K
# per cohort from the fused network alone (cluster stability, with the Laplacian
# eigengap as a cross-check). Selection never uses survival, so it cannot inflate
# the subtype-survival metric. A fixed K keeps cohorts directly comparable; "auto"
# is the right choice when cohorts plausibly differ in subtype count.
K_SUBTYPES = 3
K_RANGE = (2, 6)               # search range when K_SUBTYPES == "auto"
K_BOOTSTRAPS = 20              # subsamples per K for the stability criterion

# ---------------------------------------------------------- feature selection --
# Cox screening was designed for raw transcriptomes (~10k+ genes) where
# dimensionality reduction is unavoidable.
# Three modes:
#   "none"           SNF and generation both use the full input panel. Cleanest
#                    feature-space match, but on a curated panel the resulting
#                    subtypes are often NOT prognostic.
#   "subtypes_only"  (DEFAULT) Cox screening selects the genes that DEFINE the
#                    subtypes, while the generator still emits every input gene.
#                    Gives prognostic subtypes AND an exact feature-space match.
#   "full"           Cox screening reduces the panel for both subtype discovery
#                    and generation. Appropriate only for raw high-dimensional
#                    input; on a curated panel it shrinks the output.
FEATURE_SELECTION = "subtypes_only"
USE_FEATURE_SELECTION = FEATURE_SELECTION != "none"   # backwards compatibility
COX_CUTOFF = 0.10              # univariate Cox p-value threshold (when enabled)
MIN_KEEP = 20                  # minimum genes retained          (when enabled)

# SNF
SNF_K = 20                     # nearest neighbours in the similarity network
SNF_T = 20                     # cross-diffusion iterations
SNF_VIEWS = 3                  # co-expression modules -> views for fusion
COND_DIM = 16                  # spectral embedding dimension used as conditioning

# diffusion model
D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
DIFFUSION_STEPS = 200

# ------------------------------------------------------------ training mode --
# "per_cohort" : fit an independent SubSurGen on each cohort. Every baseline is
#                fitted per cohort, so this is the like-for-like comparison and
#                is the DEFAULT. Each synthetic cohort depends only on its own
#                real cohort -- no cross-cohort information leaks in.
# "episodic"   : one generator trained across all cohorts. Tests the transfer /
#                foundation-model claim, but gives SubSurGen strictly more data
#                than the baselines, so it must not be presented as a
#                head-to-head result.
# ------------------------------------------------------------------ device --
# "auto" uses CUDA when a GPU is visible, else CPU. Override per job with the
# SUBSURGEN_DEVICE environment variable ("cpu" or "cuda").
#

DEVICE = os.environ.get("SUBSURGEN_DEVICE", "auto")

TRAINING_MODE = "per_cohort"

# training
TRAIN_STEPS = 2500
BATCH_SIZE = 64
LR = 2e-3
LAMBDA_COX = 0.2               # weight on the auxiliary Cox partial-likelihood
FEATURE_DROPOUT = 0.30         # simulates heterogeneous feature spaces
COND_DROPOUT = 0.15            # enables classifier-free guidance

# sampling
# Guidance is TUNED AUTOMATICALLY by step 2, which writes the selected value to
# results/tuning.json; steps 3 and 5 read it from there. The value below is
# only the fallback used when step 2 has not been run (or --no-autotune is
# passed).
AUTO_TUNE_GUIDANCE = True
GUIDANCE = 1.0                 # default operating point (see the guidance sweep)
GUIDANCE_SWEEP = [1.0, 2.0, 3.0, 4.0, 6.0]

# Weighting used when the tuner ranks candidate guidance values. Equal weights
# treat all five metrics alike; raise a weight to prioritise a metric the paper
# leads on. Set DISPERSION_PENALTY > 0 to also penalise |dispersion - 1|, which
# discourages settings that win subtype metrics by over- or under-dispersing.
TUNE_WEIGHTS = {"wasserstein": 1.0, "centroid_corr": 1.0,
                "subtype_km_div": 1.0, "km_div": 1.0, "tstr_cindex": 1.0}
DISPERSION_PENALTY = 1.0

# evaluation
N_REPEATS = 5                  # train/test repeats for downstream C-index
TEST_FRACTION = 0.30
# A cohort's subtypes are called prognostic when the real log-rank test clears
# this threshold. Metric [3] (subtype-specific KM divergence) asks whether
# subtype-specific prognosis is preserved -- in a cohort whose real subtypes do
# not differ in survival there is nothing to preserve, so those cohorts are
# reported separately rather than averaged in silently.
PROGNOSTIC_ALPHA = 0.05

IBS_GRID_POINTS = 40           # time points for the integrated Brier score
IBS_LOWER_PCT = 5              # grid spans these percentiles of observed time
IBS_UPPER_PCT = 80             # (upper bound avoids the unstable IPCW tail)

# baselines
BASELINE_EPOCHS = 150          # training epochs for the learned baselines

# CTAB-GAN+ is by far the slowest baseline (a WGAN-GP CNN with a gradient
# penalty). Its stock settings waste most of that time on small cohorts; see
# baselines.ctabgan_plus for the measurements behind these defaults.
CTABGAN_EPOCHS = 20            # None -> use BASELINE_EPOCHS; stock is 150
CTABGAN_BATCH_SIZE = None      # None -> scale to the cohort (recommended)
CTABGAN_NUM_CHANNELS = 16      # stock is 64; each halving is ~4x cheaper per conv

# --- synthcity plugins -----------------------------------------------------
# synthcity resamples until a generated batch matches the training schema, up to
# `sampling_patience` attempts (its default is 500). Each attempt is a full
# generation pass -- for a diffusion plugin, the entire reverse process -- so a
# plugin whose output sits marginally outside the observed ranges can spend
# twenty minutes silently retrying. synthcity returns its best attempt when the
# budget is exhausted, so a smaller number bounds the cost without failing.
SYNTHCITY_SAMPLING_PATIENCE = 25

# SurvivalPipeline (SurVAE, SurvivalGAN, SurvivalCTGAN, SurvivalDDPM) fits an
# uncensoring model before the generator. synthcity's default,
# "survival_function_regression", is an XGBoost survival ensemble across time
# horizons -- minutes to hours per cohort on CPU. "cox_ph" fills the same role
# in seconds. Use "survival_function_regression" to reproduce synthcity's
# default, or "none" to skip uncensoring.
SYNTHCITY_UNCENSORING_MODEL = "cox_ph"



def apply_quick_mode() -> None:
    """Shrink the workload for a fast smoke test (`--quick`)."""
    global TRAIN_STEPS, DIFFUSION_STEPS, N_REPEATS, GUIDANCE_SWEEP
    global BASELINE_EPOCHS
    TRAIN_STEPS = 200
    DIFFUSION_STEPS = 50
    N_REPEATS = 2
    GUIDANCE_SWEEP = [1.0, 3.0]
    BASELINE_EPOCHS = 40
