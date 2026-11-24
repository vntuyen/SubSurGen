"""Helpers shared across the step scripts: IO, model construction, reporting."""
from __future__ import annotations

import json
import os
import pickle
import sys


def _allocated_cpus() -> int:
    for var in ("NCPUS", "PBS_NCPUS", "SLURM_CPUS_PER_TASK",
                "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return os.cpu_count() or 1


_NCPU = _allocated_cpus()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(_NCPU))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_NCPU))


def _silence_joblib_core_probe() -> None:
    """."""
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

import numpy as np
import torch

torch.set_num_threads(_NCPU)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from subsurgen.metrics import (at_risk_balance, correlation_error, km_divergence,  # noqa: E402
                             logrank_test, mmd_rbf, optimism, short_sightedness)
from subsurgen.model import Diffusion, SubSurGenDenoiser  # noqa: E402
from subsurgen.snf import run_snf  # noqa: E402


def banner(msg: str) -> None:
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74, flush=True)


def set_seed(seed: int = C.SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> str:
    """Resolve config.DEVICE. 'auto' selects CUDA when a GPU is visible."""
    want = str(getattr(C, "DEVICE", "auto")).lower()
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: DEVICE=cuda but no GPU is visible; falling back to CPU")
        return "cpu"
    return want


def describe_device() -> str:
    d = get_device()
    if d.startswith("cuda") and torch.cuda.is_available():
        return f"{d} ({torch.cuda.get_device_name(0)})"
    return f"{d} ({_NCPU} thread(s))"


def make_model(vocab_size: int, seed: int = C.SEED) -> SubSurGenDenoiser:
    set_seed(seed)
    m = SubSurGenDenoiser(vocab_size, d_model=C.D_MODEL, n_layers=C.N_LAYERS,
                          n_heads=C.N_HEADS, cond_dim=C.COND_DIM)
    return m.to(get_device())


def make_diffusion() -> Diffusion:
    return Diffusion(C.DIFFUSION_STEPS, device=get_device())


def save_episodes(episodes: dict, vocab: dict) -> None:
    with open(C.EPISODES_PKL, "wb") as f:
        pickle.dump({"episodes": episodes, "vocab": vocab}, f)


def load_episodes() -> tuple:
    if not os.path.exists(C.EPISODES_PKL):
        raise SystemExit(
            "episodes.pkl not found -- run scripts/step0_prepare_data.py first"
        )
    with open(C.EPISODES_PKL, "rb") as f:
        d = pickle.load(f)
    return d["episodes"], d["vocab"]


def training_mode() -> str:
    """Which mode step 1 used: 'per_cohort' (default) or 'episodic'."""
    if os.path.exists(C.MODE_MARKER):
        return open(C.MODE_MARKER).read().strip()
    return "episodic" if os.path.exists(C.MODEL_PT) else C.TRAINING_MODE


def load_trained_model():
    """Return (models, diffusion, episodes, vocab).

    `models` is a single model in episodic mode.
    """
    episodes, vocab = load_episodes()
    mode = training_mode()

    if mode == "per_cohort":
        if not os.path.isdir(C.MODELS_DIR):
            raise SystemExit("per-cohort models not found -- "
                             "run scripts/step1_train.py first")
        models = {}
        for name in episodes:
            path = os.path.join(C.MODELS_DIR, f"{name}.pt")
            if not os.path.exists(path):
                raise SystemExit(f"missing model for cohort {name} -- rerun step 1")
            m = make_model(len(vocab))
            m.load_state_dict(torch.load(path, map_location=get_device()))
            m.eval()
            models[name] = m.to(get_device())
        return models, make_diffusion(), episodes, vocab

    if not os.path.exists(C.MODEL_PT):
        raise SystemExit("model.pt not found -- run scripts/step1_train.py first")
    model = make_model(len(vocab))
    model.load_state_dict(torch.load(C.MODEL_PT, map_location=get_device()))
    model.eval()
    return model.to(get_device()), make_diffusion(), episodes, vocab


def model_for(models, cohort: str):
    """Resolve the right model whichever training mode was used."""
    return models[cohort] if isinstance(models, dict) else models


def _syn_name(cohort: str, seed=None) -> str:
    return f"{cohort}.npz" if seed is None else f"{cohort}__s{seed}.npz"


def save_synthetic(method: str, cohort: str, X, t, e, genes=None, seed=None) -> str:
    """Cache one synthetic cohort under results/synthetic/<method>/<cohort>.npz

    With `seed`, the file is suffixed `__s<seed>` so several sampling seeds can be
    cached side by side and aggregated with error bars in step 3.
    """
    d = os.path.join(C.SYNTHETIC_DIR, method)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, _syn_name(cohort, seed))
    np.savez(path, X=X, time=t, event=e,
             genes=np.array(list(genes) if genes is not None else []))
    return path


def list_synthetic_seeds(method: str, cohort: str) -> list:
    """Seeds cached for this method/cohort. [None] means a single unsuffixed file."""
    d = os.path.join(C.SYNTHETIC_DIR, method)
    if not os.path.isdir(d):
        return []
    seeds = []
    for f in os.listdir(d):
        if not f.endswith(".npz"):
            continue
        if f == f"{cohort}.npz":
            seeds.append(None)
        elif f.startswith(f"{cohort}__s"):
            try:
                seeds.append(int(f[len(cohort) + 3:-4]))
            except ValueError:
                pass
    return sorted(seeds, key=lambda x: (x is not None, x))


def load_synthetic(method: str, cohort: str, seed=None):
    """Load a cached synthetic cohort, or None if it was never generated."""
    path = os.path.join(C.SYNTHETIC_DIR, method, _syn_name(cohort, seed))
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return d["X"], d["time"], d["event"]


def list_synthetic_methods() -> list:
    if not os.path.isdir(C.SYNTHETIC_DIR):
        return []
    return sorted(m for m in os.listdir(C.SYNTHETIC_DIR)
                  if os.path.isdir(os.path.join(C.SYNTHETIC_DIR, m)))


TUNING_JSON = "tuning.json"


def save_tuning(d: dict) -> None:
    """Persist auto-selected hyperparameters chosen by the tuning step."""
    write_json(TUNING_JSON, d)


def load_tuning() -> dict:
    return read_json(TUNING_JSON) or {}


def resolve_guidance(override=None) -> tuple:
    """Guidance to use, and where it came from.

    Precedence: explicit --guidance > results/tuning.json (written by the tuning
    step) > config.GUIDANCE. Returning the source lets every script state which
    value it used, so a table can never be silently produced at an untuned
    setting.
    """
    if override is not None:
        return float(override), "command line"
    t = load_tuning()
    if C.AUTO_TUNE_GUIDANCE and "guidance" in t:
        return float(t["guidance"]), f"auto-tuned (step 2, {t.get('n_candidates', '?')} candidates)"
    return float(C.GUIDANCE), "config.py fallback"


def write_json(name: str, obj) -> str:
    path = os.path.join(C.RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  -> {path}")
    return path


def read_json(name: str, default=None):
    path = os.path.join(C.RESULTS_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def fidelity_report(episode, Xs, ts, es, intended_labels) -> dict:
    """All fidelity + subtype-consistency metrics for one synthetic sample.

    The subtype-consistency check: re-run SNF and clustering on the SYNTHETIC data
    with no knowledge of how it was generated, then compare recovered labels
    against the labels used for conditioning.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    Xr = episode.cohort.X
    tr, er = episode.cohort.time, episode.cohort.event

    recovered = run_snf(Xs, k_subtypes=C.K_SUBTYPES, K=C.SNF_K, t=C.SNF_T,
                        n_views=C.SNF_VIEWS, emb_dim=C.COND_DIM,
                        seed=C.SEED)["labels"]
    _, p_syn, _ = logrank_test(ts, es, recovered)

    return {
        "mmd": mmd_rbf(Xr, Xs),
        "corr_err": correlation_error(Xr, Xs),
        "km_div": km_divergence(tr, er, ts, es),
        "optimism": optimism(tr, er, ts, es),
        "short_sight": short_sightedness(tr, ts),
        "at_risk_bal": at_risk_balance(tr, ts),
        "event_rate_syn": float(es.mean()),
        "event_rate_real": float(er.mean()),
        "ari": float(adjusted_rand_score(intended_labels, recovered)),
        "nmi": float(normalized_mutual_info_score(intended_labels, recovered)),
        "logrank_p_syn": float(p_syn),
    }


def parse_common_args(parser):
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke test with reduced steps")
    args = parser.parse_args()
    if args.quick:
        C.apply_quick_mode()
        print("[quick mode] reduced training/diffusion steps")
    return args
