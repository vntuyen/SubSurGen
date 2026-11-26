"""Baseline generators for the SubSurGen benchmark.

Eight baselines. Every generator exposes the same interface:

    fn(X, time, event, n, rng, labels=None, **kw) -> (X_syn, time_syn, event_syn)

Provenance is tracked explicitly. Each entry records how it is obtained:

    "pip"    installed from PyPI, official implementation called directly
    "repo"   official GitHub repository, cloned by scripts/setup_external.py



    | Family        | Baseline       | Source                                |
    |---------------|----------------|----------------------------------------|
    | Statistical   | GaussianCopula | SDV              (pip install sdv)      |
    | GAN           | CTGAN          | SDV              (pip install sdv)      |
    | VAE           | TVAE           | SDV              (pip install sdv)      |
    | GAN           | CTABGANPlus    | Team-TUD/CTAB-GAN-Plus  (official repo) |
    | Tab. diffusion| TabDDPM        | synthcity        (pip install synthcity)|
    | Survival VAE  | SurVAE         | synthcity        (pip install synthcity)|
    | Survival GAN  | SurvivalGAN    | synthcity        (pip install synthcity)|
    | Survival GAN  | SurvivalCTGAN  | synthcity        (pip install synthcity)|
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

EXTERNAL_DIR = os.environ.get(
    "SUBSURGEN_EXTERNAL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "external"),
)


# ----------------------------------------------------------- availability --
def _install_trapz_shim() -> bool:
    """Restore `scipy.integrate.trapz` and `numpy.trapz` for older synthcity.

    `trapz` was deprecated in SciPy 1.12 and REMOVED in 1.14; NumPy removed its
    own `trapz` in 2.0. synthcity <= 0.2.4 imports the SciPy name unguarded:

        ImportError: cannot import name 'trapz' from 'scipy.integrate'

    Later versions fall back to `numpy.trapz`, which fails identically on
    NumPy >= 2. Both were pure aliases for `trapezoid`, so re-adding them is
    exact -- no numerical difference, and no need to downgrade SciPy or NumPy,
    which would drag other packages backwards.

    Must run BEFORE synthcity is imported. Returns True if anything was patched.
    """
    patched = False
    try:
        import scipy.integrate as _si
        if not hasattr(_si, "trapz") and hasattr(_si, "trapezoid"):
            _si.trapz = _si.trapezoid
            patched = True
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import numpy as _np
        if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
            _np.trapz = _np.trapezoid
            patched = True
    except Exception:                                        # noqa: BLE001
        pass
    return patched


TRAPZ_SHIM_APPLIED = _install_trapz_shim()


def _have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def _have_repo(folder: str) -> bool:
    return os.path.isdir(os.path.join(EXTERNAL_DIR, folder))


HAVE_SDV = _have_module("sdv")
HAVE_SYNTHCITY = _have_module("synthcity")

REPOS = {
    "CTABGANPlus": ("ctabganplus", "Team-TUD/CTAB-GAN-Plus"),
    "TabDDPM":     ("tabddpm", "yandex-research/tab-ddpm"),
}


SYNTHCITY_MIN = (0, 2, 10)


def _synthcity_version() -> tuple:
    """Installed synthcity version, or () when it cannot be determined.


    """
    try:
        from importlib.metadata import version as _pkg_version
        v = _pkg_version("synthcity")
        parts = tuple(int(x) for x in str(v).split(".")[:3] if x.isdigit())
        return parts if parts else ()
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import synthcity
        v = getattr(synthcity, "__version__", None)
        if v:
            parts = tuple(int(x) for x in str(v).split(".")[:3] if x.isdigit())
            return parts if parts else ()
    except Exception:                                        # noqa: BLE001
        pass
    return ()


def synthcity_status() -> str:
    """
    """
    if not HAVE_SYNTHCITY:
        return "not installed  (pip install synthcity)"
    suffix = "  [trapz shim applied]" if TRAPZ_SHIM_APPLIED else ""


    ver = _synthcity_version()

    if len(ver) >= 3 and ver < SYNTHCITY_MIN:
        vs = ".".join(map(str, ver))
        mn = ".".join(map(str, SYNTHCITY_MIN))
        return (f"version {vs} is too old (needs >= {mn}); it predates "
                f"pydantic v2 support. Fix:  "
                f"pip install -r requirements.txt")
    if not _torch_supports_synthcity():
        return f"installed but unusable: {_SC_PROBE['why']}"
    try:
        from synthcity.plugins import Plugins
        if "survival_gan" not in Plugins().list():
            return "installed but the survival_gan plugin is not registered"
    except Exception as exc:                                 # noqa: BLE001
        return f"import failed: {type(exc).__name__}: {str(exc)[:90]}"
    return "ready" + suffix


def availability() -> dict:
    out = {"sdv": HAVE_SDV, "synthcity": HAVE_SYNTHCITY}
    for name, (folder, _) in REPOS.items():
        out[name] = _have_repo(folder)
    return out


REPO_ADAPTERS = {"CTABGANPlus"}

#: method -> source actually used on the most recent call ("pip"/"repo"/"ref")
_LAST_SOURCE: dict = {}


def _record(method: str, src: str) -> None:
    _LAST_SOURCE[method] = src


def last_source(method: str) -> str:
    """What this method ACTUALLY ran last time. Use this for result tables."""
    return _LAST_SOURCE.get(method, source_of(method))


SDV_METHODS = ("GaussianCopula", "CTGAN", "TVAE")
SYNTHCITY_METHODS = ("TabDDPM", "SurVAE", "SurvivalGAN", "SurvivalCTGAN")



def is_official(method: str) -> bool:
    if method in SDV_METHODS:
        return HAVE_SDV
    if method in SYNTHCITY_METHODS:
        return synthcity_usable()
    if method in REPOS:
        return method in REPO_ADAPTERS and _have_repo(REPOS[method][0])
    return False


def source_of(method: str) -> str:
    if not is_official(method):
        return "ref"
    if method in SDV_METHODS or method in SYNTHCITY_METHODS:
        return "pip"
    return "repo"


# ------------------------------------------------------------- data packing --
def _pack(X, time, event, tanh=True):
    """[X | normalised log-time | event], scaled to [-1,1] when tanh."""
    lo, hi = float(np.log1p(time.min())), float(np.log1p(time.max()))
    tn = (np.log1p(time) - lo) / (hi - lo + 1e-10)
    if tanh:
        data = np.column_stack([X * 2 - 1, tn * 2 - 1, event * 2 - 1])
    else:
        data = np.column_stack([X, tn, event])
    return data, {"lo": lo, "hi": hi, "p": X.shape[1], "tanh": tanh}


def _unpack(out, meta, event_rate: float | None = None):
    """Invert _pack.


    """
    p, lo, hi, tanh = meta["p"], meta["lo"], meta["hi"], meta["tanh"]
    raw_e = out[:, p + 1]
    if tanh:
        Xs = np.clip((out[:, :p] + 1) / 2, 0, 1)
        tn = np.clip((out[:, p] + 1) / 2, 0, 1.2)
        default_thr = 0.0
    else:
        Xs = np.clip(out[:, :p], 0, 1)
        tn = np.clip(out[:, p], 0, 1.2)
        default_thr = 0.5

    if event_rate is not None and 0.0 < event_rate < 1.0 and np.isfinite(raw_e).any():
        thr = float(np.quantile(np.nan_to_num(raw_e), 1.0 - event_rate))
    else:
        thr = default_thr
    es = (raw_e > thr).astype(float)

    ts = np.clip(np.expm1(tn * (hi - lo) + lo), 1e-3, None)
    return np.nan_to_num(Xs, nan=0.5), np.nan_to_num(ts, nan=1.0), es


def _frame(X, time, event, genes=None):
    import pandas as pd
    cols = list(genes) if genes is not None else [f"g{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["time"] = time
    df["event"] = event.astype(int)
    return df


def _from_frame(df, p):
    X = np.clip(np.nan_to_num(df.iloc[:, :p].to_numpy(float), nan=0.5), 0, 1)
    t = np.clip(np.nan_to_num(df["time"].to_numpy(float), nan=1.0), 1e-3, None)
    e = (np.nan_to_num(df["event"].to_numpy(float), nan=0.0) > 0.5).astype(float)
    return X, t, e


# =============================================================== SDV (pip) ===
def _sdv(kind, X, time, event, n, epochs, seed=None):
    """Fit an SDV synthesizer and sample n rows.

    SDV reseeds its sampler with a fixed internal seed (FIXED_RNG_SEED = 73251)
    the first time sample() is called after fit(), regardless of the global
    numpy/torch seeds. Without `seed`, every run of a deterministic model such
    as GaussianCopula therefore draws the identical cohort. Passing `seed`
    hands the run's seed to SDV's sampler. CTGAN and TVAE are left on SDV's
    default (seed=None) so their cached results stay reproducible; their runs
    already differ because their training is stochastic.
    """
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (CTGANSynthesizer, GaussianCopulaSynthesizer,
                                  TVAESynthesizer)
    df = _frame(X, time, event)
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    md.update_column("event", sdtype="categorical")
    syn = {"CTGAN": lambda: CTGANSynthesizer(md, epochs=epochs, verbose=False),
           "TVAE": lambda: TVAESynthesizer(md, epochs=epochs),
           "GaussianCopula": lambda: GaussianCopulaSynthesizer(md)}[kind]()
    syn.fit(df)
    if seed is not None:
        syn._set_random_state(int(seed))   # SDV's own seeding hook (SDV >= 1.0)
    return _from_frame(syn.sample(num_rows=n), X.shape[1])


def gaussian_copula(X, time, event, n, rng, labels=None, **kw):
    if HAVE_SDV:
        _record("GaussianCopula", "pip")
        # The fit is deterministic, so without a per-run sampling seed all five
        # runs would be identical (zero error bars). Derive it from the run's rng.
        seed = int(rng.integers(0, 2 ** 31 - 1)) if rng is not None else None
        return _sdv("GaussianCopula", X, time, event, n, 0, seed=seed)
    _record("GaussianCopula", "ref")
    return _ref_copula(X, time, event, n, rng)


def ctgan(X, time, event, n, rng, labels=None, epochs=300, **kw):
    if HAVE_SDV:
        _record("CTGAN", "pip")
        return _sdv("CTGAN", X, time, event, n, epochs)
    _record("CTGAN", "ref")
    return _ref_gan(X, time, event, n, rng, epochs,
                     seed=method_seed("CTGAN"))


def tvae(X, time, event, n, rng, labels=None, epochs=300, **kw):
    if HAVE_SDV:
        _record("TVAE", "pip")
        return _sdv("TVAE", X, time, event, n, epochs)
    _record("TVAE", "ref")
    return _ref_vae(X, time, event, n, rng, epochs,
                     seed=method_seed("TVAE"))


# ========================================================= synthcity (pip) ===
def synthcity_usable() -> bool:
    """Installed, importable, new enough, and with a compatible torch."""
    if not HAVE_SYNTHCITY:
        return False
    ver = _synthcity_version()
    if len(ver) >= 3 and ver < SYNTHCITY_MIN:
        return False
    return _torch_supports_synthcity()


_SC_PROBE = {"done": False, "ok": False, "why": ""}


def _torch_supports_synthcity() -> bool:
    """
    """
    if _SC_PROBE["done"]:
        return _SC_PROBE["ok"]
    _SC_PROBE["done"] = True
    try:
        from synthcity.plugins import Plugins
        names = Plugins().list()
        if not names:
            _SC_PROBE["why"] = "plugin registry is empty"
        else:
            _SC_PROBE["ok"] = True
    except Exception as exc:                                 # noqa: BLE001
        msg = str(exc)
        name0 = type(exc).__name__
        if not isinstance(exc, AttributeError):
            if "non-annotated attribute" in msg or "Pydantic" in name0:
                v = ".".join(map(str, _synthcity_version())) or "unknown"
                _SC_PROBE["why"] = (
                    f"synthcity {v} predates pydantic v2 support (it declares "
                    f"an unannotated model field). Fix:  "
                    f"pip install -r requirements.txt")
            elif "XGBoost" in name0 or "libxgboost" in msg or "libomp" in msg:
                import platform
                hint = ("brew install libomp" if platform.system() == "Darwin"
                        else "install the OpenMP runtime (apt install libgomp1)")
                _SC_PROBE["why"] = (
                    f"xgboost cannot load its shared library -- it needs the "
                    f"OpenMP runtime, absent by default on "
                    f"{platform.system()}. Fix:  {hint}")
            else:
                _SC_PROBE["why"] = f"{name0}: {msg[:110]}"
            return _SC_PROBE["ok"]
        if "RMSNorm" in msg:
            import torch
            try:
                import opacus
                ov = getattr(opacus, "__version__", "?")
            except Exception:                                # noqa: BLE001
                ov = "not installed"
            _SC_PROBE["why"] = (
                f"a dependency needs torch.nn.RMSNorm but torch is "
                f"{torch.__version__} (opacus {ov}). This is almost always "
                f"opacus: newer releases require torch>=2.5, while synthcity "
                f"pins torch<2.3, so upgrading torch cannot resolve it. Fix by "
                f"pinning opacus down:  pip install opacus==1.4.0   (or "
                f"pip install -r requirements-synthcity.txt)")
        else:
            _SC_PROBE["why"] = f"AttributeError: {msg[:110]}"
    except Exception as exc:                                 # noqa: BLE001
        name = type(exc).__name__
        msg = str(exc)
        if "XGBoost" in name or "libxgboost" in msg or "libomp" in msg:
            import platform
            # xgboost needs the OpenMP runtime, which Apple's clang does not
            # ship. synthcity imports xgboost transitively (xgbse), so the whole
            # package fails to import with an error that never mentions OpenMP.
            hint = ("brew install libomp" if platform.system() == "Darwin"
                    else "install the OpenMP runtime for your platform "
                         "(e.g. apt install libgomp1)")
            _SC_PROBE["why"] = (
                f"xgboost cannot load its shared library -- it needs the OpenMP "
                f"runtime, which is not installed by default on "
                f"{platform.system()}. Fix:  {hint}")
        elif "non-annotated attribute" in msg or "PydanticUserError" in name:
            v = ".".join(map(str, _synthcity_version())) or "unknown"
            _SC_PROBE["why"] = (
                f"synthcity {v} predates pydantic v2 support (it declares an "
                f"unannotated model field). Fix:  "
                f"pip install -r requirements.txt")
        else:
            _SC_PROBE["why"] = f"{name}: {msg[:110]}"
    return _SC_PROBE["ok"]


def survivalgan(X, time, event, n, rng, labels=None, epochs=300, **kw):
    if HAVE_SYNTHCITY and not _torch_supports_synthcity():
        raise RuntimeError(f"SurvivalGAN unavailable: {_SC_PROBE['why']}")
    if HAVE_SYNTHCITY:
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
        loader = SurvivalAnalysisDataLoader(
            _frame(X, time, event), target_column="event",
            time_to_event_column="time")
        plugin = Plugins().get("survival_gan", n_iter=epochs)
        plugin.fit(loader)
        _record("SurvivalGAN", "pip")
        return _from_frame(plugin.generate(count=n).dataframe(), X.shape[1])
    _record("SurvivalGAN", "ref")
    return _ref_survival_gan(X, time, event, n, rng, epochs,
                             seed=method_seed("SurvivalGAN"))


# ============================================== official repos (setup script) ==
def _add_repo_path(folder):
    path = os.path.join(EXTERNAL_DIR, folder)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def ctabgan_plus(X, time, event, n, rng, labels=None, epochs=150, genes=None,
                 batch_size=None, num_channels=None, **kw):
    """CTAB-GAN+ via the official repo. Its API is CSV-path driven.

    Two defaults in the reference implementation make it pathologically slow on
    small cohorts, so both are overridable here:

    * `batch_size=500` combined with `steps_per_epoch = max(1, n // batch_size)`
      means a 115-patient cohort still trains on 500-sample batches (drawn with
      replacement). Cost is then almost independent of n -- which is exactly what
      the observed timings show, ~4.6 s/epoch at n=115 versus ~5.5 s/epoch at
      n=499. Scaling the batch to the cohort removes that waste.
    * `num_channels=64` in a WGAN-GP CNN whose gradient penalty needs a second
      backward pass. Halving the channels cuts convolution cost roughly fourfold.


    """
    if not _have_repo("ctabganplus"):
        _record("CTABGANPlus", "ref")
        return _ref_gan(X, time, event, n, rng, epochs,
                        seed=method_seed("CTABGANPlus"))
    import inspect
    import tempfile
    root = _add_repo_path("ctabganplus")
    cwd = os.getcwd()
    try:
        from model.ctabgan import CTABGAN
        df = _frame(X, time, event, genes)
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, "data.csv")
            df.to_csv(csv, index=False)
            os.chdir(root)

            # The constructor signature differs between forks: the reference
            # repo takes no `epochs` argument and keeps it on the synthesizer
            # instead. Pass only what this copy actually accepts.
            wanted = dict(raw_csv_path=csv, test_ratio=0.2,
                          categorical_columns=["event"], log_columns=[],
                          mixed_columns={}, general_columns=[],
                          non_categorical_columns=[], integer_columns=[],
                          problem_type={"Classification": "event"},
                          epochs=epochs)
            params = inspect.signature(CTABGAN.__init__).parameters
            model = CTABGAN(**{k: v for k, v in wanted.items() if k in params})
            syn = getattr(model, "synthesizer", None)
            if syn is not None:
                if hasattr(syn, "epochs"):
                    syn.epochs = epochs
                if hasattr(syn, "batch_size"):
                    bs = batch_size
                    if bs is None:                    # scale to the cohort
                        n_train = int(len(X) * 0.8)   # test_ratio=0.2
                        bs = max(50, min(500, n_train - n_train % 2))
                    syn.batch_size = int(bs)
                if hasattr(syn, "num_channels") and num_channels is not None:
                    syn.num_channels = int(num_channels)

            model.fit()
            out = model.generate_samples()
        _record("CTABGANPlus", "repo")
        return _from_frame(out.iloc[:n], X.shape[1])
    except Exception as exc:                                 # noqa: BLE001
        print(f"      [CTAB-GAN+ repo adapter failed -> reference impl] "
              f"{type(exc).__name__}: {str(exc)[:70]}")
        _record("CTABGANPlus", "ref")
        return _ref_gan(X, time, event, n, rng, epochs,
                        seed=method_seed("CTABGANPlus"))
    finally:
        os.chdir(cwd)


def tabddpm(X, time, event, n, rng, labels=None, epochs=800, **kw):
    """TabDDPM. The official repo is config/CLI driven with no importable
    fit/sample API, so the in-repo Gaussian-diffusion reimplementation is used
    unless a repo adapter is supplied. Flagged as reference accordingly."""
    _record("TabDDPM", "ref")
    return _ref_tabular_diffusion(X, time, event, n, rng, epochs,
                                  seed=method_seed("TabDDPM"))


# ======================================================================
# synthcity plugins
# ======================================================================
# synthcity ships official implementations of several baselines we would
# otherwise have to drive through their own repositories, each with conflicting
# pins. Verified against vanderschaarlab/synthcity `main`:
#
#   generic/plugin_ddpm.py            name "ddpm"            TabDDPM (Kotelnikov
#                                                            et al., 2022)
#   generic/plugin_tvae.py            name "tvae"            TVAE
#   generic/plugin_ctgan.py           name "ctgan"           CTGAN
#   survival_analysis/plugin_survae   name "survae"          SurVAE
#   survival_analysis/..._gan         name "survival_gan"    SurvivalGAN
#   survival_analysis/..._ctgan       name "survival_ctgan"  Survival CTGAN
#   survival_analysis/..._nflow       name "survival_nflow"  Survival NFlow
#
# `_survival_pipeline.SurvivalPipeline` wraps ANY generic plugin with
# time-to-event modelling and censoring handling -- survae is literally
# SurvivalPipeline("tvae") and survival_gan is SurvivalPipeline("adsgan"). The
# same construction gives a survival-aware diffusion model from "ddpm", which is
# the closest published comparator to SubSurGen: a diffusion generator that
# models censored survival rather than treating time as another column.

# Verified working construction (synthcity 0.2.12 + torch 2.2.2 + opacus 1.4.0).
# Each entry gives the plugin name and which DataLoader it needs.
#
#   generic  -> GenericDataLoader: time and event are ordinary columns. The
#               event column is passed as target_column only so synthcity can
#               infer a schema and offer conditional sampling; it does not change
#               the generative objective.
#   survival -> SurvivalAnalysisDataLoader: censoring is modelled explicitly.
#
# "survival_ddpm" is not a registered plugin name. synthcity builds its survival
# plugins by wrapping a generic one in SurvivalPipeline -- survae is
# SurvivalPipeline("tvae"), survival_gan is SurvivalPipeline("adsgan") -- so
# SurvivalPipeline("ddpm") yields a survival-aware diffusion model.
SYNTHCITY_PLUGINS = {
    "TabDDPM":       {"plugin": "ddpm",           "loader": "generic"},
    "SurVAE":        {"plugin": "survae",         "loader": "survival"},
    "SurvivalGAN":   {"plugin": "survival_gan",   "loader": "survival"},
    "SurvivalCTGAN": {"plugin": "survival_ctgan", "loader": "survival"},

}


#: Uncensoring model used by SurvivalPipeline-based plugins. The synthcity
#: default ("survival_function_regression") is an XGBoost survival ensemble that
#: can take hours per cohort on CPU. "cox_ph" fills the same role far faster.
#: Set to "survival_function_regression" to reproduce synthcity's default, or
#: "none" to skip uncensoring entirely.
def _cfg(name: str, default):
    """Read a setting from config.py, falling back to an env var then default."""
    env = os.environ.get("SUBSURGEN_" + name.replace("SYNTHCITY_", ""), None)
    if env is not None:
        return type(default)(env)
    try:
        import config as _C
        return getattr(_C, name, default)
    except Exception:                                        # noqa: BLE001
        return default


SURVIVAL_UNCENSORING_MODEL = _cfg("SYNTHCITY_UNCENSORING_MODEL", "cox_ph")

#: synthcity resamples until the generated batch matches the training schema,
#: up to `sampling_patience` times (its default is 500). Each retry is a full
#: generation pass, and for a diffusion plugin that is the whole reverse process
#: -- so a plugin whose output sits marginally outside the observed ranges can
#: spend twenty minutes silently retrying with no progress output at all. A
#: smaller budget bounds that; synthcity returns its best attempt when the
#: budget is exhausted rather than failing.
SYNTHCITY_SAMPLING_PATIENCE = _cfg("SYNTHCITY_SAMPLING_PATIENCE", 25)


def _configure_synthcity_runtime():
    """Use conservative native runtime settings for Synthcity on macOS."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass


def _synthcity_generate(method: str, X, time, event, n, epochs=300,
                        batch_size=200, genes=None, seed=0):
    """Fit a synthcity plugin and sample from it.

    Follows the call sequence verified end to end against these cohorts:
    build the right DataLoader, construct the plugin with `random_state`,
    `n_iter` and `batch_size`, then `fit(loader)` and `generate(count=n)`.
    """
    _configure_synthcity_runtime()
    import torch
    sc_device = torch.device("cpu")
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import (GenericDataLoader,
                                                   SurvivalAnalysisDataLoader)

    spec = SYNTHCITY_PLUGINS[method]
    df = _frame(X, time, event, genes)

    if spec["loader"] == "survival":
        # SurvivalAnalysisDataLoader requires strictly positive times. The
        # cohort loader already clips non-positive values, but a generator can
        # still be handed a stray zero, so guard here too.
        keep = df["time"] > 0
        if not keep.all():
            df = df.loc[keep].reset_index(drop=True)
        loader = SurvivalAnalysisDataLoader(
            df, target_column="event", time_to_event_column="time")
    else:
        loader = GenericDataLoader(df, target_column="event")

    hp = {"n_iter": int(epochs), "batch_size": int(batch_size),
          "sampling_patience": SYNTHCITY_SAMPLING_PATIENCE}
    if spec.get("pipeline"):
        from synthcity.plugins.survival_analysis._survival_pipeline import (
            SurvivalPipeline)

        model = SurvivalPipeline(spec["plugin"], random_state=seed,
                                 uncensoring_model=SURVIVAL_UNCENSORING_MODEL,
                                 device=sc_device, **hp)
    else:
        # survae, survival_gan and survival_ctgan each wrap SurvivalPipeline
        # internally and expose the same uncensoring_model argument, so they
        # inherit the same bottleneck unless it is set here too.
        if spec["loader"] == "survival":
            hp["uncensoring_model"] = SURVIVAL_UNCENSORING_MODEL
        model = Plugins().get(spec["plugin"], random_state=seed,
                              device=sc_device, **hp)

    model.fit(loader)
    out = model.generate(count=int(n))
    out = out.dataframe() if hasattr(out, "dataframe") else out
    return _from_frame(out, X.shape[1])


assert set(SYNTHCITY_METHODS) == set(SYNTHCITY_PLUGINS), (
    "SYNTHCITY_METHODS and SYNTHCITY_PLUGINS disagree; provenance labels would "
    "be wrong")


def make_synthcity_baseline(method: str):
    """Baseline callable for one synthcity plugin."""
    def _fn(X, time, event, n, rng, labels=None, epochs=300, genes=None,
            batch_size=200, **kw):
        if not synthcity_usable():
            raise RuntimeError(f"{method} unavailable: {synthcity_status()}")
        _record(method, "pip")
        seed = int(rng.integers(0, 2 ** 31 - 1)) if rng is not None else 0
        return _synthcity_generate(method, X, time, event, n, epochs,
                                   batch_size, genes, seed)
    _fn.__name__ = f"synthcity_{method}"
    return _fn


# ==================================== in-repo reference implementations =======
def method_seed(method: str, base: int = 0) -> int:
    """Deterministic per-method seed.

    """
    import hashlib
    h = int(hashlib.md5(method.encode("utf-8")).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


def _torch(seed):
    import torch
    torch.manual_seed(seed)
    return torch


def _ref_gan(X, time, event, n, rng, epochs=300, zdim=32, seed=0):
    torch = _torch(seed)
    import torch.nn as nn
    data, meta = _pack(X, time, event)
    d = data.shape[1]
    G = nn.Sequential(nn.Linear(zdim, 128), nn.LeakyReLU(0.2),
                      nn.Linear(128, 128), nn.LeakyReLU(0.2),
                      nn.Linear(128, d), nn.Tanh())
    D = nn.Sequential(nn.Linear(d, 128), nn.LeakyReLU(0.2),
                      nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    oG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    real = torch.tensor(data, dtype=torch.float32)
    for _ in range(epochs):
        idx = torch.randint(0, len(real), (min(64, len(real)),))
        xb = real[idx]
        fake = G(torch.randn(len(xb), zdim)).detach()
        ld = bce(D(xb), torch.ones(len(xb), 1)) + bce(D(fake), torch.zeros(len(xb), 1))
        oD.zero_grad(); ld.backward(); oD.step()
        lg = bce(D(G(torch.randn(len(xb), zdim))), torch.ones(len(xb), 1))
        oG.zero_grad(); lg.backward(); oG.step()
    with torch.no_grad():
        return _unpack(G(torch.randn(n, zdim)).numpy(), meta, event.mean())


def _ref_vae(X, time, event, n, rng, epochs=300, zdim=16, seed=0):
    torch = _torch(seed)
    import torch.nn as nn
    data, meta = _pack(X, time, event)
    d = data.shape[1]
    enc = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 2 * zdim))
    dec = nn.Sequential(nn.Linear(zdim, 128), nn.ReLU(),
                        nn.Linear(128, d), nn.Tanh())
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    real = torch.tensor(data, dtype=torch.float32)
    for _ in range(epochs):
        xb = real[torch.randint(0, len(real), (min(64, len(real)),))]
        h = enc(xb)
        mu, lv = h[:, :zdim], h[:, zdim:].clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        loss = ((dec(z) - xb) ** 2).mean() + \
            1e-3 * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return _unpack(dec(torch.randn(n, zdim)).numpy(), meta, event.mean())


def _ref_copula(X, time, event, n, rng):
    from scipy import stats
    data, meta = _pack(X, time, event, tanh=False)
    d = data.shape[1]
    U = np.column_stack([stats.rankdata(data[:, j]) / (len(data) + 1)
                         for j in range(d)])
    Z = stats.norm.ppf(U)
    cov = np.corrcoef(Z, rowvar=False) + np.eye(d) * 1e-6
    Us = stats.norm.cdf(rng.multivariate_normal(np.zeros(d), cov, size=n))
    out = np.column_stack([np.quantile(data[:, j], np.clip(Us[:, j], 0, 1))
                           for j in range(d)])
    return _unpack(out, meta, event.mean())


def _ref_tabular_diffusion(X, time, event, n, rng, epochs=800, seed=0,
                           learnable_schedule=False, survival_loss=False):
    """Shared backbone for the tabular-diffusion family. Currently only
    tabddpm() calls this, always with the defaults below -- the two flags
    were for TabDiff and SurvDiff respectively, both since removed. Left in
    place (untested without those callers to exercise them) rather than
    stripped, since the only live caller never sets either to True.

    """
    torch = _torch(seed)
    import torch.nn as nn
    data, meta = _pack(X, time, event)
    d = data.shape[1]
    T = 200
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    ac = torch.cos(((steps + s) / (1 + s)) * np.pi / 2) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - ac[1:] / ac[:-1], 0, 0.999).float()
    alphas = 1 - betas
    acp = torch.cumprod(alphas, 0)

    net = nn.Sequential(nn.Linear(d + 1, 256), nn.SiLU(),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, d))
    params = list(net.parameters())
    log_scale = None
    if learnable_schedule:
        log_scale = torch.zeros(d, requires_grad=True)
        params.append(log_scale)
    risk_head = None
    if survival_loss:
        risk_head = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1))
        params += list(risk_head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    real = torch.tensor(data, dtype=torch.float32)
    tt = torch.tensor(time, dtype=torch.float32)
    ee = torch.tensor(event, dtype=torch.float32)

    for _ in range(epochs):
        idx = torch.randint(0, len(real), (min(64, len(real)),))
        x0 = real[idx]
        t = torch.randint(0, T, (len(x0),))
        noise = torch.randn_like(x0)
        if log_scale is not None:
            noise = noise * log_scale.exp().clamp(0.5, 2.0)[None, :]
        a = acp[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        pred = net(torch.cat([xt, t[:, None].float() / T], 1))
        loss = ((pred - noise) ** 2).mean()
        if risk_head is not None:
            from .model import cox_ph_loss
            r = risk_head(x0).squeeze(-1)
            if ee[idx].sum() > 1:
                loss = loss + 0.2 * cox_ph_loss(r, tt[idx], ee[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        x = torch.randn(n, d)
        for i in reversed(range(T)):
            eps = net(torch.cat([x, torch.full((n, 1), i / T)], 1))
            mean = (x - (1 - alphas[i]) / (1 - acp[i]).sqrt() * eps) / alphas[i].sqrt()
            x = mean + betas[i].sqrt() * torch.randn_like(x) if i > 0 else mean
        return _unpack(x.numpy(), meta, event.mean())


def _ref_survival_gan(X, time, event, n, rng, epochs=300, seed=0):
    """SurvivalGAN-style: covariate GAN, then a conditional time regressor.

    Follows the published decomposition (covariates and event flag first, then
    time given covariates and censoring status) rather than joint modelling.
    """
    torch = _torch(seed)
    import torch.nn as nn
    p = X.shape[1]
    Xs, _, _ = _ref_gan(X, time, event, n, rng, epochs, seed=seed)
    es = (rng.random(n) < event.mean()).astype(float)

    lo, hi = float(np.log1p(time.min())), float(np.log1p(time.max()))
    y = (np.log1p(time) - lo) / (hi - lo + 1e-10)
    net = nn.Sequential(nn.Linear(p + 1, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xin = torch.tensor(np.column_stack([X, event]), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)[:, None]
    for _ in range(max(100, epochs // 2)):
        loss = ((net(xin) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(np.column_stack([Xs, es]),
                                dtype=torch.float32)).numpy().ravel()
        resid = float(np.std(y - net(xin).numpy().ravel()))
    pred = np.clip(pred + rng.normal(0, max(resid, 1e-3), n), 0, 1.2)
    ts = np.clip(np.expm1(pred * (hi - lo) + lo), 1e-3, None)
    return Xs, ts, es


# ======================================================================
# Baseline registry
# ======================================================================
# One implementation per method, chosen by this preference order:
#
#   pip    an official package (SDV, synthcity) -- best, no repo wrangling
#   repo   an adapter drives the official repository (CTAB-GAN+ only)
#
# synthcity covers the survival and diffusion families that would otherwise need
# per-repo environments with conflicting pins:
#
#   ddpm            TabDDPM (Kotelnikov et al. 2022)
#   survae          SurVAE                     = SurvivalPipeline("tvae")
#   survival_gan    SurvivalGAN                = SurvivalPipeline("adsgan")
#   survival_ctgan  Survival CTGAN
#



IMPORT_ONLY: set = set()

_SC_OK = synthcity_usable()


def _unavailable(method: str, why: str):
    def _fn(*a, **kw):
        raise RuntimeError(f"{method} unavailable: {why}")
    return _fn


BASELINES = {
    # ---- statistical / GAN / VAE, from SDV -------------------------------
    "GaussianCopula": (gaussian_copula, "Statistical"),
    "CTGAN":          (ctgan, "GAN"),
    "TVAE":           (tvae, "VAE"),

    # ---- GAN, official repository via adapter ----------------------------
    "CTABGANPlus":    (ctabgan_plus, "GAN"),
}

# ---- synthcity: preferred for the diffusion and survival families --------
if _SC_OK:
    BASELINES.update({
        "TabDDPM":       (make_synthcity_baseline("TabDDPM"),
                          "Tabular diffusion"),
        "SurVAE":        (make_synthcity_baseline("SurVAE"),
                          "Survival VAE"),
        "SurvivalGAN":   (make_synthcity_baseline("SurvivalGAN"),
                          "Survival GAN"),
        "SurvivalCTGAN": (make_synthcity_baseline("SurvivalCTGAN"),
                          "Survival GAN"),
    })
else:
    # Without synthcity these fall back to in-repo reimplementations, which are
    # clearly labelled `ref` and must not be presented as the published models.
    BASELINES.update({
        "TabDDPM":     (tabddpm, "Tabular diffusion"),
        "SurvivalGAN": (survivalgan, "Survival GAN"),
        "SurVAE":      (_unavailable("SurVAE",
                                     "needs synthcity ('pip install synthcity')"),
                        "Survival VAE"),
    })


def available_methods(include_import_only: bool = False) -> list:
    """Methods step 3 can actually generate."""
    return [m for m in BASELINES
            if include_import_only or m not in IMPORT_ONLY]


def registry_report() -> str:
    lines = [f"{'method':16s} {'family':20s} source"]
    for m, (_, fam) in BASELINES.items():
        lines.append(f"{m:16s} {fam:20s} {source_of(m)}")
    return "\n".join(lines)