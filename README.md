# SubSurGen

**Subtype- and survival-preserving synthetic data generation for molecular cancer subtyping.**

SubSurGen generates synthetic patient cohorts that preserve both the molecular
subtype structure and the censored survival outcomes, reproducing the real
cohort's subtypes and their prognosis.

This repository contains the code, datasets, and scripts needed to
reproduce all experiments and figures reported in the paper.

> 📢 **News:** The paper *SubSurGen: Synthetic Data Generation Preserving
> Molecular Subtypes and Survival in Breast Cancer* has been accepted at the
> **20th ACM International Conference on Web Search and Data Mining
> ([WSDM 2027](https://doi.org/10.1145/3862322.3863656))**, Hong Kong,
> February 15–19, 2027. See [Citation](#citation) to cite this work.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Baselines](#baselines)
- [Metrics](#metrics)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Repository Layout](#repository-layout)
- [Citation](#citation)

---

## Environment Setup

**Requirements:** Python 3.10, macOS or Linux.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependency notes

- Core stack: NumPy, SciPy, pandas, scikit-learn, matplotlib, PyTorch 2.2.2.
- `synthcity==0.2.12` supplies TabDDPM, SurVAE, SurvivalGAN, SurvivalCTGAN. It requires `torch<2.3`, so **do not** pin `torch>=2.4`
  alongside it — the two are mutually uninstallable.
- `sdv>=1.17` supplies GaussianCopula, CTGAN, TVAE.
- `opacus==1.4.0` and `transformers<4.46` are pinned for compatibility with the rest of the stack.
- See `requirements.txt` for the full, verified-working pin set.

---

### Infrastructure Used to Run Experiments

| Component | Spec |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 7 268V, 2.20 GHz |
| RAM | 32 GB |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 (pinned) |
| Approx. wall-clock time | full `run_all.sh` run ≈ 7–8 hours; `--quick` ≈ 30 min |

Multi-seed generation and evaluation (steps 3–5) are the most compute-heavy
stages; the ablation study (step 5) is the slowest and can be skipped with
`--skip-ablation` for a faster end-to-end check.

---

## Datasets

Twelve cohorts are used, each stored as a CSV in `data/`, named
`<COHORT>_driver_genes.csv`, with the format:

```
GENE1,GENE2,...,GENEp,time,event
```

All cohorts share the same 30-gene driver panel.

**Prognostic** indicates a log-rank test with $p < 0.05$ for the subtypes
discovered by the SNF-based subtyping workflow (step 0).

| Cohort | $n$ | Event rate | Log-rank $p$ | Prognostic | Source |
|---|---:|---:|---:|:---:|---|
| GEO | 736 | 0.474 | $3.0\times10^{-4}$ | yes | [GEO](https://www.ncbi.nlm.nih.gov/geo/) |
| GSE19783 | 99 | 0.434 | 0.137 | no | [GEO](https://www.ncbi.nlm.nih.gov/geo/) |
| GSE6532 | 401 | 0.347 | $5.7\times10^{-5}$ | yes | [GEO](https://www.ncbi.nlm.nih.gov/geo/) |
| HEL | 115 | 0.217 | $1.4\times10^{-4}$ | yes | [GEO](https://www.ncbi.nlm.nih.gov/geo/) |
| MAINZ | 200 | 0.230 | 0.129 | no | [Bioconductor](https://bioconductor.org/) |
| METABRIC | 1977 | 0.577 | 0.852 | no | [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric) |
| NKI | 319 | 0.342 | 0.047 | yes | [Bioconductor](https://bioconductor.org/) |
| TCGA500 | 499 | 0.090 | 0.223 | no | [TCGA](https://www.cancer.gov/tcga) |
| TRANSBIG | 198 | 0.313 | 0.010 | yes | [Bioconductor](https://bioconductor.org/) |
| UK | 207 | 0.372 | $1.7\times10^{-3}$ | yes | [GEO](https://www.ncbi.nlm.nih.gov/geo/) |
| UNT | 133 | 0.211 | 0.724 | no | [Bioconductor](https://bioconductor.org/) |
| UPP | 234 | 0.231 | 0.082 | no | [Bioconductor](https://bioconductor.org/) |

To add a new cohort, place a correctly formatted CSV in `data/` following the
naming convention above; `subsurgen/data.py` handles cohort loading and gene
vocabulary construction, and step 0 (`step0_prepare_data.py`) will pick it up
automatically.

---

## Baselines

SubSurGen is compared against eight generators spanning statistical,
GAN-based, VAE-based, diffusion-based, and survival-specific approaches. All
baselines use their official package or official repository implementation,
and runtime provenance is recorded for every experiment so that reported
results reflect the implementation actually executed. Baseline adapters live
in `subsurgen/baselines.py`.

| Method | Family | Source |
|---|---|---|
| Gaussian Copula | Statistical | `pip` (SDV) |
| CTGAN | GAN | `pip` (SDV) |
| CTAB-GAN+ | GAN | `repo` (official adapter) |
| TVAE | VAE | `pip` (SDV) |
| TabDDPM | Tabular diffusion | `pip` (synthcity) |
| SurvivalGAN | Survival-specific GAN | `pip` (synthcity) |
| SurvivalCTGAN | Survival-specific GAN | `pip` (synthcity) |
| SurVAE | Survival-specific VAE | `pip` (synthcity) |

---

## Metrics

| # | Dimension | Metric | Better |
|---|---|---|---|
| 1 | Covariate fidelity | Mean Wasserstein distance | ↓ |
| 2 | Subtype fidelity | Subtype-centroid correlation | ↑ |
| 3 | Subtype–survival fidelity | Subtype-specific KM divergence | ↓ |
| 4 | Downstream utility | TSTR C-index | ↑ |

Implemented in `subsurgen/metrics.py`.

---

## Reproducing the Paper Results

1. **Set up the environment** (see [Environment Setup](#environment-setup)).
2. **Place cohort data** in `data/` as `<COHORT>_driver_genes.csv` (see [Datasets](#datasets)).
3. **Run the full pipeline:**

   ```bash
   bash run_all.sh
   ```

   This runs steps 0–6 with default hyperparameters (`config.py`) and
   produces trained generators, tuned guidance, synthetic cohorts across all
   baselines and seeds, the four-metric benchmark, the ablation study, and
   the final report. The paper results use per-cohort training with 2500
   steps per model and five sampling seeds (0–4).

4. **Check outputs:**
   - `results/tuning.json` — guidance chosen in step 2, reused by steps 3 and 5.
   - `results/` — per-cohort metric tables and raw evaluation outputs.
   - `RESULTS.md` — the compiled results report generated by step 6.
   - Figures generated alongside `RESULTS.md` by step 6.

5. **Partial or faster reproductions:**

   ```bash
   bash run_all.sh --quick          # ~30 min plumbing/sanity check, not paper numbers
   bash run_all.sh --skip-ablation  # skip step 5 (slowest) for a faster full run
   python run_all.py --steps 0 1 2 3 4   # run through evaluation, skip ablation/report
   python run_all.py --steps 6           # rebuild tables/figures from existing results
   ```

6. **Multi-seed / custom training budget** (the paper uses the default of 2500 steps):

   ```bash
   TRAIN_STEPS=4000 SEEDS="0 1 2 3 4" bash run_all.sh
   ```

For a check that the environment and pipeline are wired correctly before
committing to a full run, use `python run_all.py --check` followed by
`bash run_all.sh --quick`.

---

## Repository Layout

```
subsurgen/          library
├── data.py         cohort loading, gene vocabulary
├── features.py     univariate Cox feature selection
├── snf.py          SNF, spectral embedding, K selection
├── model.py        token diffusion denoiser, DDPM, guidance, Cox loss
├── trainer.py      training, generation
├── metrics.py      the four metrics + dispersion
├── downstream.py   Cox model, survival curves
└── baselines.py    baseline registry and adapters

scripts/            pipeline steps + external-repo tooling
config.py           all hyperparameters
run_all.py          master runner
run_all.sh          macOS/Linux one-shot runner
```

---

## Citation

If you use SubSurGen in your research, please cite our WSDM 2027 paper:

```bibtex
@inproceedings{vu2027subsurgen,
  author    = {Tuyen Vu and Thuc D. Le and Ha X. Tran and Lin Liu and
               Jiuyong Li and Dang Nguyen and Jia Tina Du},
  title     = {{SubSurGen}: Synthetic Data Generation Preserving Molecular
               Subtypes and Survival in Breast Cancer},
  booktitle = {Proceedings of the 20th ACM International Conference on Web
               Search and Data Mining (WSDM)},
  publisher = {ACM},
  year      = {2027},
  doi       = {10.1145/3862322.3863656}
}
```
