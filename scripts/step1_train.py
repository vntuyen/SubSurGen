"""Step 1 -- Train the SubSurGen generator.


  An INDEPENDENT generator is fitted on each cohort. Every baseline
  (GaussianCopula, CTGAN, TVAE, TabSyn, ...) is also fitted per cohort, so
  this is the only like-for-like comparison. Each synthetic cohort depends
  solely on its own real cohort, with no cross-cohort information.


Usage:
    python scripts/step1_train.py [--steps N]
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch

from common import (banner, describe_device, get_device, load_episodes,
                    make_diffusion, make_model, parse_common_args, write_json)
import os
import config as C
from subsurgen.trainer import train_episodic

warnings.filterwarnings("ignore")


def main() -> dict:
    p = argparse.ArgumentParser(description="Train the SubSurGen generator")
    p.add_argument("--steps", type=int, default=None, help="override training steps")
    p.add_argument("--mode", choices=["per_cohort", "episodic"], default=None,
                   help="override config.TRAINING_MODE")
    args = parse_common_args(p)
    steps = args.steps or C.TRAIN_STEPS
    mode = args.mode or C.TRAINING_MODE

    banner(f"STEP 1 -- TRAINING ({mode})")
    episodes, vocab = load_episodes()
    print(f"cohorts: {list(episodes)}")
    print(f"device : {describe_device()}")
    print(f"steps={steps}  lambda_cox={C.LAMBDA_COX}  "
          f"feature_dropout={C.FEATURE_DROPOUT}  cond_dropout={C.COND_DROPOUT}")

    diffusion = make_diffusion()
    t0 = time.time()
    meta = {"mode": mode, "steps": steps}

    if mode == "per_cohort":
        print("\nfitting an INDEPENDENT generator per cohort "
              "(matches how every baseline is fitted)\n")
        os.makedirs(C.MODELS_DIR, exist_ok=True)
        n_params = 0
        for name, ep in episodes.items():
            t1 = time.time()
            model = make_model(len(vocab))
            train_episodic(model, diffusion, [ep], steps=steps,
                           batch_size=C.BATCH_SIZE, lr=C.LR,
                           lambda_cox=C.LAMBDA_COX,
                           feature_dropout=C.FEATURE_DROPOUT,
                           cond_dropout=C.COND_DROPOUT, seed=C.SEED,
                           verbose=False, log_every=10 ** 9,
                           device=get_device())
            torch.save(model.state_dict(),
                       os.path.join(C.MODELS_DIR, f"{name}.pt"))
            n_params = sum(q.numel() for q in model.parameters())
            print(f"  {name:10s} {time.time()-t1:5.0f}s")
        # marker so downstream steps know which layout to load
        with open(C.MODE_MARKER, "w") as f:
            f.write("per_cohort")
        print(f"\nmodels saved under {C.MODELS_DIR}")
    else:

        model = make_model(len(vocab))
        history = train_episodic(
            model, diffusion, list(episodes.values()), steps=steps,
            batch_size=C.BATCH_SIZE, lr=C.LR, lambda_cox=C.LAMBDA_COX,
            feature_dropout=C.FEATURE_DROPOUT, cond_dropout=C.COND_DROPOUT,
            seed=C.SEED, log_every=max(1, steps // 5),
            device=get_device())
        torch.save(model.state_dict(), C.MODEL_PT)
        n_params = sum(q.numel() for q in model.parameters())
        meta["history"] = [{"step": s_, "loss_diffusion": d, "loss_cox": c}
                           for s_, d, c in history]
        with open(C.MODE_MARKER, "w") as f:
            f.write("episodic")
        print(f"\nmodel saved to {C.MODEL_PT}")

    elapsed = time.time() - t0
    print(f"trained {n_params:,} parameters per model in {elapsed:.0f}s")
    meta.update(elapsed_sec=round(elapsed, 1), n_parameters=int(n_params))
    write_json("step1_train_history.json", meta)
    return {"elapsed": elapsed}


if __name__ == "__main__":
    main()
