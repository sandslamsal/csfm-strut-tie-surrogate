"""Evaluation of the strut and tie surrogate network.

The network predicts the failure load factor directly, so evaluation is a
direct comparison of the predicted lambda_f against the CSFM oracle on the
held-out test split.

Usage:
    python evaluate.py --archetype deepBeam
"""
from __future__ import annotations

import argparse
import os

import torch

from config import Config, get_config
from data import load_archetype
from model import STMNet


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--archetype", default=Config.archetype)
    p.add_argument("--dataset_path", default=Config.dataset_path)
    p.add_argument("--device", default=Config.device)
    a = p.parse_args()
    return get_config(archetype=a.archetype, dataset_path=a.dataset_path,
                      device=a.device)


def metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    err = pred - true
    rmse = torch.sqrt((err ** 2).mean()).item()
    mape = (err.abs() / true.clamp(min=1e-6)).mean().item() * 100.0
    ss_res = (err ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum().clamp(min=1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()
    return {"MAPE": mape, "RMSE": rmse, "R2": r2}


def main() -> None:
    cfg = parse_args()
    ckpt_path = os.path.join(cfg.out_dir, cfg.archetype, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no trained model at {ckpt_path}; run train.py")
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)

    data = load_archetype(cfg).to(cfg.device)
    model = STMNet(len(data.theta_keys), data.n_members, cfg).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    idx = data.test_idx
    with torch.no_grad():
        pred, _ = model(data.theta[idx])
    true = data.lambda_f[idx]

    censored = true >= cfg.lambda_max - 1e-3
    print(f"\narchetype={cfg.archetype}  test designs={len(idx)}  "
          f"censored (lambda_f={cfg.lambda_max})={int(censored.sum())}")

    all_m = metrics(pred, true)
    fail_m = (metrics(pred[~censored], true[~censored]) if (~censored).any()
              else {"MAPE": float("nan"), "RMSE": float("nan"),
                    "R2": float("nan")})

    print("\n  failure-load accuracy (lambda_f)")
    print(f"  {'subset':<24}{'MAPE %':>10}{'RMSE':>10}{'R2':>10}")
    for name, m in (("genuine failures", fail_m),
                    ("all (incl. censored)", all_m)):
        print(f"  {name:<24}{m['MAPE']:>10.2f}{m['RMSE']:>10.3f}{m['R2']:>10.3f}")


if __name__ == "__main__":
    main()
