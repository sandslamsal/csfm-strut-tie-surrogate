"""Analysis of the non-failing (right-censored) designs.

Some designs do not fail by the load-factor ceiling lambda = 3.0; they are
right-censored. The accuracy tables report MAPE/R2 on genuine-failure designs,
so this script examines the censored designs directly:

  * failure-vs-no-failure CLASSIFICATION: does the surrogate correctly decide
    whether a design fails within the analysed range? (predicted lambda >= 3.0
    -> censored). Reports accuracy and the confusion counts per archetype.
  * a censored-aware error: for censored designs the only requirement is
    lambda_hat >= ceiling, so their contribution is a one-sided hinge
    max(0, ceiling - lambda_hat); reported alongside the genuine-failure MAPE
    so the two regimes are explicit rather than one being dropped.

Run:  python censoring.py
"""
from __future__ import annotations

import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet

ARCHS = ["deepBeam", "hammerhead", "multiColumnBent", "pileCap"]
CEIL = 3.0


def main() -> None:
    print(f"{'archetype':16s} {'ntest':>6s} {'ncens':>6s} {'cls_acc':>8s} "
          f"{'TP':>4s} {'TN':>4s} {'FP':>4s} {'FN':>4s} {'hinge':>7s}")
    tot = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
    for arch in ARCHS:
        cfg = get_config(archetype=arch)
        data = load_archetype(cfg)
        model = STMNet(len(data.theta_keys), data.n_members, cfg)
        ckpt = torch.load(f"runs/{arch}/model.pt", map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        idx = data.test_idx
        with torch.no_grad():
            pred = model(data.theta[idx])[0].numpy()
        true = data.lambda_f[idx].numpy()

        true_cens = true >= CEIL - 1e-3
        pred_cens = pred >= CEIL - 1e-3
        tp = int(np.sum(~true_cens & ~pred_cens))   # correctly called failure
        tn = int(np.sum(true_cens & pred_cens))     # correctly called censored
        fp = int(np.sum(true_cens & ~pred_cens))    # said failure, was censored
        fn = int(np.sum(~true_cens & pred_cens))    # said censored, was failure
        acc = (tp + tn) / len(idx)
        # one-sided hinge error on censored designs (only under-prediction hurts)
        hinge = float(np.mean(np.clip(CEIL - pred[true_cens], 0, None))) \
            if true_cens.any() else 0.0
        print(f"{arch:16s} {len(idx):6d} {int(true_cens.sum()):6d} "
              f"{acc:8.3f} {tp:4d} {tn:4d} {fp:4d} {fn:4d} {hinge:7.3f}")
        for k, v in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn),
                     ("n", len(idx))):
            tot[k] += v
    acc = (tot["tp"] + tot["tn"]) / tot["n"]
    print(f"{'POOLED':16s} {tot['n']:6d} {'':6s} {acc:8.3f} "
          f"{tot['tp']:4d} {tot['tn']:4d} {tot['fp']:4d} {tot['fn']:4d}")


if __name__ == "__main__":
    main()
