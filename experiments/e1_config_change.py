"""E1: retraining for a changed loading configuration.

The training deep beam carries one midspan load. A four-point variant (two
loads at the quarter points; 8 nodes, 13 members instead of 6 and 9) is
generated with the same sampler and solver (exportVariant.ts). The script
measures what the new configuration requires: (a) a network trained from
scratch on growing fractions of the new data, three subsamples each; (b) a
network warm-started from the released deep-beam model's hidden layers (the
input space is identical, only the output head changes shape); and (c) the
wall-clock of the solver sweep and of one training run.
"""
import json
import os
import time

import numpy as np
import torch
from common import (HERE, Log, dump, evaluate, get_config, load_archetype, load_json, train)

torch.set_num_threads(3)
FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]
SUBS = 3
DATA = str(HERE.parent / "data" / "dataset_deepBeam2P.json.gz")
log = Log("e1_config_change.log")

meta = load_json(DATA)["meta"]
log(f"solver sweep: {meta['accepted']} designs in {meta['sweepSeconds']:.1f} s "
    f"({1e3 * meta['sweepSeconds'] / meta['accepted']:.2f} ms per design)")

cfg = get_config(archetype="deepBeam2P", dataset_path=DATA)
data = load_archetype(cfg)
log(f"deepBeam2P: nodes {data.n_nodes}, members {data.n_members}, "
    f"train/val/test {len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)}, "
    f"non-failing in test {int((data.lambda_f[data.test_idx] >= cfg.lambda_max - 1e-3).sum())}")
src = torch.load(os.path.join(cfg.out_dir, "deepBeam", "model.pt"),
                 map_location="cpu", weights_only=False)["state_dict"]

# reference: the released single-load deep-beam accuracy, for the same data size
res = {"sweep_seconds": meta["sweepSeconds"], "n_designs": meta["accepted"],
       "scratch": [], "warm": []}
t0 = time.time()
m_full = train(data, cfg)
t_train = time.time() - t0
s_full = evaluate(m_full, data, cfg)
log(f"full data, from scratch: MAPE {s_full['mape']:.2f}% R2 {s_full['r2']:.3f} "
    f"forceR2 {s_full['force_r2']:.3f} eq {s_full['eq']:.2f}%  (training {t_train:.0f} s)")
res["train_seconds"] = t_train
res["full"] = s_full

n_tr = len(data.train_idx)
for mode in ("scratch", "warm"):
    for f in FRACS:
        n = max(8, int(f * n_tr))
        runs = []
        for k in range(SUBS):
            g = torch.Generator().manual_seed(1000 + k)
            sub = data.train_idx[torch.randperm(n_tr, generator=g)[:n]]
            m = train(data, cfg, seed=cfg.seed + k, train_idx=sub,
                      init_state=src if mode == "warm" else None)
            runs.append(evaluate(m, data, cfg))
        mape = np.array([r["mape"] for r in runs])
        r2 = np.array([r["r2"] for r in runs])
        res[mode].append({"frac": f, "n": n, "mape": float(mape.mean()), "sd": float(mape.std()),
                          "r2": float(r2.mean()), "runs": runs})
        log(f"{mode:7s} n={n:3d} ({int(100*f):3d}%): MAPE {mape.mean():5.2f}±{mape.std():.2f}%  "
            f"R2 {r2.mean():.3f}")
dump("e1_config_change.json", res)
log.done()
