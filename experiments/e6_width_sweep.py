"""E6: network-width sweep and train/test gap.

Retrains every archetype at hidden widths 16..256 (six layers, three seeds)
and reports test accuracy, train accuracy and parameter count; then reports
the train/test gap of the released headline models.
"""
import torch
from common import (ARCHS, Log, dump, evaluate, get_config, load_archetype,
                    load_headline, n_params, train)

torch.set_num_threads(3)
WIDTHS = [16, 32, 64, 128, 256]
SEEDS = 3
log = Log("e6_width_sweep.log")
res = {}
log("=== headline models: train vs test ===")
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    m = load_headline(arch, data, cfg)
    tr, te = evaluate(m, data, cfg, "train"), evaluate(m, data, cfg, "test")
    res[arch] = {"headline": {"train": tr, "test": te, "params": n_params(m)}}
    log(f"{arch:16s} params {n_params(m):6d}  train MAPE {tr['mape']:5.2f}% R2 {tr['r2']:.3f}"
        f"  |  test MAPE {te['mape']:5.2f}% R2 {te['r2']:.3f}  (n_train {tr['n']}, n_test {te['n']})")
log("\n=== width sweep (3 seeds) ===")
for arch in ARCHS:
    data = load_archetype(get_config(archetype=arch))
    res[arch]["widths"] = []
    for w in WIDTHS:
        cfg = get_config(archetype=arch, hidden_width=w)
        runs = []
        for k in range(SEEDS):
            m = train(data, cfg, seed=cfg.seed + k)
            runs.append({"train": evaluate(m, data, cfg, "train"),
                         "test": evaluate(m, data, cfg, "test"),
                         "params": n_params(m)})
        mt = sum(r["test"]["mape"] for r in runs) / SEEDS
        st = (sum((r["test"]["mape"] - mt) ** 2 for r in runs) / SEEDS) ** 0.5
        mtr = sum(r["train"]["mape"] for r in runs) / SEEDS
        r2 = sum(r["test"]["r2"] for r in runs) / SEEDS
        fr2 = sum(r["test"]["force_r2"] for r in runs) / SEEDS
        res[arch]["widths"].append({"width": w, "params": runs[0]["params"],
                                    "test_mape": mt, "test_mape_sd": st,
                                    "train_mape": mtr, "test_r2": r2,
                                    "force_r2": fr2, "runs": runs})
        log(f"{arch:16s} width {w:3d} params {runs[0]['params']:6d}  "
            f"train MAPE {mtr:5.2f}%  test MAPE {mt:5.2f}±{st:.2f}%  "
            f"test R2 {r2:.3f}  forceR2 {fr2:.3f}")
dump("e6_width_sweep.json", res)
log.done()
