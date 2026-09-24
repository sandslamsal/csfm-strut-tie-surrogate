"""E5: plain ceiling label vs a one-sided hinge for non-failing
designs. Three seeds per setting; reports the failure/no-failure decision
(accuracy, false positives, false negatives), the one-sided shortfall on the
non-failing designs and the in-range accuracy.
"""
import torch
from common import ARCHS, Log, dump, evaluate, fmt, get_config, load_archetype, train

torch.set_num_threads(3)
SEEDS = 3
log = Log("e5_censored_loss.log")
res = {}
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    ceil = cfg.lambda_max - 1e-3
    counts = {s: int((data.lambda_f[getattr(data, s + '_idx')] >= ceil).sum())
              for s in ("train", "val", "test")}
    log(f"\n=== {arch}: non-failing designs train/val/test = "
        f"{counts['train']}/{counts['val']}/{counts['test']} of "
        f"{len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)} ===")
    res[arch] = {"counts": counts}
    for mode in ("plain", "hinge"):
        runs = []
        for k in range(SEEDS):
            m = train(data, cfg, seed=cfg.seed + k, hinge=(mode == "hinge"))
            s = evaluate(m, data, cfg)
            runs.append(s)
            log(f"  {mode:5s} seed{k}: {fmt(s)}  hinge {s['hinge']:.3f}")
        keys = ("mape", "r2", "cls_acc", "fp", "fn", "hinge", "force_r2")
        agg = {k_: sum(r[k_] for r in runs) / SEEDS for k_ in keys}
        res[arch][mode] = {"mean": agg, "runs": runs}
        log(f"  {mode:5s} mean : MAPE {agg['mape']:.2f}%  R2 {agg['r2']:.3f}  "
            f"cls {agg['cls_acc']:.3f}  fp {agg['fp']:.1f}  fn {agg['fn']:.1f}  "
            f"shortfall {agg['hinge']:.3f}")
dump("e5_censored_loss.json", res)
log.done()
