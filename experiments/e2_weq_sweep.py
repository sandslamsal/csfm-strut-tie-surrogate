"""E2: equilibrium-weight sweep on all four archetypes.

Trains one network per (archetype, w_eq) at w_F = 1 and reports the
failure-load accuracy, the force accuracy and the nodal-equilibrium residual
of the predicted state, together with the residual of the reference labels
themselves (the floor set by the solver's own convergence).
"""
import torch
from common import (ARCHS, Log, dump, evaluate, eq_residual_pct, fmt,
                    get_config, load_archetype, train)

torch.set_num_threads(3)
WEQ = [0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SEEDS = 3
log = Log("e2_weq_sweep.log")
res = {}
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    gi = data.test_idx[data.lambda_f[data.test_idx] < cfg.lambda_max - 1e-3]
    ref_eq = float(eq_residual_pct(data.failure_force[gi],
                                   data.lambda_f[gi], data, gi).mean())
    log(f"\n=== {arch}: reference-label residual {ref_eq:.3f}% ===")
    res[arch] = {"ref_eq": ref_eq, "rows": []}
    for w in WEQ:
        runs = []
        for k in range(SEEDS):
            c = get_config(archetype=arch, w_sup=1.0, w_eq=w)
            m = train(data, c, seed=c.seed + k)
            runs.append(evaluate(m, data, c))
        agg = {key: (sum(r[key] for r in runs) / SEEDS) for key in
               ("mape", "r2", "force_r2", "force_rel", "eq")}
        sd = {key: (sum((r[key] - agg[key]) ** 2 for r in runs) / SEEDS) ** 0.5
              for key in agg}
        row = {"w_eq": w, **agg, "sd": sd, "runs": runs}
        res[arch]["rows"].append(row)
        log(f"w_eq={w:7.2f}  MAPE {agg['mape']:5.2f}±{sd['mape']:.2f}%  "
            f"R2 {agg['r2']:.3f}±{sd['r2']:.3f}  forceR2 {agg['force_r2']:.4f}  "
            f"forceRel {agg['force_rel']:5.2f}%  eq {agg['eq']:5.3f}±{sd['eq']:.3f}%")
dump("e2_weq_sweep.json", res)
log.done()
