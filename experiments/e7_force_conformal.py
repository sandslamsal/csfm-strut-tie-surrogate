"""E7: split-conformal intervals on the member forces.

Uses the released bagged ensembles. For every (design, member) the ensemble
mean and spread of the normalised force F/F0 give a sigma-normalised
nonconformity score; the score's finite-sample quantile on the validation
split sets the interval, whose coverage is then measured on the test split,
pooled over members and per member. A Bonferroni-corrected simultaneous
interval (all members of a design at once) is reported as well. The
failure-load interval is recomputed alongside as a check against make_ensemble.py.
"""
import math
import numpy as np
import torch
from common import (ARCHS, F0_of, Log, dump, get_config, load_archetype,
                    load_ensemble)

torch.set_num_threads(2)
LEVEL, FLOOR = 0.90, 1e-4
log = Log("e7_force_conformal.log")


def cq(scores, level):
    n = scores.size
    k = math.ceil((n + 1) * level)
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def ens_predict(models, theta):
    with torch.no_grad():
        lam = np.stack([m(theta)[0].numpy() for m in models])
        F = np.stack([m(theta)[1].numpy() for m in models])
    return lam.mean(0), lam.std(0, ddof=1), F.mean(0), F.std(0, ddof=1)


res = {}
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    models = load_ensemble(arch, data, cfg)
    out = {}
    split = {}
    for name, idx in (("val", data.val_idx), ("test", data.test_idx)):
        gen = data.lambda_f[idx] < cfg.lambda_max - 1e-3
        idx = idx[gen]
        lm, ls, Fm, Fs = ens_predict(models, data.theta[idx])
        F0 = F0_of(data, idx).numpy()
        Ft = data.failure_force[idx].numpy() / F0[:, None]
        split[name] = {"lam_true": data.lambda_f[idx].numpy(), "lm": lm,
                       "ls": np.clip(ls, 1e-3, None), "Ft": Ft, "Fm": Fm,
                       "Fs": np.clip(Fs, FLOOR, None), "F0": F0}
    v, t = split["val"], split["test"]
    M = t["Ft"].shape[1]
    # failure load (check against make_ensemble.py)
    q_lam = cq(np.abs(v["lam_true"] - v["lm"]) / v["ls"], LEVEL)
    cov_lam = float(np.mean(np.abs(t["lam_true"] - t["lm"]) <= q_lam * t["ls"]))
    # member forces, pooled score
    sv = (np.abs(v["Ft"] - v["Fm"]) / v["Fs"]).ravel()
    q_F = cq(sv, LEVEL)
    hit = np.abs(t["Ft"] - t["Fm"]) <= q_F * t["Fs"]
    cov_F = float(hit.mean())
    cov_member = hit.mean(0)
    half = q_F * t["Fs"]                                   # in units of F0
    half_rel = half / np.clip(np.abs(t["Ft"]), 1e-3, None)
    # simultaneous (all members of a design) with Bonferroni
    q_sim = cq(sv, 1 - (1 - LEVEL) / M)
    cov_sim = float(np.mean(np.all(np.abs(t["Ft"] - t["Fm"]) <= q_sim * t["Fs"], axis=1)))
    # per-member calibration instead of pooled
    q_pm = np.array([cq(np.abs(v["Ft"][:, m] - v["Fm"][:, m]) / v["Fs"][:, m], LEVEL)
                     for m in range(M)])
    cov_pm = float(np.mean(np.abs(t["Ft"] - t["Fm"]) <= q_pm[None, :] * t["Fs"]))
    rho = float(np.corrcoef(np.argsort(np.argsort(t["Fs"].ravel())),
                            np.argsort(np.argsort(np.abs(t["Ft"] - t["Fm"]).ravel())))[0, 1])
    out = {"n_val": int(len(v["lm"])), "n_test": int(len(t["lm"])), "members": M,
           "lam_q90": q_lam, "lam_cov90": cov_lam,
           "F_q90": q_F, "F_cov90": cov_F, "F_cov90_min_member": float(cov_member.min()),
           "F_cov90_max_member": float(cov_member.max()),
           "F_cov90_per_member_calibrated": cov_pm,
           "F_half_mean_F0": float(half.mean()), "F_half_median_rel_pct": float(np.median(half_rel) * 100),
           "F_sim_q": q_sim, "F_sim_cov": cov_sim, "rho_sigma_err": rho,
           "member_ids": data.member_ids, "cov_member": cov_member.tolist()}
    res[arch] = out
    log(f"\n=== {arch}: {M} members, cal {out['n_val']}, test {out['n_test']} ===")
    log(f"failure load : q90 {q_lam:.2f}  coverage {cov_lam:.3f}")
    log(f"member forces: q90 {q_F:.2f}  pooled coverage {cov_F:.3f}  "
        f"(per member {cov_member.min():.2f}-{cov_member.max():.2f}; per-member calibration {cov_pm:.3f})")
    log(f"               half-width mean {half.mean():.3f} F0, median {np.median(half_rel)*100:.1f}% of |F|;  "
        f"spread/error rank corr {rho:.2f}")
    log(f"simultaneous (all {M} members, Bonferroni): q {q_sim:.2f}  coverage {cov_sim:.3f}")
dump("e7_force_conformal.json", res)
log.done()
