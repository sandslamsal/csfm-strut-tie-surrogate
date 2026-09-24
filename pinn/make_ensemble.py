"""Uncertainty quantification for the failure-load surrogate.

Combines two standard tools so that each surrogate prediction carries a
trustworthy interval:

  * a *bagged* deep ensemble -- K networks, each trained on an independent
    bootstrap resample of the training split, so the members genuinely
    disagree where the data is sparse. The mean is the prediction and the
    across-member standard deviation sigma is a raw uncertainty estimate.

  * sigma-normalised *split-conformal* calibration -- the nonconformity
    score |y - mean| / sigma is evaluated on the held-out validation split,
    and its empirical quantile turns sigma into a prediction interval with
    a finite-sample coverage guarantee. The interval width still adapts to
    sigma, so it is wide for the designs the ensemble is unsure about.

Writes, in ../figures/:
  uncertainty.pdf   -- (a) parity with conformal intervals,
                       (b) reliability diagram: raw ensemble vs conformal
  uncertainty.tex   -- a thin \\includegraphics wrapper + caption

and prints the per-archetype summary.

Run (after the dataset is generated):
    python make_ensemble.py
"""
from __future__ import annotations

import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet
from figstyle import panel

# archetype -> (legend label, marker, colour) -- csfd palette
ARCHS = [
    ("deepBeam",        "Deep beam",         "o", "#2B63A6"),
    ("hammerhead",      "Hammerhead",        "s", "#1F8A70"),
    ("multiColumnBent", "Multi-column bent", "^", "#D9761A"),
    ("pileCap",         "Pile cap",          "D", "#B8352B"),
]
K = 10                # bagged ensemble members per archetype
BASE_SEED = 41000     # member k uses BASE_SEED + k
CENSOR = 2.999        # designs with lambda_f >= CENSOR are right-censored
TARGET = 0.90         # headline conformal coverage level
SIG_FLOOR = 1e-3      # floor on sigma in the normalised conformal score

PNG = "../figures/uncertainty.pdf"
TEX = "../figures/uncertainty.tex"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
})


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def train_member(data, cfg, seed: int) -> STMNet:
    """Train one bagged member: a bootstrap resample of the training split.

    The train/validation/test split is fixed by cfg.seed in load_archetype;
    `seed` controls the bootstrap resample, the weight initialisation and the
    mini-batch order. Validation data is never seen, so it is free for the
    conformal calibration step.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    n_tr = len(data.train_idx)
    boot = data.train_idx[torch.randint(n_tr, (n_tr,), generator=g)]
    for _ in range(cfg.epochs):
        model.train()
        order = boot[torch.randperm(n_tr, generator=g)]
        for s in range(0, n_tr, cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            lam_p, F_p = model(data.theta[idx])
            lam_t = data.lambda_f[idx]
            F_t = data.failure_force[idx]
            # forces learned normalised by the applied-load scale F0 = ||P_ref||
            F0 = torch.linalg.norm(
                data.ref_load[idx].reshape(len(idx), -1), dim=1
            ).clamp(min=1.0).view(-1, 1)
            loss = (((lam_p - lam_t) ** 2).mean()
                    + cfg.w_sup * ((F_p - F_t / F0) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
    model.eval()
    return model


def predict(models: list[STMNet], theta: torch.Tensor) -> tuple:
    """Return (ensemble mean, ensemble std) over the K members."""
    with torch.no_grad():
        preds = np.stack([np.asarray(m(theta)[0].tolist(), dtype=float)
                          for m in models])
    return preds.mean(0), preds.std(0, ddof=1)


def conformal_q(scores: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile of the nonconformity scores."""
    n = scores.size
    k = math.ceil((n + 1) * level)
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def ensemble_archetype(arch: str) -> dict:
    """Bagged ensemble + conformal calibration for one archetype."""
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    out_dir = os.path.join(cfg.out_dir, arch, "ensemble")
    os.makedirs(out_dir, exist_ok=True)

    models = []
    for k in range(K):
        path = os.path.join(out_dir, f"member_{k}.pt")
        if os.path.exists(path):
            m = STMNet(len(data.theta_keys), data.n_members, cfg)
            m.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=False)["state_dict"])
            m.eval()
            print(f"  [{arch}] member {k + 1}/{K} loaded")
        else:
            m = train_member(data, cfg, BASE_SEED + k)
            torch.save({"state_dict": m.state_dict(), "seed": BASE_SEED + k}, path)
            print(f"  [{arch}] member {k + 1}/{K} trained")
        models.append(m)

    # calibration on the validation split, evaluation on the test split
    cal_mean, cal_std = predict(models, data.theta[data.val_idx])
    test_mean, test_std = predict(models, data.theta[data.test_idx])
    cal_true = np.asarray(data.lambda_f[data.val_idx].tolist(), dtype=float)
    test_true = np.asarray(data.lambda_f[data.test_idx].tolist(), dtype=float)

    # genuine-failure designs only (consistent with the accuracy tables)
    ck, tk = cal_true < CENSOR, test_true < CENSOR
    cal = {"true": cal_true[ck], "mean": cal_mean[ck],
           "std": np.clip(cal_std[ck], SIG_FLOOR, None)}
    test = {"true": test_true[tk], "mean": test_mean[tk],
            "std": np.clip(test_std[tk], SIG_FLOOR, None)}

    # sigma-normalised split-conformal scores on the calibration set
    cal["scores"] = np.abs(cal["true"] - cal["mean"]) / cal["std"]
    return {"cal": cal, "test": test}


def summarise(res: dict) -> dict:
    test, cal = res["test"], res["cal"]
    err = np.abs(test["mean"] - test["true"])
    q = conformal_q(cal["scores"], TARGET)
    half = q * test["std"]
    rho = float(np.corrcoef(np.argsort(np.argsort(test["std"])),
                            np.argsort(np.argsort(err)))[0, 1])
    return {
        "n_cal": int(cal["true"].size),
        "n_test": int(test["true"].size),
        "r2": r2_score(test["true"], test["mean"]),
        "mape": float(np.mean(err / test["true"]) * 100),
        "mean_std": float(test["std"].mean()),
        "rho": rho,
        "q90": q,
        "cov90": float(np.mean(err <= half)),
        "halfwidth": float(half.mean()),
    }


def make_figure(results: dict) -> None:
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(9.2, 4.3))

    # ---- (a) ensemble parity with 90% conformal intervals ----
    all_t = np.concatenate([r["test"]["true"] for r in results.values()])
    all_m = np.concatenate([r["test"]["mean"] for r in results.values()])
    lo = min(all_t.min(), all_m.min()) - 0.10
    hi = max(all_t.max(), all_m.max()) + 0.15
    axp.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color="0.30", zorder=2)
    for arch, label, mark, colour in ARCHS:
        test, cal = results[arch]["test"], results[arch]["cal"]
        half = conformal_q(cal["scores"], TARGET) * test["std"]
        axp.errorbar(test["true"], test["mean"], yerr=half, fmt="none",
                     ecolor=colour, elinewidth=0.6, capsize=1.2,
                     alpha=0.50, zorder=3)
        axp.scatter(test["true"], test["mean"], s=18, marker=mark,
                    facecolor=colour, edgecolor="white", linewidth=0.35,
                    alpha=0.85, zorder=4, label=label)
    axp.set_xlim(lo, hi)
    axp.set_ylim(lo, hi)
    axp.set_aspect("equal")
    axp.set_xlabel("Reference solver failure load factor, $\\lambda_f$",
                   fontsize=9.2)
    axp.set_ylabel("Ensemble mean with 90% conformal interval",
                   fontsize=9.2)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axp.spines[sp].set_color("0.4")
    axp.tick_params(length=3, color="0.4")
    axp.grid(True, linewidth=0.4, color="0.90", zorder=0)
    axp.set_axisbelow(True)
    leg = axp.legend(loc="upper left", frameon=True, fontsize=7.6,
                     handletextpad=0.4, borderpad=0.6, labelspacing=0.4)
    leg.get_frame().set_edgecolor("0.80")
    leg.get_frame().set_linewidth(0.6)
    panel(axp, "a", "Ensemble mean with 90% conformal interval")

    # ---- (b) reliability diagram: raw ensemble vs conformal ----
    axc.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.40", zorder=2)
    axc.text(0.97, 0.88, "under-confident", fontsize=6.6, color="0.50",
             ha="right", va="center")
    axc.text(0.60, 0.16, "over-confident", fontsize=6.6, color="0.50",
             ha="center", va="center")

    levels = np.linspace(0.10, 0.95, 30)
    cal_all_err, cal_all_std, cal_all_scores = [], [], []
    test_all_err, test_all_std = [], []
    for r in results.values():
        cal, test = r["cal"], r["test"]
        cal_all_scores.append(cal["scores"])
        test_all_err.append(np.abs(test["mean"] - test["true"]))
        test_all_std.append(test["std"])
    test_all_err = np.concatenate(test_all_err)
    test_all_std = np.concatenate(test_all_std)

    # raw Gaussian ensemble: nominal coverage from erf, observed from sigma
    zs = np.array([math.sqrt(2) * _erfinv(2 * p - 1) for p in levels])
    raw_obs = np.array([float(np.mean(test_all_err <= z * test_all_std))
                        for z in zs])
    axc.plot(levels, raw_obs, "-", color="#B8352B", lw=1.6, zorder=4,
             label="raw ensemble $\\sigma$")

    # conformal: per-archetype quantile, pooled observed coverage
    conf_obs = []
    for p in levels:
        hit = []
        for r in results.values():
            q = conformal_q(r["cal"]["scores"], p)
            t = r["test"]
            hit.append(np.abs(t["mean"] - t["true"]) <= q * t["std"])
        conf_obs.append(float(np.mean(np.concatenate(hit))))
    axc.plot(levels, conf_obs, "-", color="0.15", lw=1.9, zorder=5,
             label="conformal calibration")

    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.set_aspect("equal")
    axc.set_xlabel("Target coverage", fontsize=9.2)
    axc.set_ylabel("Observed coverage on test designs", fontsize=9.2)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axc.spines[sp].set_color("0.4")
    axc.tick_params(length=3, color="0.4")
    axc.grid(True, linewidth=0.4, color="0.90", zorder=0)
    axc.set_axisbelow(True)
    leg = axc.legend(loc="upper left", frameon=True, fontsize=7.6,
                     handletextpad=0.6, borderpad=0.6)
    leg.get_frame().set_edgecolor("0.80")
    leg.get_frame().set_linewidth(0.6)
    panel(axc, "b", "Reliability diagram, pooled")

    fig.savefig(PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PNG}")


def _erfinv(x: float) -> float:
    """Inverse error function (Winitzki approximation; ample for plotting)."""
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


def write_tex() -> None:
    tex = r"""%% Figure: bagged-ensemble uncertainty with conformal calibration.
%% PNG rendered by pinn/make_ensemble.py (600 dpi) -- do not edit by hand.
\begin{figure}[!htb]
  \centering
  \includegraphics[width=\linewidth]{uncertainty.pdf}
  \caption{Predictive uncertainty of the surrogate, on the genuine-failure
    test designs of all four archetypes. (a)~Bagged-ensemble mean
    prediction against the reference Compatible Stress Field Method
    solver, with bars showing the 90\% split-conformal prediction
    interval; its width
    adapts to the ensemble spread. (b)~Reliability diagram: the raw
    ensemble standard deviation, read as a Gaussian interval, is
    under-dispersed and falls below the diagonal, whereas the
    $\sigma$-normalised conformal interval tracks the target coverage, as
    its finite-sample guarantee requires.}
  \label{fig:uncertainty}
\end{figure}
"""
    with open(TEX, "w") as fh:
        fh.write(tex)


def main() -> None:
    results = {}
    for arch, label, _, _ in ARCHS:
        print(f"[ensemble] {label} ...")
        results[arch] = ensemble_archetype(arch)

    make_figure(results)
    write_tex()

    print("\n=== bagged ensemble + conformal calibration "
          "(genuine-failure test designs) ===")
    print(f"{'archetype':18s} {'ncal':>5s} {'ntest':>6s} {'R2':>7s} "
          f"{'MAPE%':>7s} {'rho':>6s} {'q90':>6s} {'cov90':>7s} {'half':>7s}")
    pooled_err, pooled_std, pooled_hit = [], [], []
    for arch, label, _, _ in ARCHS:
        s = summarise(results[arch])
        print(f"{label:18s} {s['n_cal']:5d} {s['n_test']:6d} {s['r2']:7.3f} "
              f"{s['mape']:7.2f} {s['rho']:6.2f} {s['q90']:6.2f} "
              f"{s['cov90']:7.2f} {s['halfwidth']:7.3f}")
        t, c = results[arch]["test"], results[arch]["cal"]
        err = np.abs(t["mean"] - t["true"])
        q = conformal_q(c["scores"], TARGET)
        pooled_err.append(err)
        pooled_std.append(t["std"])
        pooled_hit.append(err <= q * t["std"])
    pe = np.concatenate(pooled_err)
    ps = np.concatenate(pooled_std)
    rho = float(np.corrcoef(np.argsort(np.argsort(ps)),
                            np.argsort(np.argsort(pe)))[0, 1])
    print(f"{'POOLED':18s} {'':5s} {pe.size:6d} {'':7s} {'':7s} {rho:6.2f} "
          f"{'':6s} {np.mean(np.concatenate(pooled_hit)):7.2f}")
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
