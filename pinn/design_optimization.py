"""Illustrative application: reliability-aware reinforcement minimisation.

Demonstrates the surrogate driving a design optimisation that a scalar capacity
regressor cannot. For a fixed deep-beam geometry and a required load, the
bottom-tie steel is minimised over the reinforcement grid (bar diameter x bar
count) subject to a capacity constraint expressed on the *lower* end of the
split-conformal interval (reliability-aware), while the domain-of-validity flag
keeps the search inside the trained region. The surrogate evaluates the whole
grid in a flash; the chosen optimum and every grid point are then VERIFIED
against the reference CSFM solver (scripts/verifyDesign.ts), and the predicted
member-force state identifies the governing tie.

Outputs:
  runs/design_optimization.json     -- all numbers of the demonstration
  ../figures/optimization.pdf/.tex  -- the two-panel figure

Run (from pinn, after the ensemble exists):
    python design_optimization.py
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from data import load_archetype
from model import STMNet
from make_ensemble import predict, conformal_q, K, CENSOR, SIG_FLOOR, TARGET

ARCH = "deepBeam"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "solver"))  # solver root (holds scripts/)
FLAG_PCTL = 95.0                       # domain-of-validity percentile
BAR_DIA = [20, 25, 28, 32, 36]         # standard bar diameters (mm)
BAR_COUNT = list(range(4, 11))         # 4..10 bars

# deepBeam design-space ranges, u-ordering -- mirrors scripts/datasetSpecs.ts
RANGES = {"span": (2000, 6000), "height": (1400, 3600), "thickness": (300, 700),
          "supportWidth": (300, 600), "P": (1000e3, 6000e3), "botDia": (20, 36),
          "botCount": (4, 10), "fck": (21, 69), "fy": (280, 690)}
U_DIMS = ["span", "height", "thickness", "supportWidth", "P",
          "botDia", "botCount", "fck", "fy"]

# fixed design envelope (mid-range, in-domain): geometry + materials
BASE = {"span": 4000.0, "height": 2200.0, "thickness": 500.0,
        "supportWidth": 400.0, "fck": 35.0, "fy": 500.0}

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.linewidth": 0.9, "savefig.dpi": 600, "pdf.fonttype": 42,
                     "mathtext.fontset": "cm"})
C_SUR, C_SOL, C_OK, C_BAD = "#26629E", "#BE342E", "#0E7072", "#E47E1C"
C_FLAG = "#7A4FA3"


# --------------------------------------------------------------------------- #
# load surrogate + bagged ensemble                                            #
# --------------------------------------------------------------------------- #
cfg = get_config(archetype=ARCH)
data = load_archetype(cfg)
THETA_KEYS = data.theta_keys
TMEAN = data.theta_mean.numpy()
TSTD = data.theta_std.numpy()
MEMBER_IDS = data.member_ids


def _load(path: str) -> STMNet:
    m = STMNet(len(THETA_KEYS), data.n_members, cfg)
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["state_dict"])
    m.eval()
    return m


main_model = _load(os.path.join("runs", ARCH, "model.pt"))
ens = [_load(os.path.join("runs", ARCH, "ensemble", f"member_{k}.pt"))
       for k in range(K)]

# conformal q90 on the non-censored validation split; domain-of-validity tau
cal_mean, cal_std = predict(ens, data.theta[data.val_idx])
cal_true = np.asarray(data.lambda_f[data.val_idx].tolist(), dtype=float)
ck = cal_true < CENSOR
scores = np.abs(cal_true[ck] - cal_mean[ck]) / np.clip(cal_std[ck], SIG_FLOOR, None)
Q90 = conformal_q(scores, TARGET)
test_mean, test_std = predict(ens, data.theta[data.test_idx])
test_true = np.asarray(data.lambda_f[data.test_idx].tolist(), dtype=float)
TAU = float(np.percentile(test_std[test_true < CENSOR], FLAG_PCTL))
print(f"[demo] conformal q90={Q90:.3f}  domain-of-validity tau={TAU:.4f}")


# --------------------------------------------------------------------------- #
# surrogate helpers                                                           #
# --------------------------------------------------------------------------- #
def _theta(p: dict) -> torch.Tensor:
    raw = np.array([[p[k] for k in THETA_KEYS]], dtype=np.float32)
    return torch.tensor((raw - TMEAN) / TSTD, dtype=torch.float32)


def surrogate(p: dict) -> dict:
    """Ensemble lambda_f (mean, std) + main-model member forces (physical N)."""
    th = _theta(p)
    lam_mean, lam_std = predict(ens, th)
    with torch.no_grad():
        _, f_norm = main_model(th)
    F0 = p["P"]                                   # ||P_ref|| for a single load
    forces = f_norm.numpy()[0] * F0
    sig = float(lam_std[0])
    return {"lam_mean": float(lam_mean[0]), "lam_std": sig,
            "lam_lo": float(lam_mean[0]) - Q90 * sig,      # conformal lower bound
            "flag": sig > TAU,                              # out-of-domain?
            "forces": {m: float(forces[i]) for i, m in enumerate(MEMBER_IDS)}}


def steel_area(dia: float, count: int) -> float:
    return count * math.pi / 4.0 * dia ** 2        # mm^2


def to_u(p: dict) -> list:
    return [(p[k] - RANGES[k][0]) / (RANGES[k][1] - RANGES[k][0]) for k in U_DIMS]


def verify(designs: list) -> list:
    """Run the reference CSFM solver on a list of param dicts (one tsx call)."""
    req = json.dumps({"archetype": ARCH, "us": [to_u(p) for p in designs]})
    res = subprocess.run(["npx", "tsx", "scripts/verifyDesign.ts"],
                         input=req, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"verifier failed:\n{res.stderr[-2000:]}")
    return json.loads(res.stdout)


# --------------------------------------------------------------------------- #
# 1. calibrate the required load so the constraint bites near mid-grid         #
# --------------------------------------------------------------------------- #
def lam_at(P: float, dia: float, count: int) -> float:
    p = {**BASE, "P": P, "botDia": dia, "botCount": count}
    return surrogate(p)["lam_mean"]


# bisect P so the median reinforcement (28 mm, 7 bars) sits at lambda ~ 1.15
target_lam, dia_m, cnt_m = 1.15, 28, 7
lo_P, hi_P = RANGES["P"]
for _ in range(40):
    mid = 0.5 * (lo_P + hi_P)
    if lam_at(mid, dia_m, cnt_m) > target_lam:
        lo_P = mid                       # lambda too high -> increase load
    else:
        hi_P = mid
P_REQ = 0.5 * (lo_P + hi_P)
print(f"[demo] required load P_req = {P_REQ/1e3:.0f} kN "
      f"(median reinf lambda={lam_at(P_REQ, dia_m, cnt_m):.2f})")


# --------------------------------------------------------------------------- #
# 2. evaluate the full reinforcement grid with the surrogate                   #
# --------------------------------------------------------------------------- #
grid = []
for dia in BAR_DIA:
    for count in BAR_COUNT:
        p = {**BASE, "P": P_REQ, "botDia": dia, "botCount": count}
        s = surrogate(p)
        grid.append({"botDia": dia, "botCount": count,
                     "As": steel_area(dia, count), "params": p, **s})

# 3. verify EVERY grid design against the reference solver (single tsx call)
vr = verify([g["params"] for g in grid])
for g, v in zip(grid, vr):
    g["lam_solver"] = v["failureLoadFactor"]
    g["mode_solver"] = v["failureMode"]
    g["stable"] = v["stable"]
    g["forces_solver"] = v["failureForces"]
    g["f0_solver"] = v.get("f0")


# --------------------------------------------------------------------------- #
# 4. pick the optima                                                          #
# --------------------------------------------------------------------------- #
def min_steel(feasible):
    return min(feasible, key=lambda g: g["As"]) if feasible else None


point_feas = [g for g in grid if g["lam_mean"] >= 1.0 and not g["flag"]]
uq_feas = [g for g in grid if g["lam_lo"] >= 1.0 and not g["flag"]]
opt_point = min_steel(point_feas)       # point-estimate optimum
opt_uq = min_steel(uq_feas)             # reliability-aware optimum

# governing tie at the UQ optimum: the member named in the solver's failure
# mode (the tie that ruptures), which the predicted force state should flag as
# the most heavily loaded tension member.
_m = re.search(r"tie (\w+)", opt_uq["mode_solver"] or "")
gov_id = (_m.group(1) if _m and _m.group(1) in opt_uq["forces"]
          else max(opt_uq["forces"], key=lambda m: opt_uq["forces"][m]))


# --------------------------------------------------------------------------- #
# 5. report + persist                                                         #
# --------------------------------------------------------------------------- #
def summary(g, tag):
    return (f"  {tag:11} {g['botCount']}x{g['botDia']}mm  As={g['As']:.0f} mm^2  "
            f"lam_sur={g['lam_mean']:.3f} (lo {g['lam_lo']:.3f})  "
            f"lam_solver={g['lam_solver']:.3f}  flag={g['flag']}")


print("\n[demo] grid surrogate-vs-solver lambda_f:")
lam_s = np.array([g["lam_mean"] for g in grid])
lam_v = np.array([g["lam_solver"] for g in grid], dtype=float)
print(f"  grid R2(sur,solver)={1 - np.sum((lam_v-lam_s)**2)/np.sum((lam_v-lam_v.mean())**2):.3f}  "
      f"max|dlam|={np.max(np.abs(lam_v-lam_s)):.3f}")
print(summary(opt_point, "point-est"))
print(summary(opt_uq, "UQ-aware"))
steel_premium = 100.0 * (opt_uq["As"] - opt_point["As"]) / opt_point["As"]
print(f"  UQ premium over point estimate: {steel_premium:+.1f}% steel")
print(f"  governing member at UQ optimum: {gov_id}  "
      f"(surrogate {opt_uq['forces'][gov_id]/1e3:.0f} kN, "
      f"solver {opt_uq['forces_solver'].get(gov_id, float('nan'))/1e3:.0f} kN)")
print(f"  UQ optimum solver mode: {opt_uq['mode_solver']}")

result = {
    "archetype": ARCH, "base": BASE, "P_req_kN": P_REQ / 1e3,
    "q90": Q90, "tau": TAU,
    "grid_r2_surrogate_solver": float(
        1 - np.sum((lam_v - lam_s) ** 2) / np.sum((lam_v - lam_v.mean()) ** 2)),
    "grid_max_abs_dlam": float(np.max(np.abs(lam_v - lam_s))),
    "opt_point": {k: opt_point[k] for k in
                  ("botDia", "botCount", "As", "lam_mean", "lam_lo", "lam_solver", "flag")},
    "opt_uq": {k: opt_uq[k] for k in
               ("botDia", "botCount", "As", "lam_mean", "lam_lo", "lam_solver", "flag")},
    "steel_premium_pct": steel_premium,
    "governing_member": gov_id,
    "gov_force_surrogate_kN": opt_uq["forces"][gov_id] / 1e3,
    "gov_force_solver_kN": opt_uq["forces_solver"].get(gov_id, None) and
                           opt_uq["forces_solver"][gov_id] / 1e3,
    "uq_solver_mode": opt_uq["mode_solver"],
    "grid": [{k: g[k] for k in ("botDia", "botCount", "As", "lam_mean",
              "lam_std", "lam_lo", "lam_solver", "flag")} for g in grid],
}
os.makedirs("runs", exist_ok=True)
with open(os.path.join("runs", "design_optimization.json"), "w") as fh:
    json.dump(result, fh, indent=2)
print("\n[demo] wrote runs/design_optimization.json")


# --------------------------------------------------------------------------- #
# 6. figure                                                                   #
# --------------------------------------------------------------------------- #
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9))

# (a) capacity vs steel: surrogate (conformal bars) + solver, constraint, optima
As = np.array([g["As"] for g in grid]) / 1e3            # 10^3 mm^2
order = np.argsort(As)
axL.axhline(1.0, color="0.45", lw=1.0, ls="--", zorder=1)
axL.text(As.max() * 0.99, 1.02, r"$\lambda_f = 1$ (required)", ha="right",
         va="bottom", fontsize=7.5, color="0.35")
axL.errorbar(As, lam_s, yerr=Q90 * np.array([g["lam_std"] for g in grid]),
             fmt="o", ms=3.2, color=C_SUR, ecolor=C_SUR, elinewidth=0.7,
             capsize=1.3, alpha=0.85, zorder=3,
             label="surrogate (90\\% conformal)")
axL.scatter(As, lam_v, s=20, marker="x", color=C_SOL, linewidth=0.9,
            zorder=4, label="reference solver")
flagged = [i for i, g in enumerate(grid) if g["flag"]]
if flagged:
    axL.scatter(As[flagged], lam_s[flagged], s=80, marker="o", facecolor="none",
                edgecolor=C_FLAG, linewidth=1.5, zorder=5,
                label="domain-of-validity flag")
for opt, col, lab in [(opt_point, C_BAD, "point-estimate optimum"),
                      (opt_uq, C_OK, "reliability-aware optimum")]:
    axL.scatter([opt["As"] / 1e3], [opt["lam_mean"]], s=120, marker="*",
                facecolor=col, edgecolor="black", linewidth=0.5, zorder=6, label=lab)
axL.set_xlabel(r"bottom-tie steel area $A_s$  ($10^3\,\mathrm{mm}^2$)")
axL.set_ylabel(r"failure load factor $\lambda_f$")
axL.legend(fontsize=6.6, loc="upper left", framealpha=0.9)
axL.set_title("(a) reinforcement minimisation", fontsize=9)

# (b) member-force state at the UQ optimum: surrogate vs solver
ids = list(MEMBER_IDS)
fs = np.array([opt_uq["forces"][m] for m in ids]) / 1e3
fv = np.array([opt_uq["forces_solver"].get(m, 0.0) for m in ids]) / 1e3
ordf = np.argsort(-np.abs(fv))[:10]                      # 10 largest by |solver|
ids10 = [ids[i] for i in ordf]
x = np.arange(len(ids10))
axR.bar(x - 0.2, fs[ordf], width=0.38, color=C_SUR, label="surrogate", zorder=3)
axR.bar(x + 0.2, fv[ordf], width=0.38, color=C_SOL, alpha=0.85,
        label="reference solver", zorder=3)
axR.axhline(0, color="0.5", lw=0.7)
gi = ids10.index(gov_id) if gov_id in ids10 else None
if gi is not None:
    axR.annotate("governing tie", (gi, fv[ordf][gi]), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.7))
axR.set_xticks(x)
axR.set_xticklabels(ids10, rotation=60, fontsize=6.5, ha="right")
axR.set_ylabel(r"member force at failure  (kN)")
axR.legend(fontsize=7, loc="upper right", framealpha=0.9)
axR.set_title("(b) force state at the optimum", fontsize=9)

fig.tight_layout()
fig.savefig("../figures/optimization.pdf", bbox_inches="tight")
print("[demo] wrote ../figures/optimization.pdf")
