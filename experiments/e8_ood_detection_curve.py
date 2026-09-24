"""E8: the full out-of-domain detection picture.

For the released ensembles: the flag rate on the shells delta = 0.1, 0.2,
0.3 at in-domain alarm rates of 1, 2, 5, 10 and 20 %; the AUROC of the
ensemble spread as an in-/out-of-domain score; and, more usefully, how many
of the shell designs whose error exceeds the in-domain 95th-percentile
error are caught (recall) and what fraction of flagged designs are such
harmful cases (precision). Writes the vector figure ood_detection.pdf.
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (ARCHS, COLOUR, FIG_DIR, LABEL, Log, dump, get_config,
                    load_archetype, load_ensemble, load_json)

torch.set_num_threads(2)
DELTAS = [0.1, 0.2, 0.3]
ALARMS = [1, 2, 5, 10, 20]
log = Log("e8_ood_detection_curve.log")


def sig(d):
    return (tuple(n["id"] for n in d["truss"]["nodes"]),
            tuple((m["id"], m["ni"], m["nj"]) for m in d["truss"]["members"]))


def auroc(pos, neg):
    """Probability that a random out-of-domain score exceeds an in-domain one."""
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def ens(models, theta):
    with torch.no_grad():
        p = np.stack([m(theta)[0].numpy() for m in models])
    return p.mean(0), p.std(0, ddof=1)


res = {}
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    models = load_ensemble(arch, data, cfg)
    te = data.test_idx
    gen = (data.lambda_f[te] < cfg.lambda_max - 1e-3).numpy()
    m_in, s_in = ens(models, data.theta[te])
    e_in = np.abs(m_in - data.lambda_f[te].numpy())[gen]
    s_in = s_in[gen]
    e95 = float(np.percentile(e_in, 95))
    raw = load_json(cfg.dataset_path)["designs"]
    dom = Counter(sig(d) for d in raw if d["archetype"] == arch).most_common(1)[0][0]
    mean, std = data.theta_mean.numpy(), data.theta_std.numpy()
    res[arch] = {"e95": e95, "n_in": int(gen.sum()), "shells": {}}
    log(f"\n=== {arch}: in-domain n={int(gen.sum())}, 95th-pct |err| = {e95:.3f} ===")
    for delta in DELTAS:
        raw_e = load_json(f"../data/dataset_extrap_{delta}.json.gz")["designs"]
        keep = [d for d in raw_e if d["archetype"] == arch and sig(d) == dom]
        th = np.array([[d["params"][k] for k in data.theta_keys] for d in keep], dtype=np.float32)
        lam = np.array([d["labels"]["failureLoadFactor"] for d in keep], dtype=np.float32)
        g = lam < cfg.lambda_max - 1e-3
        m_o, s_o = ens(models, torch.tensor((th - mean) / std))
        e_o = np.abs(m_o - lam)[g]
        s_o = s_o[g]
        harmful = e_o > e95
        row = {"n": int(g.sum()), "auroc": auroc(s_o, s_in),
               "harmful_frac": float(harmful.mean()), "alarm": {}}
        for a in ALARMS:
            tau = float(np.percentile(s_in, 100 - a))
            flag = s_o > tau
            rec = float(flag[harmful].mean()) if harmful.any() else float("nan")
            prec = float(harmful[flag].mean()) if flag.any() else float("nan")
            row["alarm"][a] = {"flag_rate": float(flag.mean()), "recall_harmful": rec,
                               "precision_harmful": prec,
                               "mean_err_flagged": float(e_o[flag].mean()) if flag.any() else float("nan"),
                               "mean_err_unflagged": float(e_o[~flag].mean()) if (~flag).any() else float("nan")}
        res[arch]["shells"][delta] = row
        log(f"delta={delta}: n={row['n']}  AUROC {row['auroc']:.3f}  harmful {100*row['harmful_frac']:.0f}%  " +
            "  ".join(f"[{a}%: flag {100*row['alarm'][a]['flag_rate']:.0f}% rec {100*row['alarm'][a]['recall_harmful']:.0f}% "
                      f"prec {100*row['alarm'][a]['precision_harmful']:.0f}%]" for a in ALARMS))
dump("e8_ood_detection_curve.json", res)

# ---- figure: (a) flag rate vs delta at 5 % alarm, (b) recall of harmful errors vs alarm rate
from figstyle import MARKER, tidy, panel, legend_below, INK2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.6), gridspec_kw={"wspace": 0.34})
handles = []
for arch in ARCHS:
    r = res[arch]
    xs = [0.0] + DELTAS
    ys = [5.0] + [100 * r["shells"][d]["alarm"][5]["flag_rate"] for d in DELTAS]
    h, = a1.plot(xs, ys, "-", marker=MARKER[arch], color=COLOUR[arch], lw=1.6, ms=5,
                 markeredgecolor="white", markeredgewidth=0.7, label=LABEL[arch], zorder=3)
    handles.append(h)
    ys2 = [100 * r["shells"][0.3]["alarm"][a]["recall_harmful"] for a in ALARMS]
    a2.plot(ALARMS, ys2, "-", marker=MARKER[arch], color=COLOUR[arch], lw=1.6, ms=5,
            markeredgecolor="white", markeredgewidth=0.7, zorder=3)
a1.axhline(5, ls="--", lw=0.9, color=INK2, zorder=2)
a1.text(0.305, 7.5, "in-domain alarm rate 5%", fontsize=7, color=INK2, ha="right")
a1.set_xlabel("Distance outside the training box, $\\delta$")
a1.set_ylabel("Designs flagged (%)")
a1.set_xlim(-0.01, 0.31); a1.set_ylim(0, 100)
a1.set_xticks([0, 0.1, 0.2, 0.3])
panel(a1, "a", "Flag rate at a 5% in-domain alarm rate")
a2.set_xscale("log")
a2.set_xticks(ALARMS); a2.set_xticklabels([str(a) for a in ALARMS])
a2.minorticks_off()
a2.set_xlabel("In-domain alarm rate (%)")
a2.set_ylabel("Harmful-error designs caught (%)")
a2.set_ylim(0, 100)
panel(a2, "b", "Recall of harmful errors at $\\delta = 0.3$")
for ax in (a1, a2):
    tidy(ax)
fig.tight_layout(w_pad=2.5)
fig.subplots_adjust(bottom=0.30)
legend_below(fig, handles, [LABEL[a] for a in ARCHS], ncol=4, y=0.02)
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "ood_detection.pdf", bbox_inches="tight")
log(f"wrote {FIG_DIR / 'ood_detection.pdf'}")
log.done()
