"""Direct experimental validation of the surrogate.

Evaluates the trained pier-cap surrogate on the five Geevar & Menon (2019)
experimental specimens and produces a three-way comparison: measured ultimate
load, the CSFM reference solver, and the surrogate. Writes the comparison
figure ../figures/experimental_validation.tex.

The reference-solver predictions are produced by scripts/validatePiercaps.ts
(deterministic) and are quoted here as constants.

Run (after training the pierCap surrogate):
    python validate_experimental.py
"""
from __future__ import annotations

import json
import os
import torch

from config import get_config
from data import load_archetype
from model import STMNet

DATA = "../validation/piercaps_geevar_menon_2018.json"

# Reference-solver total-load predictions (kN), from scripts/validatePiercaps.ts
SOLVER = {"S1": 1890, "S2": 1994, "S3": 2233, "S4": 2526, "S5": 2771}


def main() -> None:
    cfg = get_config(archetype="pierCap")
    data = load_archetype(cfg)
    ckpt = torch.load(os.path.join(cfg.out_dir, "pierCap", "model.pt"),
                      map_location="cpu", weights_only=False)
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = json.load(open(DATA))
    geom = dataset["geometry"]["reportedFigureDimensions_mm"]
    bands = geom["heightBands"]
    specimens = dataset["specimens"]

    rows = []
    for s in dataset["specimens"]:
        exp_total = s["measured"]["Pu_total_kN"]
        n_bars = (s["reinforcement"]["primary_As1"]["count"]
                  + (s["reinforcement"]["additional_As2"]["count"]
                     if s["reinforcement"]["additional_As2"] else 0))
        # raw design-parameter vector (reference load = measured total)
        raw = {
            "capWidth": geom["topCapWidth"],
            "stemWidth": geom["lowerStemWidth"],
            "capBandHeight": bands[0],
            "taperHeight": bands[1],
            "stemHeight": bands[2],
            "thickness": geom["outOfPlaneThickness_b"],
            "loadPlate": s["loadPlate_lb_mm"],
            "columnLoad": exp_total * 1e3,
            "supportPlate": 150.0,
            "edgeDistance": geom["dimension_a"],
            "mainDia": s["reinforcement"]["primary_As1"]["barDia_mm"],
            "mainCount": n_bars,
            "fck": s["materials"]["concrete"]["fc_MPa"],
            "fy": s["materials"]["mainReinf"]["fy_MPa"],
        }
        vec = torch.tensor([[raw[k] for k in data.theta_keys]],
                           dtype=torch.float32)
        theta = (vec - data.theta_mean) / data.theta_std
        with torch.no_grad():
            lam_f, _ = model(theta)
        surr_total = float(lam_f) * exp_total          # reference = measured
        rows.append((s["id"], exp_total, SOLVER[s["id"]], surr_total))

    # ---- report ----------------------------------------------------------
    print(f"{'spec':<6}{'measured':>10}{'solver':>10}{'surrogate':>11}"
          f"{'exp/solv':>10}{'exp/surr':>10}")
    rs, ru = [], []
    for sid, exp, solv, surr in rows:
        rsolv, rsurr = exp / solv, exp / surr
        rs.append(rsolv)
        ru.append(rsurr)
        print(f"{sid:<6}{exp:>10.0f}{solv:>10.0f}{surr:>11.0f}"
              f"{rsolv:>10.2f}{rsurr:>10.2f}")
    mean = lambda x: sum(x) / len(x)                   # noqa: E731
    print(f"\nexp/solver    mean {mean(rs):.2f}")
    print(f"exp/surrogate mean {mean(ru):.2f}")



if __name__ == "__main__":
    main()
