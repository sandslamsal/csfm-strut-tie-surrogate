"""Member-force accuracy and nodal-equilibrium check.

A surrogate that predicts strut-and-tie member forces is only interpretable if
those forces are (a) close to the reference CSFM forces and (b) mechanically
consistent -- i.e. they balance the applied load at every free node. This script
reports, per archetype on the held-out genuine-failure test designs:

  * member-force accuracy: pooled R2 and the per-design relative L2 error
    ||F_pred - F_ref|| / ||F_ref|| between the predicted and reference
    failure-state member forces;
  * the discrete nodal-equilibrium residual ||R|| (normalised by the applied
    reference-load magnitude, RMS over free DOFs) of the predicted forces, and
    -- as a baseline -- of the reference CSFM forces themselves.

The released models were trained with the equilibrium weight w_eq = 0, so the
predicted-force equilibrium reported here is an *emergent* property of matching
the reference forces, not something imposed.

Run:  python force_equilibrium.py
"""
from __future__ import annotations

import numpy as np
import torch

import physics as phys
from config import get_config
from data import load_archetype
from model import STMNet

ARCHS = ["deepBeam", "hammerhead", "multiColumnBent", "pileCap"]


def equilibrium_rms(force, lam, data, idx):
    """RMS nodal-equilibrium residual over free DOFs, normalised by load."""
    c, _ = phys.member_geometry(data.node_xyz[idx], data.ni, data.nj)
    R = phys.equilibrium_residual(force, c, data.ni, data.nj,
                                  data.n_nodes, lam, data.ref_load[idx])
    p_ref = torch.linalg.norm(
        data.ref_load[idx].reshape(len(idx), -1), dim=1).clamp(min=1.0)
    R_n = R / p_ref.view(-1, 1, 1)
    fm = data.free_mask[idx]
    # per-design RMS over free DOFs
    num = (R_n ** 2 * fm).sum(dim=(1, 2))
    den = fm.sum(dim=(1, 2)).clamp(min=1.0)
    return torch.sqrt(num / den)


def main() -> None:
    print(f"{'archetype':16s} {'n':>4s} {'force R2':>9s} {'relL2 %':>9s} "
          f"{'eq_pred %':>10s} {'eq_ref %':>9s}")
    rows = {}
    for arch in ARCHS:
        cfg = get_config(archetype=arch)
        data = load_archetype(cfg)
        model = STMNet(len(data.theta_keys), data.n_members, cfg)
        ckpt = torch.load(f"runs/{arch}/model.pt", map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        idx = data.test_idx
        gen = data.lambda_f[idx] < cfg.lambda_max - 1e-3
        idx = idx[gen]
        with torch.no_grad():
            lam_pred, F_pred_norm = model(data.theta[idx])
        # de-normalise the predicted forces by the applied-load scale ||P_ref||
        F0 = torch.linalg.norm(
            data.ref_load[idx].reshape(len(idx), -1), dim=1).clamp(min=1.0)
        F_pred = F_pred_norm * F0.view(-1, 1)
        F_true = data.failure_force[idx]
        lam_true = data.lambda_f[idx]

        # member-force accuracy (pooled over designs x members)
        fp = F_pred.numpy().ravel()
        ft = F_true.numpy().ravel()
        ss_res = float(((fp - ft) ** 2).sum())
        ss_tot = float(((ft - ft.mean()) ** 2).sum())
        force_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rel_l2 = float((torch.linalg.norm(F_pred - F_true, dim=1)
                        / torch.linalg.norm(F_true, dim=1).clamp(min=1.0))
                       .mean()) * 100

        eq_pred = float(equilibrium_rms(F_pred, lam_pred, data, idx).mean()) * 100
        eq_ref = float(equilibrium_rms(F_true, lam_true, data, idx).mean()) * 100

        print(f"{arch:16s} {len(idx):4d} {force_r2:9.3f} {rel_l2:9.2f} "
              f"{eq_pred:10.2f} {eq_ref:9.2f}")
        rows[arch] = {"n": len(idx), "force_r2": force_r2, "rel_l2": rel_l2,
                      "eq_pred": eq_pred, "eq_ref": eq_ref}

    import json, os
    os.makedirs("runs", exist_ok=True)
    json.dump(rows, open("runs/force_equilibrium.json", "w"), indent=2)
    print("\nwrote runs/force_equilibrium.json")


if __name__ == "__main__":
    main()
