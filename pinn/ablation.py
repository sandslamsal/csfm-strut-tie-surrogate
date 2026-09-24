"""Force-loss and equilibrium-weight ablation.

Tests whether the force-loss weight w_F = 1.0 matters and what the force term
does to the failure-load accuracy. This script trains, in memory
(without overwriting the released checkpoints), one network per (archetype,
weight) setting and reports:

  * how the failure-load accuracy (MAPE, R2) responds to the force weight w_F,
    including w_F = 0 (no force term at all); and
  * how the member-force accuracy (force R2) and the nodal-equilibrium residual
    of the predicted forces respond to w_F and to the equilibrium weight w_eq.

Run:  python ablation.py
"""
from __future__ import annotations

import numpy as np
import torch

import physics as phys
from config import get_config
from data import ArchetypeData, load_archetype
from model import STMNet
from train import batch_loss

ARCHS = ["deepBeam", "hammerhead", "multiColumnBent", "pileCap"]
WF_GRID = [0.0, 0.1, 1.0, 10.0]      # force-loss weight sweep (w_eq = 0)
WEQ_GRID = [0.0, 0.1, 1.0]           # equilibrium-weight sweep (w_F = 1)


def train_inmem(data: ArchetypeData, cfg) -> STMNet:
    torch.manual_seed(cfg.seed)
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    for _ in range(cfg.epochs):
        model.train()
        order = data.train_idx[torch.randperm(len(data.train_idx))]
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            l_lam, l_F, l_eq = batch_loss(model, data, idx, cfg)
            loss = l_lam + cfg.w_sup * l_F + cfg.w_eq * l_eq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate(model: STMNet, data: ArchetypeData, cfg) -> dict:
    idx = data.test_idx
    gen = data.lambda_f[idx] < cfg.lambda_max - 1e-3
    idx = idx[gen]
    with torch.no_grad():
        lam, F_norm = model(data.theta[idx])
    lam = lam.numpy()
    true = data.lambda_f[idx].numpy()
    err = lam - true
    mape = float((np.abs(err) / np.clip(true, 1e-6, None)).mean() * 100)
    r2 = float(1 - (err ** 2).sum() / max(((true - true.mean()) ** 2).sum(), 1e-12))

    F0 = torch.linalg.norm(data.ref_load[idx].reshape(len(idx), -1),
                           dim=1).clamp(min=1.0)
    F_phys = F_norm * F0.view(-1, 1)
    F_true = data.failure_force[idx]
    fp, ft = F_phys.numpy().ravel(), F_true.numpy().ravel()
    force_r2 = float(1 - ((fp - ft) ** 2).sum()
                     / max(((ft - ft.mean()) ** 2).sum(), 1e-12))

    c, _ = phys.member_geometry(data.node_xyz[idx], data.ni, data.nj)
    with torch.no_grad():
        lam_t, _ = model(data.theta[idx])
    R = phys.equilibrium_residual(F_phys, c, data.ni, data.nj, data.n_nodes,
                                  lam_t, data.ref_load[idx])
    R_n = R / F0.view(-1, 1, 1)
    fm = data.free_mask[idx]
    eq = float(torch.sqrt((R_n ** 2 * fm).sum(dim=(1, 2))
                          / fm.sum(dim=(1, 2)).clamp(min=1.0)).mean()) * 100
    return {"mape": mape, "r2": r2, "force_r2": force_r2, "eq": eq}


def main() -> None:
    print("=== force-loss weight w_F (w_eq = 0) ===")
    print(f"{'archetype':16s} {'w_F':>6s} {'MAPE%':>7s} {'lam R2':>7s} "
          f"{'force R2':>9s} {'eq %':>7s}")
    for arch in ARCHS:
        for wf in WF_GRID:
            cfg = get_config(archetype=arch, w_sup=wf, w_eq=0.0)
            data = load_archetype(cfg)
            m = train_inmem(data, cfg)
            s = evaluate(m, data, cfg)
            fr2 = "   n/a" if wf == 0.0 else f"{s['force_r2']:9.3f}"
            print(f"{arch:16s} {wf:6.1f} {s['mape']:7.2f} {s['r2']:7.3f} "
                  f"{fr2} {s['eq']:7.2f}")
        print()

    print("=== equilibrium weight w_eq (w_F = 1) ===")
    print(f"{'archetype':16s} {'w_eq':>6s} {'MAPE%':>7s} {'lam R2':>7s} "
          f"{'force R2':>9s} {'eq %':>7s}")
    for arch in ARCHS:
        for we in WEQ_GRID:
            cfg = get_config(archetype=arch, w_sup=1.0, w_eq=we)
            data = load_archetype(cfg)
            m = train_inmem(data, cfg)
            s = evaluate(m, data, cfg)
            print(f"{arch:16s} {we:6.1f} {s['mape']:7.2f} {s['r2']:7.3f} "
                  f"{s['force_r2']:9.3f} {s['eq']:7.2f}")
        print()


if __name__ == "__main__":
    main()
