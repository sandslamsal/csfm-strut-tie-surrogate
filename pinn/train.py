"""Training for the strut and tie surrogate network.

The network maps a D-region's design parameters directly to its failure load
factor and the member forces at the failure state. Training combines:
  - supervised loss on the failure load factor (the primary target),
  - supervised loss on the failure-state member forces,
  - the discrete nodal-equilibrium residual of the predicted failure state,
    used as an optional physics regulariser.

Usage:
    python train.py --archetype deepBeam
    python train.py --archetype pileCap --epochs 300
"""
from __future__ import annotations

import argparse
import os

import torch

import physics as phys
from config import Config, get_config
from data import ArchetypeData, load_archetype
from model import STMNet


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--archetype", default=Config.archetype)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--w_sup", type=float, default=Config.w_sup)
    p.add_argument("--w_eq", type=float, default=Config.w_eq)
    p.add_argument("--train_frac", type=float, default=Config.train_frac)
    p.add_argument("--dataset_path", default=Config.dataset_path)
    p.add_argument("--device", default=Config.device)
    a = p.parse_args()
    return get_config(archetype=a.archetype, epochs=a.epochs, w_sup=a.w_sup,
                      w_eq=a.w_eq, train_frac=a.train_frac,
                      dataset_path=a.dataset_path, device=a.device)


def batch_loss(model: STMNet, data: ArchetypeData, idx: torch.Tensor,
               cfg: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (failure-load loss, force loss, equilibrium regulariser).

    The failure-load loss supervises the primary target. The force loss
    supervises the failure-state member forces. The equilibrium regulariser
    checks that the network's *own* predicted failure forces balance its
    predicted failure load at every free node.
    """
    theta = data.theta[idx]
    lam_pred, F_pred = model(theta)                       # [B], [B,M]
    lam_true = data.lambda_f[idx]
    F_true = data.failure_force[idx]

    # ---- supervised: failure load factor (primary) -----------------------
    l_lam = ((lam_pred - lam_true) ** 2).mean()

    # ---- supervised: failure-state member forces -------------------------
    # Forces are learned in NORMALISED form F / F0, with F0 the applied
    # reference-load magnitude ||P_ref|| of the design. F0 is known at
    # inference (it is set by the load the user applies), unlike the
    # per-design peak member force, so the head can be de-normalised without
    # the solver. Member forces are O(F0), so the target is O(1) and the
    # head actually trains (raw-Newton targets of O(1e6) do not).
    p_ref = torch.linalg.norm(
        data.ref_load[idx].reshape(idx.shape[0], -1), dim=1).clamp(min=1.0)
    F0 = p_ref.view(-1, 1)
    l_F = ((F_pred - F_true / F0) ** 2).mean()

    # ---- physics regulariser: equilibrium of the predicted failure state -
    # de-normalise the predicted forces before checking nodal balance
    F_phys = F_pred * F0
    c, _ = phys.member_geometry(data.node_xyz[idx], data.ni, data.nj)
    R = phys.equilibrium_residual(F_phys, c, data.ni, data.nj,
                                  data.n_nodes, lam_pred, data.ref_load[idx])
    R_n = R / p_ref.view(-1, 1, 1)
    fm = data.free_mask[idx]
    l_eq = (R_n ** 2 * fm).sum() / fm.sum().clamp(min=1.0)

    return l_lam, l_F, l_eq


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    data = load_archetype(cfg).to(cfg.device)
    # optionally restrict the training split (data-efficiency study)
    if cfg.train_frac < 1.0:
        g = torch.Generator().manual_seed(cfg.seed)
        n = max(8, int(cfg.train_frac * len(data.train_idx)))
        perm = torch.randperm(len(data.train_idx), generator=g)[:n]
        data.train_idx = data.train_idx[perm.to(data.train_idx.device)]
    model = STMNet(len(data.theta_keys), data.n_members, cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    print(f"archetype={cfg.archetype}  designs train/val/test="
          f"{len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)}  "
          f"nodes={data.n_nodes} members={data.n_members} "
          f"theta_dim={len(data.theta_keys)}")

    out = os.path.join(cfg.out_dir, cfg.archetype)
    os.makedirs(out, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        order = data.train_idx[torch.randperm(len(data.train_idx))]
        run_lam = run_eq = 0.0
        nb = 0
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            l_lam, l_F, l_eq = batch_loss(model, data, idx, cfg)
            loss = l_lam + cfg.w_sup * l_F + cfg.w_eq * l_eq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run_lam += l_lam.item()
            run_eq += l_eq.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v_lam, v_F, v_eq = batch_loss(model, data, data.val_idx, cfg)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d}  "
                  f"train_lam={run_lam / max(1, nb):.3e} "
                  f"train_eq={run_eq / max(1, nb):.3e}  "
                  f"val_lam={v_lam.item():.3e} val_eq={v_eq.item():.3e}")
        if v_lam.item() < best_val:
            best_val = v_lam.item()
            torch.save({"state_dict": model.state_dict(),
                        "archetype": cfg.archetype,
                        "theta_keys": data.theta_keys,
                        "n_members": data.n_members,
                        "cfg": cfg.__dict__},
                       os.path.join(out, "model.pt"))

    print(f"done. best val_lam(MSE)={best_val:.3e}  saved -> {out}/model.pt")


if __name__ == "__main__":
    main()
