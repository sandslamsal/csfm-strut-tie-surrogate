"""Shared helpers for the experiments E1-E8.

Every script in this folder imports this module, which puts ../pinn on the
path and makes it the working directory so the released data/checkpoint
paths of the pinn scripts resolve unchanged. Training mirrors pinn/train.py
(AdamW, cosine schedule, gradient clipping, best-validation checkpoint) with
three additions: an optional one-sided hinge for the
non-failing designs (E5), a warm start from another archetype's hidden
layers (E1) and a width override (E6).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINN = HERE.parent / "pinn"
sys.path.insert(0, str(PINN))
os.chdir(PINN)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import physics as phys  # noqa: E402
from config import get_config  # noqa: E402
from data import load_archetype, load_json  # noqa: E402
from model import STMNet  # noqa: E402

ARCHS = ["deepBeam", "hammerhead", "multiColumnBent", "pileCap"]
LABEL = {"deepBeam": "Deep beam", "hammerhead": "Hammerhead",
         "multiColumnBent": "Multi-column bent", "pileCap": "Pile cap",
         "deepBeam2P": "Deep beam, two-point loading"}
COLOUR = {"deepBeam": "#26629E", "hammerhead": "#0E7072",
          "multiColumnBent": "#E47E1C", "pileCap": "#BE342E"}
FIG_DIR = HERE.parent / "figures"


def F0_of(data, idx):
    return torch.linalg.norm(
        data.ref_load[idx].reshape(len(idx), -1), dim=1).clamp(min=1.0)


def losses(model, data, idx, cfg, hinge=False):
    """(failure-load loss, force loss, equilibrium regulariser) on idx."""
    lam_pred, F_pred = model(data.theta[idx])
    lam_true = data.lambda_f[idx]
    F_true = data.failure_force[idx]
    if hinge:
        cens = lam_true >= cfg.lambda_max - 1e-3
        err = torch.where(cens, torch.relu(cfg.lambda_max - lam_pred),
                          lam_pred - lam_true)
        l_lam = (err ** 2).mean()
    else:
        l_lam = ((lam_pred - lam_true) ** 2).mean()
    p_ref = F0_of(data, idx)
    F0 = p_ref.view(-1, 1)
    l_F = ((F_pred - F_true / F0) ** 2).mean()
    F_phys = F_pred * F0
    c, _ = phys.member_geometry(data.node_xyz[idx], data.ni, data.nj)
    R = phys.equilibrium_residual(F_phys, c, data.ni, data.nj, data.n_nodes,
                                  lam_pred, data.ref_load[idx])
    R_n = R / p_ref.view(-1, 1, 1)
    fm = data.free_mask[idx]
    l_eq = (R_n ** 2 * fm).sum() / fm.sum().clamp(min=1.0)
    return l_lam, l_F, l_eq


def train(data, cfg, seed=None, hinge=False, init_state=None,
          train_idx=None, quiet=True):
    """Train one network in memory; returns the best-validation model."""
    seed = cfg.seed if seed is None else seed
    torch.manual_seed(seed)
    model = STMNet(len(data.theta_keys), data.n_members, cfg)
    if init_state is not None:
        own = model.state_dict()
        n_copied = 0
        for k, v in init_state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.clone()
                n_copied += 1
        model.load_state_dict(own)
        if not quiet:
            print(f"    warm start: {n_copied}/{len(own)} tensors copied")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    tr = data.train_idx if train_idx is None else train_idx
    best_val, best_state = float("inf"), None
    for _ in range(cfg.epochs):
        model.train()
        order = tr[torch.randperm(len(tr))]
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            l_lam, l_F, l_eq = losses(model, data, idx, cfg, hinge)
            loss = l_lam + cfg.w_sup * l_F + cfg.w_eq * l_eq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v_lam, _, _ = losses(model, data, data.val_idx, cfg, hinge)
        if v_lam.item() < best_val:
            best_val = v_lam.item()
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


def load_headline(arch, data, cfg):
    m = STMNet(len(data.theta_keys), data.n_members, cfg)
    ck = torch.load(os.path.join(cfg.out_dir, arch, "model.pt"),
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def load_ensemble(arch, data, cfg):
    models = []
    ens = os.path.join(cfg.out_dir, arch, "ensemble")
    for f in sorted(os.listdir(ens)):
        if f.endswith(".pt"):
            m = STMNet(len(data.theta_keys), data.n_members, cfg)
            m.load_state_dict(torch.load(os.path.join(ens, f),
                              map_location="cpu",
                              weights_only=False)["state_dict"])
            m.eval()
            models.append(m)
    return models


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def eq_residual_pct(F_phys, lam, data, idx):
    """Per-design RMS nodal residual over free DOFs, % of applied load."""
    c, _ = phys.member_geometry(data.node_xyz[idx], data.ni, data.nj)
    R = phys.equilibrium_residual(F_phys, c, data.ni, data.nj, data.n_nodes,
                                  lam, data.ref_load[idx])
    F0 = F0_of(data, idx)
    R_n = R / F0.view(-1, 1, 1)
    fm = data.free_mask[idx]
    return torch.sqrt((R_n ** 2 * fm).sum(dim=(1, 2))
                      / fm.sum(dim=(1, 2)).clamp(min=1.0)) * 100.0


def evaluate(model, data, cfg, split="test"):
    """Accuracy, force and classification metrics on one split."""
    idx = {"test": data.test_idx, "val": data.val_idx,
           "train": data.train_idx}[split]
    with torch.no_grad():
        lam, F_norm = model(data.theta[idx])
    lam = lam.numpy()
    true = data.lambda_f[idx].numpy()
    ceil = cfg.lambda_max
    cens = true >= ceil - 1e-3
    gen = ~cens
    out = {"n": int(len(idx)), "n_cens": int(cens.sum())}
    e = lam[gen] - true[gen]
    out["mape"] = float(np.mean(np.abs(e) / np.clip(true[gen], 1e-6, None)) * 100)
    out["rmse"] = float(np.sqrt(np.mean(e ** 2)))
    out["r2"] = float(1 - (e ** 2).sum()
                      / max(((true[gen] - true[gen].mean()) ** 2).sum(), 1e-12))
    # failure / no-failure decision
    pred_cens = lam >= ceil
    out["cls_acc"] = float(np.mean(pred_cens == cens))
    out["fp"] = int(np.sum(cens & ~pred_cens))     # said fails, survives
    out["fn"] = int(np.sum(~cens & pred_cens))     # said survives, fails
    out["hinge"] = float(np.mean(np.clip(ceil - lam[cens], 0, None))) if cens.any() else 0.0
    # forces on genuine-failure designs
    gi = idx[torch.tensor(gen)]
    with torch.no_grad():
        lam_g, Fn_g = model(data.theta[gi])
    F0 = F0_of(data, gi)
    F_phys = Fn_g * F0.view(-1, 1)
    F_true = data.failure_force[gi]
    fp_, ft_ = F_phys.numpy().ravel(), F_true.numpy().ravel()
    out["force_r2"] = float(1 - ((fp_ - ft_) ** 2).sum()
                            / max(((ft_ - ft_.mean()) ** 2).sum(), 1e-12))
    rel = (torch.linalg.norm(F_phys - F_true, dim=1)
           / torch.linalg.norm(F_true, dim=1).clamp(min=1.0))
    out["force_rel"] = float(rel.mean() * 100)
    out["eq"] = float(eq_residual_pct(F_phys, lam_g, data, gi).mean())
    return out


def fmt(s):
    return (f"MAPE {s['mape']:6.2f}%  R2 {s['r2']:.3f}  forceR2 {s['force_r2']:.3f}  "
            f"forceRel {s['force_rel']:5.1f}%  eq {s['eq']:5.2f}%  "
            f"cls {s['cls_acc']:.3f} fp {s['fp']:2d} fn {s['fn']:2d}")


class Log:
    def __init__(self, name):
        self.fh = open(HERE / name, "w")
        self.t0 = time.time()

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()

    def done(self):
        self(f"[elapsed {time.time() - self.t0:.0f} s]")
        self.fh.close()


def dump(name, obj):
    with open(HERE / name, "w") as fh:
        json.dump(obj, fh, indent=1)
