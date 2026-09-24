"""Configuration for the strut and tie surrogate network.

A single archetype is trained at a time: within an archetype the strut-and-tie
topology is fixed (only coordinates, section areas and loads vary), so the MLP
has a fixed input/output size. Train one model per archetype.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    dataset_path: str = "../data/dataset.json.gz"
    archetype: str = "deepBeam"  # deepBeam | hammerhead | multiColumnBent | pileCap
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train/val/test
    seed: int = 20260517

    # ---- model ------------------------------------------------------------
    hidden_layers: int = 6
    hidden_width: int = 128
    activation: str = "tanh"  # tanh | sin (SIREN-style)

    # ---- training ---------------------------------------------------------
    epochs: int = 400
    train_frac: float = 1.0       # fraction of the training split to use
    batch_size: int = 64          # designs per batch
    lambda_per_design: int = 8    # load factors sampled per design per batch
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    # ---- L-BFGS fine-tuning (kept for reference; 0 = disabled) -----------
    lbfgs_steps: int = 0           # disabled: line search stalls on the
    lbfgs_lambda_grid: int = 6     #           non-smooth constitutive laws

    # ---- loss weights -----------------------------------------------------
    w_sup: float = 1.0            # supervised failure-force loss weight (w_F)
    w_eq: float = 0.0             # equilibrium term — unused (kept for record)
    w_reg: float = 1e-5           # unused

    # ---- load-factor range ------------------------------------------------
    lambda_min: float = 0.05
    lambda_max: float = 3.0       # solver search ceiling (MAX_LF in nonlinearCsfm.ts)

    # ---- failure-load extraction (evaluate.py) ---------------------------
    sweep_steps: int = 120        # lambda grid resolution for lambda_f search

    # ---- runtime ----------------------------------------------------------
    device: str = "cpu"           # "cuda" if available
    out_dir: str = "runs"

    # ---- material constants (must match scripts/exportDataset.ts) --------
    steel_Es: float = 200000.0
    steel_eps_u: float = 0.08
    steel_ft_factor: float = 1.2  # f_t = ft_factor * f_y

    archetypes: tuple[str, ...] = field(
        default=("deepBeam", "hammerhead", "multiColumnBent", "pileCap"))


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field: {k}")
        setattr(cfg, k, v)
    return cfg
