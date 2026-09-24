"""Differentiable CSFM physics for the strut and tie network.

Direct port of the constitutive laws in src/core/csfm/constitutive.ts and the
discrete nodal equilibrium of src/core/stm/truss.ts. Every function is a pure
tensor operation so gradients flow from the equilibrium residual back to the
network.

Units: stress MPa, length mm, force N, strain dimensionless.
"""
from __future__ import annotations

import torch
from torch import Tensor

# --------------------------------------------------------------------------
# Concrete in compression -- parabola-rectangle law (EN 1992-1-1 Table 3.1)
# --------------------------------------------------------------------------


def parabola_rect_params(fck: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return (eps_c2, eps_cu2, n) as functions of f_ck (MPa), vectorised."""
    low = fck <= 50.0
    hi = torch.clamp(90.0 - fck, min=0.0) / 100.0
    eps_c2 = torch.where(
        low,
        torch.full_like(fck, 0.0020),
        (2.0 + 0.085 * torch.clamp(fck - 50.0, min=0.0) ** 0.53) / 1000.0,
    )
    eps_cu2 = torch.where(
        low, torch.full_like(fck, 0.0035), (2.6 + 35.0 * hi ** 4) / 1000.0)
    n = torch.where(
        low, torch.full_like(fck, 2.0),
        torch.clamp(1.4 + 23.4 * hi ** 4, min=1.4))
    return eps_c2, eps_cu2, n


def eta_fc(fc: Tensor) -> Tensor:
    """Brittleness factor eta_fc = (30 / f_c)^(1/3), capped at 1 (fib MC2010)."""
    return torch.clamp((30.0 / fc) ** (1.0 / 3.0), max=1.0)


def softening_kc2(eps1: Tensor) -> Tensor:
    """Compression-softening factor k_c2 of cracked concrete (CSFM Fig. 3.1e)."""
    kc2 = 1.0 / (0.8 + 140.0 * torch.clamp(eps1, min=0.0))
    return torch.clamp(kc2, max=1.0)


def effective_fcd(fc: Tensor, eps1: Tensor) -> Tensor:
    """Effective compressive strength of cracked concrete -- CSFM Eq. (3.1)."""
    return eta_fc(fc) * softening_kc2(eps1) * fc


def concrete_stress(fck: Tensor, eps_comp: Tensor, fc_eff: Tensor) -> Tensor:
    """Concrete compressive stress (MPa, positive) at compressive strain >= 0.

    fck drives the law shape; fc_eff is the (softened) peak strength.
    """
    eps_c2, _, n = parabola_rect_params(fck)
    e = torch.clamp(eps_comp, min=0.0)
    base = torch.clamp(1.0 - e / eps_c2, min=0.0, max=1.0)
    return fc_eff * (1.0 - base ** n)


# --------------------------------------------------------------------------
# Reinforcing steel -- idealised bilinear law
# --------------------------------------------------------------------------


def steel_stress(fy: Tensor, ft: Tensor, Es: float, eps_u: float,
                 eps: Tensor) -> Tensor:
    """Bare-bar stress (MPa) at average strain ``eps`` for the bilinear model."""
    s = torch.abs(eps)
    eps_y = fy / Es
    Esh = (ft - fy) / (eps_u - eps_y)
    elastic = Es * s
    hard = torch.minimum(fy + Esh * (s - eps_y), ft)
    mag = torch.where(s <= eps_y, elastic, hard)
    return torch.sign(eps) * mag


# --------------------------------------------------------------------------
# Kinematics, member forces and discrete nodal equilibrium
# --------------------------------------------------------------------------


def member_geometry(node_xyz: Tensor, ni: Tensor, nj: Tensor
                     ) -> tuple[Tensor, Tensor]:
    """Return member unit vectors c [B,M,3] and lengths L [B,M]."""
    vec = node_xyz[:, nj, :] - node_xyz[:, ni, :]          # [B,M,3]
    L = torch.linalg.norm(vec, dim=-1).clamp(min=1e-6)     # [B,M]
    c = vec / L.unsqueeze(-1)
    return c, L


def member_strain(u: Tensor, node_xyz: Tensor, ni: Tensor, nj: Tensor
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """Average member strain eps [B,M] from nodal displacements u [B,N,3].

    Also returns the unit vectors c and lengths L for reuse downstream.
    """
    c, L = member_geometry(node_xyz, ni, nj)
    du = u[:, nj, :] - u[:, ni, :]                         # [B,M,3]
    elong = (du * c).sum(-1)                               # [B,M]
    eps = elong / L
    return eps, c, L


def member_force(eps: Tensor, fck: Tensor, fy: Tensor, ft: Tensor,
                 a_concrete: Tensor, a_steel: Tensor, eps1: Tensor,
                 Es: float, eps_u: float) -> Tensor:
    """Member axial force F [B,M] (tension positive) (strut or tie law).

    Compression (eps<0): concrete strut law with compression softening.
    Tension  (eps>=0): reinforcement tie law.
    """
    fc_eff = effective_fcd(fck, eps1)                      # [B,1] or [B,M]
    sig_c = concrete_stress(fck, torch.abs(eps), fc_eff)   # [B,M]
    sig_s = steel_stress(fy, ft, Es, eps_u, eps)           # [B,M]
    f_strut = -a_concrete * sig_c
    f_tie = a_steel * sig_s
    return torch.where(eps < 0.0, f_strut, f_tie)


def equilibrium_residual(force: Tensor, c: Tensor, ni: Tensor, nj: Tensor,
                         n_nodes: int, lam: Tensor, ref_load: Tensor
                         ) -> Tensor:
    """Out-of-balance force R [B,N,3] at every node (discrete nodal equilibrium).

    A member in tension pulls its start node toward its end node (+F c) and
    its end node toward its start node (-F c).
    """
    B = force.shape[0]
    fc_vec = force.unsqueeze(-1) * c                       # [B,M,3]
    R = torch.zeros(B, n_nodes, 3, dtype=force.dtype, device=force.device)
    R.index_add_(1, ni, fc_vec)
    R.index_add_(1, nj, -fc_vec)
    R = R + lam.view(B, 1, 1) * ref_load                   # [B,N,3]
    return R


def estimate_transverse_strain(eps: Tensor) -> Tensor:
    """Estimate the principal tensile strain eps_1 driving compression softening.

    APPROXIMATION: in a discrete truss eps_1 is not available pointwise. Here it
    is taken as the largest positive member strain in the design, applied to
    all struts, and detached so it does not propagate gradients.
    """
    eps1 = torch.clamp(eps, min=0.0).max(dim=1, keepdim=True).values
    return eps1.detach()
