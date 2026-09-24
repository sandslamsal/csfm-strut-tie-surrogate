"""The strut and tie surrogate network.

An MLP maps a D-region's normalised design parameters directly to its failure
load factor and the member forces at the failure state. The displacement field
is no longer predicted: a sweep-and-threshold extraction proved fragile, so the
failure quantity is predicted directly and the equilibrium of the predicted
failure state is used as a physics regulariser (see train.py).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import Config


class Sine(nn.Module):
    """SIREN-style periodic activation."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return torch.sin(x)


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sine()
    raise ValueError(f"unknown activation {name!r}")


class STMNet(nn.Module):
    """MLP surrogate: design parameters -> (failure load factor, failure forces).

    Parameters
    ----------
    theta_dim  : number of design parameters
    n_members  : number of strut-and-tie members for this archetype
    """

    def __init__(self, theta_dim: int, n_members: int, cfg: Config):
        super().__init__()
        self.n_members = n_members
        act = cfg.activation
        layers: list[nn.Module] = [nn.Linear(theta_dim, cfg.hidden_width),
                                   _activation(act)]
        for _ in range(cfg.hidden_layers - 1):
            layers += [nn.Linear(cfg.hidden_width, cfg.hidden_width),
                       _activation(act)]
        # output: 1 failure load factor + n_members failure-state forces
        layers += [nn.Linear(cfg.hidden_width, 1 + n_members)]
        self.net = nn.Sequential(*layers)

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """theta [B, dtheta] -> (lambda_f [B], normalised member forces [B, M]).

        The failure load factor is passed through softplus so it is positive.
        The member-force head is unconstrained (tension positive) and predicts
        forces NORMALISED by the design's applied-load scale F0 = ||P_ref||;
        multiply by F0 to recover physical forces (see train.py). Normalising
        keeps the target O(1) so the head trains -- raw-Newton targets of
        O(1e6) collapse to zero under weight decay.
        """
        out = self.net(theta)
        lam_f = F.softplus(out[:, 0])
        force = out[:, 1:]
        return lam_f, force
