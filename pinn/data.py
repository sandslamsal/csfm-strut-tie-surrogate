"""Dataset loader for the strut and tie surrogate.

Reads the JSON produced by scripts/exportDataset.ts and, for one archetype,
builds the fixed strut-and-tie topology plus per-design tensors. Within an
archetype the node order and member incidence are identical across designs;
this is asserted on load.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from config import Config


@dataclass
class ArchetypeData:
    archetype: str
    # ---- shared topology --------------------------------------------------
    node_ids: list[str]
    member_ids: list[str]
    ni: Tensor              # [M] long  -- start-node index of each member
    nj: Tensor              # [M] long  -- end-node index of each member
    n_nodes: int
    n_members: int
    theta_keys: list[str]   # ordered design-parameter names
    # ---- per-design tensors  (first dim = design) ------------------------
    theta: Tensor           # [D, dtheta]  normalised design parameters
    theta_mean: Tensor      # [dtheta]
    theta_std: Tensor       # [dtheta]
    node_xyz: Tensor        # [D, N, 3]
    free_mask: Tensor       # [D, N, 3]   1 where the DOF is free, 0 where fixed
    a_concrete: Tensor      # [D, M]
    a_steel: Tensor         # [D, M]
    bar_dia: Tensor         # [D, M]
    ref_load: Tensor        # [D, N, 3]   reference nodal load set
    fck: Tensor             # [D]
    fy: Tensor              # [D]
    static_force: Tensor    # [D, M]      member forces at the static STM solve
    lambda_f: Tensor        # [D]         ground-truth failure load factor
    # ---- supervision targets: CSFM member states along the load path -----
    state_lambda: Tensor    # [D, S]      load factors of the sampled states
    state_strain: Tensor    # [D, S, M]   target member strains (CSFM oracle)
    state_mask: Tensor      # [D, S]      1 for a real state, 0 for padding
    n_states: int           # S — max number of states per design
    failure_force: Tensor   # [D, M]      member forces at the CSFM failure state
    # ---- splits (index arrays into the design dimension) -----------------
    train_idx: Tensor
    val_idx: Tensor
    test_idx: Tensor

    def to(self, device: str) -> "ArchetypeData":
        for f in ("ni", "nj", "theta", "theta_mean", "theta_std", "node_xyz",
                  "free_mask", "a_concrete", "a_steel", "bar_dia", "ref_load",
                  "fck", "fy", "static_force", "lambda_f",
                  "state_lambda", "state_strain", "state_mask",
                  "failure_force",
                  "train_idx", "val_idx", "test_idx"):
            setattr(self, f, getattr(self, f).to(device))
        return self


def load_json(path: str):
    """Read a JSON file, transparently decompressing ``*.json.gz``."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def _topology_sig(d: dict) -> tuple:
    """A hashable signature of a design's strut-and-tie topology."""
    return (tuple(n["id"] for n in d["truss"]["nodes"]),
            tuple((m["id"], m["ni"], m["nj"]) for m in d["truss"]["members"]))


def load_archetype(cfg: Config) -> ArchetypeData:
    raw = load_json(cfg.dataset_path)
    designs = [d for d in raw["designs"] if d["archetype"] == cfg.archetype]
    if not designs:
        raise ValueError(f"no designs for archetype {cfg.archetype!r}")

    # The MLP needs one fixed topology. Most archetypes produce a single
    # topology; a few (e.g. hammerhead) flip a diagonal for some geometries.
    # Keep the dominant topology and drop the rest, reporting the count.
    from collections import Counter
    sigs = Counter(_topology_sig(d) for d in designs)
    dominant, n_dom = sigs.most_common(1)[0]
    if len(sigs) > 1:
        print(f"[data] {cfg.archetype}: {len(sigs)} topologies; keeping the "
              f"dominant {n_dom}/{len(designs)} designs, dropping "
              f"{len(designs) - n_dom}.")
    designs = [d for d in designs if _topology_sig(d) == dominant]

    # ---- shared topology, taken from the first kept design ---------------
    d0 = designs[0]
    node_ids = [n["id"] for n in d0["truss"]["nodes"]]
    member_ids = [m["id"] for m in d0["truss"]["members"]]
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    ni = [node_index[m["ni"]] for m in d0["truss"]["members"]]
    nj = [node_index[m["nj"]] for m in d0["truss"]["members"]]
    theta_keys = sorted(d0["params"].keys())
    N, M = len(node_ids), len(member_ids)
    S = max(len(d["labels"]["csfmStates"]) for d in designs)

    theta, node_xyz, free_mask = [], [], []
    a_conc, a_steel, bar_dia, ref_load = [], [], [], []
    fck, fy, static_f, lam_f = [], [], [], []
    st_lam, st_str, st_msk = [], [], []
    fail_f = []

    for d in designs:
        tn = d["truss"]["nodes"]
        tm = d["truss"]["members"]
        # topology must match the reference design exactly
        assert [n["id"] for n in tn] == node_ids, "node order mismatch"
        assert [m["id"] for m in tm] == member_ids, "member order mismatch"

        theta.append([d["params"][k] for k in theta_keys])
        node_xyz.append([[n["x"], n["y"], n["z"]] for n in tn])
        free_mask.append([[0.0 if n["fixed"][k] else 1.0 for k in range(3)]
                          for n in tn])
        a_conc.append([m["concreteArea"] for m in tm])
        a_steel.append([m["steelArea"] for m in tm])
        bar_dia.append([m["barDia"] for m in tm])

        rl = np.zeros((N, 3), dtype=np.float32)
        for ld in d["truss"]["loads"]:
            rl[node_index[ld["node"]]] += [ld["fx"], ld["fy"], ld["fz"]]
        ref_load.append(rl)

        fck.append(d["params"]["fck"])
        fy.append(d["params"]["fy"])
        mf = d["labels"]["memberForces"]
        static_f.append([mf.get(mid, 0.0) for mid in member_ids])
        lam_f.append(d["labels"]["failureLoadFactor"])

        # CSFM member states sampled along the load path (padded to S)
        states = d["labels"]["csfmStates"]
        slam = np.zeros(S, dtype=np.float32)
        sstr = np.zeros((S, M), dtype=np.float32)
        smsk = np.zeros(S, dtype=np.float32)
        for k, stt in enumerate(states):
            slam[k] = stt["loadFactor"]
            smsk[k] = 1.0
            for mi, mid in enumerate(member_ids):
                sstr[k, mi] = stt["strains"].get(mid, 0.0)
        st_lam.append(slam)
        st_str.append(sstr)
        st_msk.append(smsk)
        # member forces at the CSFM failure state (last sampled state)
        ffrc = states[-1]["forces"] if states else {}
        fail_f.append([ffrc.get(mid, 0.0) for mid in member_ids])

    t = lambda x: torch.tensor(np.asarray(x, dtype=np.float32))   # noqa: E731
    theta_t = t(theta)
    theta_mean = theta_t.mean(0)
    theta_std = theta_t.std(0).clamp(min=1e-8)
    theta_n = (theta_t - theta_mean) / theta_std

    # ---- design-level train/val/test split -------------------------------
    D = len(designs)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(D, generator=g)
    n_tr = int(cfg.split[0] * D)
    n_va = int(cfg.split[1] * D)
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr + n_va]
    test_idx = perm[n_tr + n_va:]

    return ArchetypeData(
        archetype=cfg.archetype,
        node_ids=node_ids, member_ids=member_ids,
        ni=torch.tensor(ni, dtype=torch.long),
        nj=torch.tensor(nj, dtype=torch.long),
        n_nodes=N, n_members=M, theta_keys=theta_keys,
        theta=theta_n, theta_mean=theta_mean, theta_std=theta_std,
        node_xyz=t(node_xyz), free_mask=t(free_mask),
        a_concrete=t(a_conc), a_steel=t(a_steel), bar_dia=t(bar_dia),
        ref_load=t(ref_load), fck=t(fck), fy=t(fy),
        static_force=t(static_f), lambda_f=t(lam_f),
        state_lambda=t(st_lam), state_strain=t(st_str), state_mask=t(st_msk),
        n_states=S, failure_force=t(fail_f),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
