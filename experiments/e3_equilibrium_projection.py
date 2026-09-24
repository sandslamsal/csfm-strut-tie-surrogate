"""E3: exact equilibrium by least-squares projection.

For each genuine-failure test design the headline model's predicted force
vector F is projected onto the nodal-equilibrium manifold of its own
predicted failure load: minimise ||F* - F|| subject to A F* = -lam P on the
free DOFs. The script reports the static determinacy of each archetype's
truss (rank of A against the member count), the residual before and after,
the force accuracy before and after, and the size of the correction. A
second projection uses the reference load factor instead of the predicted
one, which separates the force error carried by lam from the error in the
force distribution itself.
"""
import numpy as np
import torch
from common import (ARCHS, F0_of, Log, dump, eq_residual_pct, get_config,
                    load_archetype, load_headline, phys)

torch.set_num_threads(2)
log = Log("e3_equilibrium_projection.log")


def operator(data, d):
    """A [n_free, M] such that A F + lam P_free = residual on free DOFs."""
    c, _ = phys.member_geometry(data.node_xyz[d:d + 1], data.ni, data.nj)
    c = c[0].numpy()                                   # [M,3]
    N, M = data.n_nodes, data.n_members
    A = np.zeros((N * 3, M))
    ni, nj = data.ni.numpy(), data.nj.numpy()
    for m in range(M):
        A[ni[m] * 3:ni[m] * 3 + 3, m] += c[m]
        A[nj[m] * 3:nj[m] * 3 + 3, m] -= c[m]
    free = data.free_mask[d].numpy().ravel() > 0.5
    P = data.ref_load[d].numpy().ravel()
    return A[free], P[free]


def project(F, lam, A, P):
    """Minimum-norm correction onto A F* = -lam P."""
    r = A @ F + lam * P
    dF = -A.T @ np.linalg.pinv(A @ A.T, rcond=1e-10) @ r
    return F + dF


def r2(pred, true):
    return float(1 - ((pred - true) ** 2).sum()
                 / max(((true - true.mean()) ** 2).sum(), 1e-12))


res = {}
for arch in ARCHS:
    cfg = get_config(archetype=arch)
    data = load_archetype(cfg)
    model = load_headline(arch, data, cfg)
    idx = data.test_idx[data.lambda_f[data.test_idx] < cfg.lambda_max - 1e-3]
    with torch.no_grad():
        lam_p, Fn = model(data.theta[idx])
    F0 = F0_of(data, idx)
    F_pred = (Fn * F0.view(-1, 1)).numpy()
    F_true = data.failure_force[idx].numpy()
    lam_p = lam_p.numpy()
    lam_t = data.lambda_f[idx].numpy()
    A0, P0 = operator(data, int(idx[0]))
    rank = int(np.linalg.matrix_rank(A0))
    n_free, M = A0.shape
    F_proj = np.zeros_like(F_pred)
    F_proj_ref = np.zeros_like(F_pred)
    for k, d in enumerate(idx.tolist()):
        A, P = operator(data, d)
        F_proj[k] = project(F_pred[k], lam_p[k], A, P)
        F_proj_ref[k] = project(F_pred[k], lam_t[k], A, P)
    eq_before = eq_residual_pct(torch.tensor(F_pred), torch.tensor(lam_p), data, idx)
    eq_after = eq_residual_pct(torch.tensor(F_proj), torch.tensor(lam_p), data, idx)
    eq_ref = eq_residual_pct(torch.tensor(F_true), torch.tensor(lam_t), data, idx)
    rel = lambda F: float(np.mean(np.linalg.norm(F - F_true, axis=1)  # noqa: E731
                                  / np.clip(np.linalg.norm(F_true, axis=1), 1, None)) * 100)
    corr = float(np.mean(np.linalg.norm(F_proj - F_pred, axis=1)
                         / np.clip(np.linalg.norm(F_pred, axis=1), 1, None)) * 100)
    row = {"n": int(len(idx)), "members": M, "free_dofs": n_free, "rank": rank,
           "determinate": bool(rank == M),
           "eq_ref_labels": float(eq_ref.mean()),
           "eq_before": float(eq_before.mean()), "eq_before_max": float(eq_before.max()),
           "eq_after": float(eq_after.mean()), "eq_after_max": float(eq_after.max()),
           "force_r2_before": r2(F_pred.ravel(), F_true.ravel()),
           "force_r2_after": r2(F_proj.ravel(), F_true.ravel()),
           "force_r2_after_reflam": r2(F_proj_ref.ravel(), F_true.ravel()),
           "force_rel_before": rel(F_pred), "force_rel_after": rel(F_proj),
           "force_rel_after_reflam": rel(F_proj_ref),
           "correction_pct": corr}
    res[arch] = row
    log(f"\n=== {arch}: M={M} members, {n_free} free DOFs, rank(A)={rank} -> "
        f"{'statically determinate' if row['determinate'] else 'indeterminate by ' + str(M - rank)} ===")
    log(f"residual of reference labels      {row['eq_ref_labels']:.4f}%")
    log(f"residual predicted   before/after {row['eq_before']:.3f}% (max {row['eq_before_max']:.2f}%) / "
        f"{row['eq_after']:.2e}% (max {row['eq_after_max']:.2e}%)")
    log(f"force R2             before/after {row['force_r2_before']:.4f} / {row['force_r2_after']:.4f}"
        f"   (with reference lam: {row['force_r2_after_reflam']:.4f})")
    log(f"force rel. L2 error  before/after {row['force_rel_before']:.2f}% / {row['force_rel_after']:.2f}%"
        f"   (with reference lam: {row['force_rel_after_reflam']:.2f}%)")
    log(f"size of the correction ||dF||/||F|| {corr:.2f}%")
dump("e3_equilibrium_projection.json", res)
log.done()
