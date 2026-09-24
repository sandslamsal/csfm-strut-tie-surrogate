/**
 * Topology optimization — CSFM book §3.4.2 ("Reinforcement locations →
 * Topology optimization", Fig. 3.6).
 *
 * Finds the optimal distribution of material inside the given concrete volume
 * for the applied loads: a fraction of the volume is filled, the rest left
 * empty, by iteratively redistributing material so the total strain energy
 * (compliance) is minimised — i.e. the stiffest possible structure. The
 * resulting shape is a truss of struts and ties.
 *
 * Implementation: classic SIMP (Solid Isotropic Material with Penalisation)
 * on the voxel-hexahedral mesh, with an Optimality-Criteria density update and
 * a sensitivity filter for mesh-independence. Each element keeps a relative
 * density 0…1; the final field is classified into compression / tension
 * regions for the book's red / blue visualisation.
 */

import { SparseSym, conjugateGradient } from '../math/linalg';
import { hexStiffness, principalStresses } from './fem3d';
import type { VoxelDomain, FemLoad, FemSupport } from './fem3d';

export interface TopologyElement {
  center: [number, number, number];
  size: [number, number, number];
  density: number;            // 0…1 relative density
  sign: 'C' | 'T' | 'N';      // compression / tension / neutral region
  intensity: number;          // 0…1 relative stress magnitude
}

export interface TopologyResult {
  elements: TopologyElement[];
  volumeFraction: number;
  iterations: number;
  complianceHistory: number[];
  converged: boolean;
}

export interface TopologyOptions {
  volumeFraction: number;     // target filled fraction (0.2 … 0.8)
  iterations?: number;
  penal?: number;
  filterRadius?: number;      // in element widths
}

/* node order matching the hex shape functions of fem3d */
const OFF = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

/** tri-linear shape-function derivatives wrt natural coords */
function dN(xi: number, eta: number, zeta: number): number[][] {
  const s = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
  ];
  return s.map(([a, b, c]) => [
    0.125 * a * (1 + b * eta) * (1 + c * zeta),
    0.125 * b * (1 + a * xi) * (1 + c * zeta),
    0.125 * c * (1 + a * xi) * (1 + b * eta),
  ]);
}

export function runTopology(
  domain: VoxelDomain,
  loads: FemLoad[],
  supports: FemSupport[],
  E0: number,
  nu: number,
  opt: TopologyOptions,
): TopologyResult {
  const { nx, ny, nz, dx, dy, dz, origin } = domain;
  const nnx = nx + 1, nny = ny + 1, nnz = nz + 1;
  const nodeId = (i: number, j: number, k: number) => (k * nny + j) * nnx + i;
  const nNode = nnx * nny * nnz;
  const ndof = 3 * nNode;

  const penal = opt.penal ?? 3;
  const rmin = opt.filterRadius ?? 1.5;
  const Emin = 1e-3;
  const move = 0.2;
  const maxIter = opt.iterations ?? 28;
  const volFrac = Math.min(0.85, Math.max(0.1, opt.volumeFraction));

  // ---- active elements --------------------------------------------------
  const elems: { i: number; j: number; k: number }[] = [];
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++)
        if (domain.filled(i, j, k)) elems.push({ i, j, k });
  const nE = elems.length;
  const eAt = new Map<number, number>();
  elems.forEach((e, idx) => eAt.set((e.k * ny + e.j) * nx + e.i, idx));

  // unit-modulus hex stiffness + element dof tables
  const Ke = hexStiffness(dx, dy, dz, 1, nu);
  const eDofs: number[][] = elems.map((e) => {
    const dofs: number[] = [];
    for (const [a, b, c] of OFF) {
      const n = nodeId(e.i + a, e.j + b, e.k + c);
      dofs.push(3 * n, 3 * n + 1, 3 * n + 2);
    }
    return dofs;
  });

  // ---- load vector ------------------------------------------------------
  const F = new Array(ndof).fill(0);
  const nearest = (p: [number, number, number]) => {
    const i = clamp(Math.round((p[0] - origin[0]) / dx), 0, nx);
    const j = clamp(Math.round((p[1] - origin[1]) / dy), 0, ny);
    const k = clamp(Math.round((p[2] - origin[2]) / dz), 0, nz);
    return nodeId(i, j, k);
  };
  for (const ld of loads) {
    const n = nearest(ld.pos);
    F[3 * n] += ld.f[0]; F[3 * n + 1] += ld.f[1]; F[3 * n + 2] += ld.f[2];
  }

  // ---- boundary conditions ---------------------------------------------
  const used = new Uint8Array(nNode);
  eDofs.forEach((d) => d.forEach((g) => (used[Math.floor(g / 3)] = 1)));
  const nodePos = (n: number): [number, number, number] => {
    const i = n % nnx;
    const j = Math.floor(n / nnx) % nny;
    const k = Math.floor(n / (nnx * nny));
    return [origin[0] + i * dx, origin[1] + j * dy, origin[2] + k * dz];
  };
  const fixed = new Uint8Array(ndof);
  for (const sp of supports) {
    for (let n = 0; n < nNode; n++) {
      if (!used[n]) continue;
      const p = nodePos(n);
      const d = Math.hypot(p[0] - sp.pos[0], p[1] - sp.pos[1], p[2] - sp.pos[2]);
      if (d <= sp.radius)
        for (let c = 0; c < 3; c++) if (sp.fix[c]) fixed[3 * n + c] = 1;
    }
  }
  for (let n = 0; n < nNode; n++)
    if (!used[n]) for (let c = 0; c < 3; c++) fixed[3 * n + c] = 1;

  // ---- sensitivity filter neighbours -----------------------------------
  const fr = Math.ceil(rmin);
  const filt: { idx: number; w: number }[][] = elems.map((e) => {
    const list: { idx: number; w: number }[] = [];
    for (let dk = -fr; dk <= fr; dk++)
      for (let dj = -fr; dj <= fr; dj++)
        for (let di = -fr; di <= fr; di++) {
          const ii = e.i + di, jj = e.j + dj, kk = e.k + dk;
          if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
          const idx = eAt.get((kk * ny + jj) * nx + ii);
          if (idx === undefined) continue;
          const w = rmin - Math.hypot(di, dj, dk);
          if (w > 0) list.push({ idx, w });
        }
    return list;
  });

  // ---- SIMP / OC iteration ---------------------------------------------
  const rho = new Float64Array(nE).fill(volFrac);
  const PEN = 1e12;
  const complianceHistory: number[] = [];
  let uPrev: number[] | undefined;
  let converged = false;
  let iter = 0;

  for (; iter < maxIter; iter++) {
    // assemble penalised global stiffness
    const K = new SparseSym(ndof);
    for (let e = 0; e < nE; e++) {
      const scale = E0 * (Emin + Math.pow(rho[e], penal) * (1 - Emin));
      const d = eDofs[e];
      for (let a = 0; a < 24; a++)
        for (let b = a; b < 24; b++) {
          const v = scale * Ke[a][b];
          if (v !== 0) K.add(d[a], d[b], v);
        }
    }
    for (let g = 0; g < ndof; g++) if (fixed[g]) K.add(g, g, PEN);

    const { u } = conjugateGradient(K, F, { tol: 1e-6, maxIter: 4000, u0: uPrev });
    uPrev = u;

    // element strain energy, compliance and sensitivities
    const dc = new Float64Array(nE);
    let C = 0;
    for (let e = 0; e < nE; e++) {
      const d = eDofs[e];
      let e0 = 0;
      for (let a = 0; a < 24; a++) {
        let row = 0;
        for (let b = 0; b < 24; b++) row += Ke[a][b] * u[d[b]];
        e0 += u[d[a]] * row;
      }
      const dEdrho = E0 * penal * Math.pow(rho[e], penal - 1) * (1 - Emin);
      C += E0 * (Emin + Math.pow(rho[e], penal) * (1 - Emin)) * e0;
      dc[e] = -dEdrho * e0;
    }
    complianceHistory.push(C);

    // sensitivity filtering (mesh-independence)
    const dcf = new Float64Array(nE);
    for (let e = 0; e < nE; e++) {
      let num = 0, den = 0;
      for (const { idx, w } of filt[e]) {
        num += w * rho[idx] * dc[idx];
        den += w;
      }
      dcf[e] = num / Math.max(1e-9, rho[e] * den);
    }

    // Optimality-Criteria density update (bisection on the multiplier)
    let l1 = 1e-9, l2 = 1e9;
    const rhoNew = new Float64Array(nE);
    while ((l2 - l1) / (l1 + l2) > 1e-4) {
      const lm = 0.5 * (l1 + l2);
      let vol = 0;
      for (let e = 0; e < nE; e++) {
        const be = Math.sqrt(Math.max(0, -dcf[e]) / lm);
        let x = rho[e] * be;
        x = Math.min(1, Math.min(rho[e] + move, x));
        x = Math.max(0.001, Math.max(rho[e] - move, x));
        rhoNew[e] = x;
        vol += x;
      }
      if (vol > volFrac * nE) l1 = lm; else l2 = lm;
    }
    let change = 0;
    for (let e = 0; e < nE; e++) {
      change = Math.max(change, Math.abs(rhoNew[e] - rho[e]));
      rho[e] = rhoNew[e];
    }
    if (change < 0.01) { converged = true; iter++; break; }
  }

  // ---- classify compression / tension on the final field ---------------
  const c = E0 / ((1 + nu) * (1 - 2 * nu));
  const D = [
    [c * (1 - nu), c * nu, c * nu, 0, 0, 0],
    [c * nu, c * (1 - nu), c * nu, 0, 0, 0],
    [c * nu, c * nu, c * (1 - nu), 0, 0, 0],
    [0, 0, 0, c * (1 - 2 * nu) / 2, 0, 0],
    [0, 0, 0, 0, c * (1 - 2 * nu) / 2, 0],
    [0, 0, 0, 0, 0, c * (1 - 2 * nu) / 2],
  ];
  const invJ = [2 / dx, 2 / dy, 2 / dz];
  const d0 = dN(0, 0, 0);
  const u = uPrev ?? new Array(ndof).fill(0);

  let maxMag = 1e-9;
  const raw = elems.map((e, idx) => {
    const d = eDofs[idx];
    const strain = [0, 0, 0, 0, 0, 0];
    for (let n = 0; n < 8; n++) {
      const dNx = d0[n][0] * invJ[0];
      const dNy = d0[n][1] * invJ[1];
      const dNz = d0[n][2] * invJ[2];
      const ux = u[d[3 * n]], uy = u[d[3 * n + 1]], uz = u[d[3 * n + 2]];
      strain[0] += dNx * ux;
      strain[1] += dNy * uy;
      strain[2] += dNz * uz;
      strain[3] += dNy * ux + dNx * uy;
      strain[4] += dNz * uy + dNy * uz;
      strain[5] += dNz * ux + dNx * uz;
    }
    const stress = D.map((row) => row.reduce((s, v, i) => s + v * strain[i], 0));
    const { values } = principalStresses(stress);
    const sT = values[0];        // most tensile
    const sC = values[2];        // most compressive
    const mag = Math.max(Math.abs(sT), Math.abs(sC));
    maxMag = Math.max(maxMag, mag);
    const sign: 'C' | 'T' | 'N' =
      mag < 1e-6 ? 'N' : Math.abs(sC) >= Math.abs(sT) ? 'C' : 'T';
    return { e, idx, sign, mag };
  });

  const elements: TopologyElement[] = raw.map(({ e, idx, sign, mag }) => ({
    center: [
      origin[0] + (e.i + 0.5) * dx,
      origin[1] + (e.j + 0.5) * dy,
      origin[2] + (e.k + 0.5) * dz,
    ],
    size: [dx, dy, dz],
    density: rho[idx],
    sign,
    intensity: Math.min(1, mag / maxMag),
  }));

  return {
    elements,
    volumeFraction: volFrac,
    iterations: iter,
    complianceHistory,
    converged,
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
