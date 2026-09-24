/**
 * 3D linear-elastic finite-element solver (8-node hexahedral / voxel mesh).
 *
 * Implements the "linear analysis" reinforcement-layout tool of the CSFM book
 * (§3.4.2): a fast elastic stress field that reveals tension and compression
 * regions, so the designer can place ties along the principal tensile stress
 * trajectories. The continuum domain is voxelised; each filled voxel is one
 * tri-linear hexahedral element.
 */

import { SparseSym, conjugateGradient } from '../math/linalg';

export interface VoxelDomain {
  nx: number; ny: number; nz: number;       // element counts
  dx: number; dy: number; dz: number;       // voxel sizes (mm)
  origin: [number, number, number];
  /** true if voxel (i,j,k) is solid concrete */
  filled: (i: number, j: number, k: number) => boolean;
}

export interface FemLoad {
  /** world position of the load (mm) — applied to nearest node */
  pos: [number, number, number];
  f: [number, number, number];               // N
}

export interface FemSupport {
  /** world position (mm) */
  pos: [number, number, number];
  /** restrained DOFs */
  fix: [boolean, boolean, boolean];
  /** radius (mm) within which nodes are restrained */
  radius: number;
}

export interface FemNodeResult {
  pos: [number, number, number];
  u: [number, number, number];
  /** principal stresses (MPa), sorted s1 >= s2 >= s3 */
  sPrincipal: [number, number, number];
  /** direction of the most tensile principal stress (unit vector) */
  dirTension: [number, number, number];
  /** direction of the most compressive principal stress */
  dirCompression: [number, number, number];
  vonMises: number;
}

export interface FemResult {
  nodes: FemNodeResult[];
  maxTension: number;
  maxCompression: number;
  converged: boolean;
  iterations: number;
  dofCount: number;
}

/* 2-point Gauss */
const G = 1 / Math.sqrt(3);
const GP = [-G, G];

/** Shape-function derivatives wrt natural coords at (xi,eta,zeta). */
function dN(xi: number, eta: number, zeta: number): number[][] {
  // node order: (-,-,-),(+,-,-),(+,+,-),(-,+,-),(-,-,+),(+,-,+),(+,+,+),(-,+,+)
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

/** Hexahedral element stiffness for a regular brick dx*dy*dz. */
export function hexStiffness(dx: number, dy: number, dz: number, E: number, nu: number): number[][] {
  const Ke: number[][] = Array.from({ length: 24 }, () => new Array(24).fill(0));
  // constitutive matrix (isotropic 3D)
  const c = E / ((1 + nu) * (1 - 2 * nu));
  const D = [
    [c * (1 - nu), c * nu, c * nu, 0, 0, 0],
    [c * nu, c * (1 - nu), c * nu, 0, 0, 0],
    [c * nu, c * nu, c * (1 - nu), 0, 0, 0],
    [0, 0, 0, c * (1 - 2 * nu) / 2, 0, 0],
    [0, 0, 0, 0, c * (1 - 2 * nu) / 2, 0],
    [0, 0, 0, 0, 0, c * (1 - 2 * nu) / 2],
  ];
  const detJ = (dx / 2) * (dy / 2) * (dz / 2);
  const invJ = [2 / dx, 2 / dy, 2 / dz];

  for (const xi of GP) for (const eta of GP) for (const zeta of GP) {
    const d = dN(xi, eta, zeta);
    // B matrix 6x24
    const B: number[][] = Array.from({ length: 6 }, () => new Array(24).fill(0));
    for (let n = 0; n < 8; n++) {
      const dNx = d[n][0] * invJ[0];
      const dNy = d[n][1] * invJ[1];
      const dNz = d[n][2] * invJ[2];
      const col = 3 * n;
      B[0][col] = dNx;
      B[1][col + 1] = dNy;
      B[2][col + 2] = dNz;
      B[3][col] = dNy; B[3][col + 1] = dNx;
      B[4][col + 1] = dNz; B[4][col + 2] = dNy;
      B[5][col] = dNz; B[5][col + 2] = dNx;
    }
    // Ke += B^T D B detJ
    const DB: number[][] = Array.from({ length: 6 }, () => new Array(24).fill(0));
    for (let r = 0; r < 6; r++)
      for (let cI = 0; cI < 24; cI++) {
        let sum = 0;
        for (let k = 0; k < 6; k++) sum += D[r][k] * B[k][cI];
        DB[r][cI] = sum;
      }
    for (let a = 0; a < 24; a++)
      for (let b = 0; b < 24; b++) {
        let sum = 0;
        for (let k = 0; k < 6; k++) sum += B[k][a] * DB[k][b];
        Ke[a][b] += sum * detJ;
      }
  }
  return Ke;
}

/** Solve the voxel FE model. */
export function solveFem(
  domain: VoxelDomain,
  loads: FemLoad[],
  supports: FemSupport[],
  E: number,
  nu = 0.2,
): FemResult {
  const { nx, ny, nz, dx, dy, dz, origin } = domain;
  const nnx = nx + 1, nny = ny + 1, nnz = nz + 1;
  const nodeId = (i: number, j: number, k: number) =>
    (k * nny + j) * nnx + i;
  const nNode = nnx * nny * nnz;
  const ndof = 3 * nNode;

  const Ke = hexStiffness(dx, dy, dz, E, nu);
  const K = new SparseSym(ndof);
  const used = new Uint8Array(nNode);

  // element corner offsets matching dN node order
  const off = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
  ];

  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        if (!domain.filled(i, j, k)) continue;
        const en = off.map(([a, b, c]) => nodeId(i + a, j + b, k + c));
        en.forEach((n) => (used[n] = 1));
        for (let a = 0; a < 24; a++) {
          const gA = 3 * en[Math.floor(a / 3)] + (a % 3);
          for (let b = 0; b < 24; b++) {
            const gB = 3 * en[Math.floor(b / 3)] + (b % 3);
            if (gA <= gB) K.add(gA, gB, Ke[a][b]);
          }
        }
      }

  // node world position
  const nodePos = (n: number): [number, number, number] => {
    const i = n % nnx;
    const j = Math.floor(n / nnx) % nny;
    const k = Math.floor(n / (nnx * nny));
    return [origin[0] + i * dx, origin[1] + j * dy, origin[2] + k * dz];
  };

  // load vector
  const F = new Array(ndof).fill(0);
  for (const ld of loads) {
    const n = nearestNode(ld.pos);
    F[3 * n] += ld.f[0];
    F[3 * n + 1] += ld.f[1];
    F[3 * n + 2] += ld.f[2];
  }

  // supports
  const fixed = new Uint8Array(ndof);
  for (const sp of supports) {
    for (let n = 0; n < nNode; n++) {
      if (!used[n]) continue;
      const p = nodePos(n);
      const d = Math.hypot(p[0] - sp.pos[0], p[1] - sp.pos[1], p[2] - sp.pos[2]);
      if (d <= sp.radius) {
        for (let c = 0; c < 3; c++) if (sp.fix[c]) fixed[3 * n + c] = 1;
      }
    }
  }
  // pin unused nodes to keep the system non-singular
  for (let n = 0; n < nNode; n++)
    if (!used[n]) for (let c = 0; c < 3; c++) fixed[3 * n + c] = 1;

  // apply BC by large-penalty / row-zeroing on the diagonal
  const PEN = 1e12;
  for (let g = 0; g < ndof; g++) {
    if (fixed[g]) {
      K.add(g, g, PEN);
      F[g] = 0;
    }
  }

  const { u, iterations, converged } = conjugateGradient(K, F, { tol: 1e-7 });

  function nearestNode(pos: [number, number, number]): number {
    const i = clamp(Math.round((pos[0] - origin[0]) / dx), 0, nx);
    const j = clamp(Math.round((pos[1] - origin[1]) / dy), 0, ny);
    const k = clamp(Math.round((pos[2] - origin[2]) / dz), 0, nz);
    return nodeId(i, j, k);
  }

  // recover stresses at nodes by averaging element contributions
  const nodeStress: number[][] = Array.from({ length: nNode }, () => new Array(6).fill(0));
  const nodeCount = new Array(nNode).fill(0);
  const c = E / ((1 + nu) * (1 - 2 * nu));
  const D = [
    [c * (1 - nu), c * nu, c * nu, 0, 0, 0],
    [c * nu, c * (1 - nu), c * nu, 0, 0, 0],
    [c * nu, c * nu, c * (1 - nu), 0, 0, 0],
    [0, 0, 0, c * (1 - 2 * nu) / 2, 0, 0],
    [0, 0, 0, 0, c * (1 - 2 * nu) / 2, 0],
    [0, 0, 0, 0, 0, c * (1 - 2 * nu) / 2],
  ];
  const invJ = [2 / dx, 2 / dy, 2 / dz];

  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        if (!domain.filled(i, j, k)) continue;
        const en = off.map(([a, b, ci]) => nodeId(i + a, j + b, k + ci));
        const ue: number[] = [];
        en.forEach((n) => { ue.push(u[3 * n], u[3 * n + 1], u[3 * n + 2]); });
        // evaluate strain/stress at element centre
        const d = dN(0, 0, 0);
        const strain = new Array(6).fill(0);
        for (let n = 0; n < 8; n++) {
          const dNx = d[n][0] * invJ[0];
          const dNy = d[n][1] * invJ[1];
          const dNz = d[n][2] * invJ[2];
          const ux = ue[3 * n], uy = ue[3 * n + 1], uz = ue[3 * n + 2];
          strain[0] += dNx * ux;
          strain[1] += dNy * uy;
          strain[2] += dNz * uz;
          strain[3] += dNy * ux + dNx * uy;
          strain[4] += dNz * uy + dNy * uz;
          strain[5] += dNz * ux + dNx * uz;
        }
        const stress = D.map((row) => row.reduce((s, v, idx) => s + v * strain[idx], 0));
        en.forEach((n) => {
          for (let s = 0; s < 6; s++) nodeStress[n][s] += stress[s];
          nodeCount[n]++;
        });
      }

  const nodes: FemNodeResult[] = [];
  let maxTension = 0, maxCompression = 0;
  for (let n = 0; n < nNode; n++) {
    if (!used[n] || nodeCount[n] === 0) continue;
    const sv = nodeStress[n].map((s) => s / nodeCount[n]);
    const { values, vectors } = principalStresses(sv);
    maxTension = Math.max(maxTension, values[0]);
    maxCompression = Math.min(maxCompression, values[2]);
    nodes.push({
      pos: nodePos(n),
      u: [u[3 * n], u[3 * n + 1], u[3 * n + 2]],
      sPrincipal: [values[0], values[1], values[2]],
      dirTension: vectors[0],
      dirCompression: vectors[2],
      vonMises: vonMises(sv),
    });
  }

  return {
    nodes,
    maxTension,
    maxCompression,
    converged,
    iterations,
    dofCount: ndof,
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function vonMises(s: number[]): number {
  const [sx, sy, sz, txy, tyz, tzx] = s;
  return Math.sqrt(
    0.5 *
      ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) +
      3 * (txy * txy + tyz * tyz + tzx * tzx),
  );
}

/** Principal stresses + directions via Jacobi eigenvalue iteration (3x3). */
export function principalStresses(s: number[]): {
  values: [number, number, number];
  vectors: [number, number, number][];
} {
  const A = [
    [s[0], s[3], s[5]],
    [s[3], s[1], s[4]],
    [s[5], s[4], s[2]],
  ];
  let V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let iter = 0; iter < 50; iter++) {
    // find largest off-diagonal
    let p = 0, q = 1, max = Math.abs(A[0][1]);
    if (Math.abs(A[0][2]) > max) { max = Math.abs(A[0][2]); p = 0; q = 2; }
    if (Math.abs(A[1][2]) > max) { max = Math.abs(A[1][2]); p = 1; q = 2; }
    if (max < 1e-9) break;
    const phi = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
    const cs = Math.cos(phi), sn = Math.sin(phi);
    const rot = (M: number[][]) => {
      for (let i = 0; i < 3; i++) {
        const mip = M[i][p], miq = M[i][q];
        M[i][p] = cs * mip + sn * miq;
        M[i][q] = -sn * mip + cs * miq;
      }
    };
    rot(A);
    // A = R^T A R  -> rotate rows too
    for (let j = 0; j < 3; j++) {
      const apj = A[p][j], aqj = A[q][j];
      A[p][j] = cs * apj + sn * aqj;
      A[q][j] = -sn * apj + cs * aqj;
    }
    rot(V);
  }
  const eig = [
    { v: A[0][0], i: 0 },
    { v: A[1][1], i: 1 },
    { v: A[2][2], i: 2 },
  ].sort((a, b) => b.v - a.v);
  const values: [number, number, number] = [eig[0].v, eig[1].v, eig[2].v];
  const vectors = eig.map((e) => {
    const col: [number, number, number] = [V[0][e.i], V[1][e.i], V[2][e.i]];
    const len = Math.hypot(...col) || 1;
    return [col[0] / len, col[1] / len, col[2] / len] as [number, number, number];
  });
  return { values, vectors };
}
