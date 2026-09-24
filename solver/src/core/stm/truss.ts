/**
 * 3D strut-and-tie (space truss) solver.
 *
 * The CSFM and classical strut-and-tie design both rely on a lower-bound
 * (static) truss model of the discontinuity region. This module assembles and
 * solves a 3D pin-jointed truss by the direct stiffness method, returning the
 * axial force in every member (tension positive) and the support reactions.
 */

import { solveDense } from '../math/linalg';
import type { NodeType } from '../codes';

export interface TrussNode {
  id: string;
  x: number; y: number; z: number;          // mm
  fixed: [boolean, boolean, boolean];         // restrained DOFs (x,y,z)
  nodeType?: NodeType;                        // for code checks at the node
  label?: string;
}

export interface TrussMember {
  id: string;
  ni: string;                  // start node id
  nj: string;                  // end node id
  kind: 'strut' | 'tie' | 'auto';
  area: number;                // mm^2 (cross-sectional area for stiffness)
  E: number;                   // MPa
}

export interface TrussLoad {
  node: string;
  fx: number; fy: number; fz: number;        // N
}

export interface MemberResult {
  id: string;
  ni: string; nj: string;
  force: number;               // N, tension positive
  kind: 'strut' | 'tie';       // resolved from sign
  length: number;              // mm
  stress: number;              // MPa (force / area)
  dir: [number, number, number];
}

export interface TrussResult {
  members: MemberResult[];
  reactions: Record<string, [number, number, number]>;
  displacements: Record<string, [number, number, number]>;
  maxStrut: number;            // largest compression magnitude (N)
  maxTie: number;              // largest tension (N)
  stable: boolean;
  message?: string;
}

/** Solve the 3D truss. Throws a descriptive error if the model is unstable. */
export function solveTruss(
  nodes: TrussNode[],
  members: TrussMember[],
  loads: TrussLoad[],
): TrussResult {
  const nNode = nodes.length;
  const ndof = 3 * nNode;
  const index = new Map<string, number>();
  nodes.forEach((n, i) => index.set(n.id, i));

  // global stiffness (dense — STM models are small)
  const K: number[][] = Array.from({ length: ndof }, () => new Array(ndof).fill(0));
  const geom: { L: number; c: number[] }[] = [];

  for (const m of members) {
    const a = index.get(m.ni);
    const b = index.get(m.nj);
    if (a === undefined || b === undefined) {
      throw new Error(`Member ${m.id} references an unknown node.`);
    }
    const na = nodes[a];
    const nb = nodes[b];
    const dx = nb.x - na.x, dy = nb.y - na.y, dz = nb.z - na.z;
    const L = Math.hypot(dx, dy, dz);
    if (L < 1e-6) throw new Error(`Member ${m.id} has zero length.`);
    const c = [dx / L, dy / L, dz / L];
    geom.push({ L, c });
    const k = (m.E * m.area) / L;
    // 6x6 local-to-global axial stiffness
    const dofs = [3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2];
    const T = [-c[0], -c[1], -c[2], c[0], c[1], c[2]];
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        K[dofs[i]][dofs[j]] += k * T[i] * T[j];
      }
    }
  }

  // load vector
  const F = new Array(ndof).fill(0);
  for (const ld of loads) {
    const i = index.get(ld.node);
    if (i === undefined) throw new Error(`Load on unknown node ${ld.node}.`);
    F[3 * i] += ld.fx;
    F[3 * i + 1] += ld.fy;
    F[3 * i + 2] += ld.fz;
  }

  // boundary conditions — collect free DOFs
  const free: number[] = [];
  const fixedDof = new Array(ndof).fill(false);
  nodes.forEach((n, i) => {
    for (let d = 0; d < 3; d++) {
      if (n.fixed[d]) fixedDof[3 * i + d] = true;
      else free.push(3 * i + d);
    }
  });
  if (free.length === 0) throw new Error('Model is fully restrained — no DOFs.');

  // reduced system
  const map = new Map<number, number>();
  free.forEach((g, i) => map.set(g, i));
  const Kr: number[][] = free.map((gi) => free.map((gj) => K[gi][gj]));
  const Fr = free.map((g) => F[g]);

  let ur: number[];
  try {
    ur = solveDense(Kr, Fr);
  } catch (e) {
    throw new Error(
      'Strut-and-tie model is geometrically unstable (a mechanism). ' +
        'Check supports and member connectivity.',
    );
  }

  // full displacement vector
  const U = new Array(ndof).fill(0);
  free.forEach((g, i) => (U[g] = ur[i]));

  // member forces
  const members_out: MemberResult[] = members.map((m, idx) => {
    const a = index.get(m.ni)!;
    const b = index.get(m.nj)!;
    const { L, c } = geom[idx];
    const du = [
      U[3 * b] - U[3 * a],
      U[3 * b + 1] - U[3 * a + 1],
      U[3 * b + 2] - U[3 * a + 2],
    ];
    const elong = du[0] * c[0] + du[1] * c[1] + du[2] * c[2];
    const force = ((m.E * m.area) / L) * elong; // tension positive
    return {
      id: m.id,
      ni: m.ni,
      nj: m.nj,
      force,
      kind: force >= 0 ? 'tie' : 'strut',
      length: L,
      stress: force / m.area,
      dir: [c[0], c[1], c[2]],
    };
  });

  // reactions: R = K u - F at fixed DOFs
  const reactions: Record<string, [number, number, number]> = {};
  const displacements: Record<string, [number, number, number]> = {};
  nodes.forEach((n, i) => {
    const r: [number, number, number] = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const g = 3 * i + d;
      if (fixedDof[g]) {
        let s = -F[g];
        for (let c = 0; c < ndof; c++) s += K[g][c] * U[c];
        r[d] = s;
      }
    }
    if (n.fixed.some((x) => x)) reactions[n.id] = r;
    displacements[n.id] = [U[3 * i], U[3 * i + 1], U[3 * i + 2]];
  });

  let maxStrut = 0, maxTie = 0;
  for (const m of members_out) {
    if (m.force < 0) maxStrut = Math.max(maxStrut, -m.force);
    else maxTie = Math.max(maxTie, m.force);
  }

  return {
    members: members_out,
    reactions,
    displacements,
    maxStrut,
    maxTie,
    stable: true,
  };
}
