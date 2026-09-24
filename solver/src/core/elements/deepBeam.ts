/**
 * Shared parallel-/variable-chord deep-beam strut-and-tie generator.
 *
 * Hammerhead bent caps and multi-column bent caps are both deep members
 * (D-regions over their whole length): a top chord, a bottom chord, verticals
 * carrying the applied loads to the chords, and diagonal struts carrying shear
 * toward the supports. The chord elevations are given as functions of x so the
 * truss follows a sloped soffit (hammerhead) and stays inside the concrete.
 */

import type { TrussNode, TrussMember, TrussLoad } from '../stm/truss';

export interface DeepBeamPanelLoad {
  x: number;     // mm (model X)
  P: number;     // N, downward magnitude (applied at top chord)
  label: string;
}

export interface DeepBeamSupport {
  x: number;     // mm
  width: number; // bearing / column width (mm)
  label: string;
}

export interface DeepBeamInput {
  /** top-chord elevation as a function of x (mm) */
  topY: (x: number) => number;
  /** bottom-chord elevation as a function of x (mm) */
  botY: (x: number) => number;
  width: number;          // out-of-plane width (mm)
  loads: DeepBeamPanelLoad[];
  supports: DeepBeamSupport[];
  barDiameter: number;
}

export interface DeepBeamTruss {
  nodes: TrussNode[];
  members: TrussMember[];
  loads: TrussLoad[];
  panelX: number[];
}

const Ec = 30000;   // MPa, stiffness only
const Es = 200000;

/** Build the deep-beam truss with chords following the given elevations. */
export function buildDeepBeamTruss(input: DeepBeamInput): DeepBeamTruss {
  // panel points = union of load and support x-positions
  const xs = new Set<number>();
  input.loads.forEach((l) => xs.add(round(l.x)));
  input.supports.forEach((s) => xs.add(round(s.x)));
  const panelX = [...xs].sort((a, b) => a - b);
  const n = panelX.length;

  const nodes: TrussNode[] = [];
  const top = (i: number) => `T${i}`;
  const bot = (i: number) => `B${i}`;

  panelX.forEach((x, i) => {
    nodes.push({
      id: top(i), x, y: input.topY(x), z: 0,
      fixed: [false, false, true], label: `Top ${i}`,
    });
    nodes.push({
      id: bot(i), x, y: input.botY(x), z: 0,
      fixed: [false, false, true], label: `Bottom ${i}`,
    });
  });

  const members: TrussMember[] = [];
  const barArea = (Math.PI / 4) * input.barDiameter ** 2;
  const chordArea = 8 * barArea;
  const meanDepth = Math.max(
    100,
    avg(panelX.map((x) => input.topY(x) - input.botY(x))),
  );
  const strutArea = input.width * meanDepth * 0.25;

  for (let i = 0; i < n - 1; i++) {
    members.push({ id: `TC${i}`, ni: top(i), nj: top(i + 1), kind: 'auto', area: chordArea, E: Es });
    members.push({ id: `BC${i}`, ni: bot(i), nj: bot(i + 1), kind: 'auto', area: chordArea, E: Es });
  }
  for (let i = 0; i < n; i++) {
    members.push({ id: `V${i}`, ni: top(i), nj: bot(i), kind: 'auto', area: strutArea, E: Ec });
  }
  // diagonals — one per panel, sloping toward the nearest support
  const supX = input.supports.map((s) => s.x);
  for (let i = 0; i < n - 1; i++) {
    const xc = (panelX[i] + panelX[i + 1]) / 2;
    const nearest = supX.reduce((a, b) => (Math.abs(b - xc) < Math.abs(a - xc) ? b : a), supX[0]);
    if (nearest >= xc) {
      members.push({ id: `D${i}`, ni: top(i), nj: bot(i + 1), kind: 'strut', area: strutArea, E: Ec });
    } else {
      members.push({ id: `D${i}`, ni: top(i + 1), nj: bot(i), kind: 'strut', area: strutArea, E: Ec });
    }
  }

  // ---- supports ----------------------------------------------------------
  input.supports.forEach((s, si) => {
    let best = 0, bd = Infinity;
    panelX.forEach((x, i) => {
      const d = Math.abs(x - s.x);
      if (d < bd) { bd = d; best = i; }
    });
    const node = nodes.find((nd) => nd.id === bot(best))!;
    node.fixed = [si === 0, true, true];
    node.label = s.label;
  });
  if (!nodes.some((nd) => nd.fixed[0])) {
    const b0 = nodes.find((nd) => nd.id === bot(0));
    if (b0) b0.fixed = [true, b0.fixed[1], true];
  }

  // ---- loads -------------------------------------------------------------
  const loads: TrussLoad[] = input.loads.map((l) => {
    let best = 0, bd = Infinity;
    panelX.forEach((x, i) => {
      const d = Math.abs(x - l.x);
      if (d < bd) { bd = d; best = i; }
    });
    return { node: top(best), fx: 0, fy: -l.P, fz: 0 };
  });

  return { nodes, members, loads, panelX };
}

function round(v: number): number {
  return Math.round(v * 10) / 10;
}
function avg(a: number[]): number {
  return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
}
