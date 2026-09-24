/**
 * 3D Pile Cap element module.
 *
 * Builds a spatial strut-and-tie model for a rectangular pile cap supporting a
 * single column on a rectangular grid of piles: a compression strut runs from
 * the column nodal zone to every pile head, and a bottom tie grid carries the
 * horizontal thrust (classic 3D D-region model — see CSFM book §6.5, pier caps,
 * and the pile-cap STM literature).
 */

import type { ElementModel, BarGroup } from './types';
import { PALETTE } from './types';
import type { TrussNode, TrussMember, TrussLoad } from '../stm/truss';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface PileCapParams {
  capLength: number;      // X (mm)
  capWidth: number;       // Z (mm)
  capDepth: number;       // Y, thickness (mm)
  columnWidth: number;    // X (mm)
  columnDepth: number;    // Z (mm)
  columnShape: 'rectangular' | 'circular';
  pileCols: number;       // piles along X
  pileRows: number;       // piles along Z
  pileDiameter: number;   // pile size — diameter / side (mm)
  pileShape: 'square' | 'circular' | 'H';
  edgeDistance: number;   // pile-centre to cap edge (mm)
  columnLoad: number;     // factored axial load (N, +ve compression)
  momentX: number;        // factored moment about X (N*mm)
  momentZ: number;        // factored moment about Z (N*mm)
  // reinforcement
  bottomX: BarGroup;      // bottom bars running in X (spaced along Z)
  bottomZ: BarGroup;      // bottom bars running in Z (spaced along X)
  /** plane the continuum CSFM slice is taken in: 'x' = X–Y, 'z' = Z–Y */
  csfmAxis: 'x' | 'z';
}

export function defaultPileCapParams(): PileCapParams {
  return {
    capLength: 3000,
    capWidth: 3000,
    capDepth: 1200,
    columnWidth: 600,
    columnDepth: 600,
    columnShape: 'rectangular',
    pileCols: 2,
    pileRows: 2,
    pileDiameter: 450,
    pileShape: 'circular',
    edgeDistance: 600,
    columnLoad: 6000e3,     // 6000 kN
    momentX: 0,
    momentZ: 0,
    bottomX: { dia: 25, spacing: 180 },
    bottomZ: { dia: 25, spacing: 180 },
    csfmAxis: 'x',
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

interface Pile {
  id: string;
  x: number;
  z: number;
  reaction: number;       // factored vertical reaction (N)
}

export function buildPileCap(p: PileCapParams): ElementModel {
  const notes: string[] = [];
  const D = p.capDepth;
  const nPile = p.pileCols * p.pileRows;

  // ---- pile layout (centred) -------------------------------------------
  const spanX = p.capLength - 2 * p.edgeDistance;
  const spanZ = p.capWidth - 2 * p.edgeDistance;
  const dxPile = p.pileCols > 1 ? spanX / (p.pileCols - 1) : 0;
  const dzPile = p.pileRows > 1 ? spanZ / (p.pileRows - 1) : 0;

  const piles: Pile[] = [];
  // second moments for moment distribution of reactions
  let sumX2 = 0, sumZ2 = 0;
  const coords: { x: number; z: number }[] = [];
  for (let r = 0; r < p.pileRows; r++) {
    for (let c = 0; c < p.pileCols; c++) {
      const x = p.pileCols > 1 ? -spanX / 2 + c * dxPile : 0;
      const z = p.pileRows > 1 ? -spanZ / 2 + r * dzPile : 0;
      coords.push({ x, z });
      sumX2 += x * x;
      sumZ2 += z * z;
    }
  }
  coords.forEach((co, i) => {
    // reaction with linear moment distribution (factored)
    let R = p.columnLoad / nPile;
    if (sumX2 > 0) R += (p.momentZ * co.x) / sumX2;
    if (sumZ2 > 0) R += (p.momentX * co.z) / sumZ2;
    piles.push({ id: `P${i + 1}`, x: co.x, z: co.z, reaction: R });
  });

  const minR = Math.min(...piles.map((pl) => pl.reaction));
  if (minR < 0) {
    notes.push(
      `Net tension (uplift) computed at one or more piles (min reaction ` +
        `${(minR / 1e3).toFixed(0)} kN). Provide tension piles or revise geometry.`,
    );
  }

  // ---- truss nodes ------------------------------------------------------
  const nodes: TrussNode[] = [];
  // apex / column node at top centre
  const apex: TrussNode = {
    id: 'COL',
    x: 0, y: D, z: 0,
    fixed: [false, false, false],
    nodeType: 'CCC',
    label: 'Column nodal zone',
  };
  nodes.push(apex);
  // pile nodes at bottom
  piles.forEach((pl) => {
    nodes.push({
      id: pl.id,
      x: pl.x, y: 0, z: pl.z,
      fixed: [false, true, false], // vertical reaction
      nodeType: 'CCT',
      label: `Pile ${pl.id}`,
    });
  });

  // ---- members ----------------------------------------------------------
  const members: TrussMember[] = [];
  // strut bearing area follows the pile shape (circular / square / H-section)
  const strutArea =
    p.pileShape === 'square' ? p.pileDiameter ** 2 :
    p.pileShape === 'H' ? 0.42 * p.pileDiameter ** 2 :
    (Math.PI / 4) * p.pileDiameter ** 2;
  const Ec = 30000; // nominal MPa (stiffness only)
  const Es = 200000;
  // tie direction map: 'X' or 'Z'
  const tieDir = new Map<string, 'X' | 'Z'>();

  // struts: column -> each pile
  piles.forEach((pl) => {
    members.push({
      id: `S-${pl.id}`,
      ni: 'COL',
      nj: pl.id,
      kind: 'strut',
      area: strutArea,
      E: Ec,
    });
  });

  // bottom tie grid: connect adjacent piles along X and Z
  const pileAt = (r: number, c: number) => `P${r * p.pileCols + c + 1}`;
  const tieAreaNominal = 8 * barArea(p.bottomX.dia);
  for (let r = 0; r < p.pileRows; r++) {
    for (let c = 0; c < p.pileCols; c++) {
      if (c < p.pileCols - 1) {
        const id = `T-${pileAt(r, c)}-${pileAt(r, c + 1)}`;
        members.push({
          id, ni: pileAt(r, c), nj: pileAt(r, c + 1),
          kind: 'tie', area: tieAreaNominal, E: Es,
        });
        tieDir.set(id, 'X');
      }
      if (r < p.pileRows - 1) {
        const id = `T-${pileAt(r, c)}-${pileAt(r + 1, c)}`;
        members.push({
          id, ni: pileAt(r, c), nj: pileAt(r + 1, c),
          kind: 'tie', area: tieAreaNominal, E: Es,
        });
        tieDir.set(id, 'Z');
      }
    }
  }

  // ---- stability restraints (remove rigid-body modes) -------------------
  applyStability(nodes, piles, apex);

  // ---- loads ------------------------------------------------------------
  const loads: TrussLoad[] = [
    { node: 'COL', fx: 0, fy: -p.columnLoad, fz: 0 },
  ];

  // ---- design data ------------------------------------------------------
  // bars provided per tie band = tributary width / spacing
  const bandX = p.capWidth / Math.max(1, p.pileRows);  // tributary for an X-tie
  const bandZ = p.capLength / Math.max(1, p.pileCols); // tributary for a Z-tie
  const nBarsX = Math.max(2, Math.floor(bandX / p.bottomX.spacing) + 1);
  const nBarsZ = Math.max(2, Math.floor(bandZ / p.bottomZ.spacing) + 1);
  const ties = members
    .filter((m) => m.kind === 'tie')
    .map((m) => {
      const dir = tieDir.get(m.id) ?? 'X';
      const g = dir === 'X' ? p.bottomX : p.bottomZ;
      const count = dir === 'X' ? nBarsX : nBarsZ;
      return {
        memberId: m.id,
        barDiameter: g.dia,
        barCount: count,
        asProvided: count * barArea(g.dia),
        asRequired: 0,
        group: dir === 'X' ? 'bottomX' : 'bottomZ',
      };
    });

  const struts = members
    .filter((m) => m.kind === 'strut')
    .map((m) => ({
      memberId: m.id,
      width: p.pileDiameter,
      thickness: p.pileDiameter,
      type: 'bottle-reinforced' as const,
    }));

  const nodeSpecs = [
    {
      id: 'COL',
      type: 'CCC' as const,
      area: p.columnWidth * p.columnDepth,
    },
    ...piles.map((pl) => ({
      id: pl.id,
      type: 'CCT' as const,
      area: (Math.PI / 4) * p.pileDiameter ** 2,
    })),
  ];

  // ---- FE voxel domain --------------------------------------------------
  const res = 14; // voxels along the longest side
  const cell = Math.max(p.capLength, p.capWidth) / res;
  const nx = Math.max(4, Math.round(p.capLength / cell));
  const ny = Math.max(3, Math.round(p.capDepth / cell));
  const nz = Math.max(4, Math.round(p.capWidth / cell));
  const domain = {
    nx, ny, nz,
    dx: p.capLength / nx,
    dy: p.capDepth / ny,
    dz: p.capWidth / nz,
    origin: [-p.capLength / 2, 0, -p.capWidth / 2] as [number, number, number],
    filled: () => true,
  };
  const femLoads: FemLoad[] = [
    { pos: [0, p.capDepth, 0], f: [0, -p.columnLoad, 0] },
  ];
  const femSupports: FemSupport[] = piles.map((pl) => ({
    pos: [pl.x, 0, pl.z],
    fix: [false, true, false] as [boolean, boolean, boolean],
    radius: p.pileDiameter * 0.6,
  }));

  // ---- render geometry --------------------------------------------------
  const render = buildRender(p, piles, members, nodes);

  // ---- 2D continuum CSFM problem (elevation through the pile rows) ------
  // continuum slice — taken in the X–Y plane (csfmAxis 'x', through the pile
  // columns) or the Z–Y plane (csfmAxis 'z', through the pile rows)
  const band = 0.2;
  const onZ = p.csfmAxis === 'z';
  const sliceLen = onZ ? p.capWidth : p.capLength;     // elevation horizontal
  const sliceThk = onZ ? p.capLength : p.capWidth;     // out-of-plane
  const colWidth = onZ ? p.columnDepth : p.columnWidth;
  const grp = onZ ? p.bottomZ : p.bottomX;
  const pileCoord = onZ
    ? [...new Set(piles.map((pl) => pl.z))]
    : [...new Set(piles.map((pl) => pl.x))];
  pileCoord.sort((a, b) => a - b);
  const nBot = Math.max(2, Math.round(sliceThk / grp.spacing));
  const continuum = buildElevationContinuum({
    xMin: -sliceLen / 2, xMax: sliceLen / 2,
    topY: () => p.capDepth,
    botY: () => 0,
    thickness: sliceThk,
    res: 32,
    loads: [{ x: 0, P: p.columnLoad, width: colWidth }],
    supports: pileCoord.map((x, i) => ({
      x, width: p.pileDiameter, fixX: i === 0,
    })),
    rhoTopBand: 0.0015,
    rhoBotBand: Math.min(0.08, (nBot * barArea(grp.dia)) /
      (sliceThk * band * p.capDepth)),
    rhoStirrup: 0.0015,
    bandFrac: band,
    barDia: grp.dia,
  });

  return {
    truss: { nodes, members, loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    continuumOrient: p.csfmAxis,
    render,
    ties,
    struts,
    nodes: nodeSpecs,
    notes,
    info: [],
    totalLoad: p.columnLoad,
  };
}

/** Remove the 6 rigid-body modes of the space truss with minimal restraints. */
function applyStability(nodes: TrussNode[], piles: Pile[], apex: TrussNode): void {
  const find = (id: string) => nodes.find((n) => n.id === id)!;
  // anchor pile fully restrained
  const anchor = find(piles[0].id);
  anchor.fixed = [true, true, true];
  // farthest pile locks the remaining in-plane translation / Y-rotation
  let far = piles[1] ?? piles[0];
  let maxd = -1;
  for (const pl of piles.slice(1)) {
    const d = Math.hypot(pl.x - piles[0].x, pl.z - piles[0].z);
    if (d > maxd) { maxd = d; far = pl; }
  }
  const fn = find(far.id);
  const dx = Math.abs(far.x - piles[0].x);
  const dz = Math.abs(far.z - piles[0].z);
  if (dx >= dz) fn.fixed = [false, true, true];
  else fn.fixed = [true, true, false];

  // collinear pile layouts need the apex braced perpendicular to the line
  const allSameX = piles.every((pl) => Math.abs(pl.x - piles[0].x) < 1e-6);
  const allSameZ = piles.every((pl) => Math.abs(pl.z - piles[0].z) < 1e-6);
  if (allSameX) apex.fixed = [true, false, apex.fixed[2]];
  if (allSameZ) apex.fixed = [apex.fixed[0], false, true];
}

function buildRender(
  p: PileCapParams,
  piles: Pile[],
  members: TrussMember[],
  nodes: TrussNode[],
): ElementModel['render'] {
  const D = p.capDepth;
  const solids: ElementModel['render']['solids'] = [];

  // cap
  solids.push({
    kind: 'box',
    center: [0, D / 2, 0],
    size: [p.capLength, D, p.capWidth],
    color: PALETTE.concrete,
    opacity: 0.55,
    role: 'concrete',
  });
  // column stub — rectangular or circular
  if (p.columnShape === 'circular') {
    solids.push({
      kind: 'cylinder',
      base: [0, D, 0], height: 800,
      radius: p.columnWidth / 2, axis: 'y',
      color: PALETTE.column, opacity: 0.9, role: 'column',
    });
  } else {
    solids.push({
      kind: 'box',
      center: [0, D + 400, 0],
      size: [p.columnWidth, 800, p.columnDepth],
      color: PALETTE.column, opacity: 0.9, role: 'column',
    });
  }
  // piles — circular, square or H-section
  const pd = p.pileDiameter;
  const pileBase = -1200, pileH = 1200, pileMidY = pileBase + pileH / 2;
  piles.forEach((pl) => {
    if (p.pileShape === 'circular') {
      solids.push({
        kind: 'cylinder',
        base: [pl.x, pileBase, pl.z], height: pileH, radius: pd / 2, axis: 'y',
        color: PALETTE.pile, opacity: 0.9, role: 'pile',
      });
    } else if (p.pileShape === 'square') {
      solids.push({
        kind: 'box',
        center: [pl.x, pileMidY, pl.z], size: [pd, pileH, pd],
        color: PALETTE.pile, opacity: 0.9, role: 'pile',
      });
    } else {
      // H-section pile — two flanges plus a web
      const tf = 0.16 * pd;
      solids.push({
        kind: 'box',
        center: [pl.x, pileMidY, pl.z + (pd - tf) / 2], size: [pd, pileH, tf],
        color: PALETTE.pile, opacity: 0.9, role: 'pile',
      });
      solids.push({
        kind: 'box',
        center: [pl.x, pileMidY, pl.z - (pd - tf) / 2], size: [pd, pileH, tf],
        color: PALETTE.pile, opacity: 0.9, role: 'pile',
      });
      solids.push({
        kind: 'box',
        center: [pl.x, pileMidY, pl.z], size: [tf, pileH, pd],
        color: PALETTE.pile, opacity: 0.9, role: 'pile',
      });
    }
  });

  // ---- reinforcement: two-layer orthogonal bottom mat --------------------
  // The long bars sit on the bottom layer; the short bars rest on top of
  // them, so the two directions never occupy the same plane.
  const rebar: ElementModel['render']['rebar'] = [];
  const cover = 75;
  const hook = Math.min(250, D - 2 * cover);
  const xHalf = p.capLength / 2 - cover;
  const zHalf = p.capWidth / 2 - cover;

  // X bars span 2*xHalf, Z bars span 2*zHalf — the longer ones go underneath
  const xLonger = xHalf >= zHalf;
  const botDia = xLonger ? p.bottomX.dia : p.bottomZ.dia;
  const topDia = xLonger ? p.bottomZ.dia : p.bottomX.dia;
  const yLow = cover;                          // bottom (long) layer
  const yHigh = cover + botDia + topDia;        // short layer resting on it
  const yX = xLonger ? yLow : yHigh;
  const yZ = xLonger ? yHigh : yLow;

  // bars running in X (resist Mz), spaced along Z per bottomX.spacing
  const nX = Math.max(2, Math.round((2 * zHalf) / p.bottomX.spacing) + 1);
  for (let i = 0; i < nX; i++) {
    const z = -zHalf + (i * 2 * zHalf) / (nX - 1);
    rebar.push({
      id: `botX-${i}`,
      points: [
        [-xHalf, yX + hook, z],
        [-xHalf, yX, z],
        [xHalf, yX, z],
        [xHalf, yX + hook, z],
      ],
      diameter: p.bottomX.dia,
      role: 'main-tie',
      color: PALETTE.rebarMain,
    });
  }
  // bars running in Z (resist Mx), spaced along X per bottomZ.spacing
  const nZ = Math.max(2, Math.round((2 * xHalf) / p.bottomZ.spacing) + 1);
  for (let i = 0; i < nZ; i++) {
    const x = -xHalf + (i * 2 * xHalf) / (nZ - 1);
    rebar.push({
      id: `botZ-${i}`,
      points: [
        [x, yZ + hook, -zHalf],
        [x, yZ, -zHalf],
        [x, yZ, zHalf],
        [x, yZ + hook, zHalf],
      ],
      diameter: p.bottomZ.dia,
      role: 'distribution',
      color: PALETTE.rebarStirrup,
    });
  }

  // ---- applied load ------------------------------------------------------
  const loads: ElementModel['render']['loads'] = [
    {
      pos: [0, D + 800, 0],
      dir: [0, -1, 0],
      magnitude: p.columnLoad,
      label: `N = ${(p.columnLoad / 1e3).toFixed(0)} kN`,
    },
  ];

  return {
    solids,
    rebar,
    loads,
    bounds: {
      min: [-p.capLength / 2, -1200, -p.capWidth / 2],
      max: [p.capLength / 2, D + 1200, p.capWidth / 2],
    },
  };
}
