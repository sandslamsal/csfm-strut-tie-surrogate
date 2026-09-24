/**
 * Cantilever Wall-Type Bridge Pier element module.
 *
 * A wall-type pier fixed at its base, carrying a vertical axial load and a
 * transverse (lateral) load at the top — the experimentally validated example
 * of the CSFM book (§6.3). The lateral load is carried by an inclined
 * compression field to the compression toe, balanced by vertical tension
 * reinforcement at the opposite face.
 */

import type { ElementModel, BarGroup, BarGroupCount } from './types';
import { PALETTE } from './types';
import type { TrussNode, TrussMember, TrussLoad } from '../stm/truss';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import type { ContinuumProblem } from '../csfm/continuum';

export interface CantileverPierParams {
  width: number;          // wall length in elevation, B (mm)
  height: number;         // pier height H (mm)
  thickness: number;      // wall thickness (mm)
  lateralLoad: number;    // factored transverse load at the top (N)
  axialLoad: number;      // factored axial load at the top (N)
  vertBars: BarGroupCount; // vertical flexural bars, per face
  tieBars: BarGroup;       // horizontal ties (⌀ @ spacing)
}

export function defaultCantileverPierParams(): CantileverPierParams {
  return {
    width: 2400,
    height: 6000,
    thickness: 600,
    lateralLoad: 450e3,
    axialLoad: 3000e3,
    vertBars: { dia: 28, count: 8 },
    tieBars: { dia: 16, spacing: 250 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildCantileverPier(p: CantileverPierParams): ElementModel {
  const notes: string[] = [];
  const B = p.width, H = p.height, t = p.thickness;
  const cover = 60;
  const Ec = 30000, Es = 200000;

  // ---- strut-and-tie model (single-panel cantilever) --------------------
  // lateral load +x: left edge in tension, right edge in compression
  const nodes: TrussNode[] = [
    { id: 'TL', x: -B / 2, y: H, z: 0, fixed: [false, false, true], nodeType: 'CCT' },
    { id: 'TR', x: B / 2, y: H, z: 0, fixed: [false, false, true], nodeType: 'CCC' },
    { id: 'BL', x: -B / 2, y: 0, z: 0, fixed: [true, true, true], nodeType: 'CCT' },
    { id: 'BR', x: B / 2, y: 0, z: 0, fixed: [true, true, true], nodeType: 'CCC' },
  ];
  const members: TrussMember[] = [
    { id: 'EDGE-T', ni: 'TL', nj: 'BL', kind: 'tie', area: 2 * p.vertBars.count * barArea(p.vertBars.dia), E: Es },
    { id: 'EDGE-C', ni: 'TR', nj: 'BR', kind: 'strut', area: B * t * 0.25, E: Ec },
    { id: 'TOP', ni: 'TL', nj: 'TR', kind: 'tie', area: B * t * 0.1, E: Es },
    { id: 'DIAG', ni: 'TL', nj: 'BR', kind: 'strut', area: B * t * 0.22, E: Ec },
  ];
  const loads: TrussLoad[] = [
    { node: 'TL', fx: p.lateralLoad / 2, fy: -p.axialLoad / 2, fz: 0 },
    { node: 'TR', fx: p.lateralLoad / 2, fy: -p.axialLoad / 2, fz: 0 },
  ];

  const ties = [
    {
      memberId: 'EDGE-T', barDiameter: p.vertBars.dia, barCount: 2 * p.vertBars.count,
      asProvided: 2 * p.vertBars.count * barArea(p.vertBars.dia), asRequired: 0,
      group: 'vertBars',
    },
    {
      memberId: 'TOP', barDiameter: p.tieBars.dia, barCount: 4,
      asProvided: 4 * barArea(p.tieBars.dia), asRequired: 0, group: 'tieBars',
    },
  ];
  const struts = [
    { memberId: 'EDGE-C', width: Math.min(t, 0.25 * B), thickness: t, type: 'prismatic' as const },
    { memberId: 'DIAG', width: Math.min(t, 0.25 * B), thickness: t, type: 'bottle-reinforced' as const },
  ];
  const nodeSpecs = nodes.map((n) => ({
    id: n.id, type: (n.nodeType ?? 'CCC') as 'CCC' | 'CCT', area: t * B * 0.3,
  }));

  // ---- FE voxel domain --------------------------------------------------
  const cell = Math.max(B, H) / 26;
  const nx = Math.max(6, Math.round(B / cell));
  const ny = Math.max(12, Math.round(H / cell));
  const nz = Math.max(2, Math.round(t / cell));
  const domain = {
    nx, ny, nz, dx: B / nx, dy: H / ny, dz: t / nz,
    origin: [-B / 2, 0, -t / 2] as [number, number, number],
    filled: () => true,
  };
  const femLoads: FemLoad[] = [
    { pos: [0, H, 0], f: [p.lateralLoad, -p.axialLoad, 0] },
  ];
  const femSupports: FemSupport[] = [
    { pos: [0, 0, 0], fix: [true, true, true] as [boolean, boolean, boolean], radius: B * 0.6 },
  ];

  // ---- 2D continuum CSFM problem (built directly) -----------------------
  const cnx = Math.max(10, Math.min(22, Math.round(B / 130)));
  const cny = Math.max(20, Math.min(46, Math.round(H / 130)));
  const dxc = B / cnx, dyc = H / cny;
  const rhoVert = Math.min(0.06, (2 * p.vertBars.count * barArea(p.vertBars.dia)) / (B * t));
  const rhoTie = Math.min(0.03, (barArea(p.tieBars.dia) / p.tieBars.spacing) / t);
  // axial load distributed across the top, lateral load at the top edge
  const topLoads: ContinuumProblem['loads'] = [];
  const nTop = 6;
  for (let k = 0; k < nTop; k++) {
    const x = -B / 2 + (k + 0.5) * (B / nTop);
    topLoads.push({ pos: [x, H], f: [p.lateralLoad / nTop, -p.axialLoad / nTop] });
  }
  const continuum: ContinuumProblem = {
    nx: cnx, ny: cny, dx: dxc, dy: dyc,
    origin: [-B / 2, 0], thickness: t,
    filled: () => true,
    rhoX: () => rhoTie,
    rhoY: () => rhoVert,
    loads: topLoads,
    barDia: p.vertBars.dia,
    supports: [
      { pos: [0, 0], radius: B * 0.55, fix: [true, true] },
    ],
  };

  // ---- render -----------------------------------------------------------
  const solids: ElementModel['render']['solids'] = [];
  solids.push({
    kind: 'box', center: [0, H / 2, 0], size: [B, H, t],
    color: PALETTE.concrete, opacity: 0.5, role: 'concrete',
  });
  // base / footing block
  solids.push({
    kind: 'box', center: [0, -300, 0], size: [B * 1.4, 600, t * 1.4],
    color: PALETTE.column, opacity: 0.92, role: 'column',
  });

  // ---- reinforcement ----------------------------------------------------
  const rebar: ElementModel['render']['rebar'] = [];
  const zc = t / 2 - cover;
  const xc = B / 2 - cover;
  // vertical bars stack clear of the closed ties (tie bar + vertical bar), so
  // the rendered cylinders sit on top of one another rather than intersecting
  const g = p.tieBars.dia + p.vertBars.dia;
  const xi = Math.max(8, xc - g), zi = Math.max(8, zc - g);
  const nF = Math.max(2, Math.min(6, p.vertBars.count));
  // vertical flexural bars on the two end faces (spread across the thickness)
  for (const face of [-1, 1]) {
    for (let k = 0; k < nF; k++) {
      const z = -zi + (k * 2 * zi) / (nF - 1);
      rebar.push({
        id: `vert-x${face}-${k}`,
        points: [
          [face * xi, cover, z],
          [face * xi, H - cover, z],
        ],
        diameter: p.vertBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
      });
    }
  }
  // vertical bars on the two wide faces (spread along the width) — so the
  // cage has vertical reinforcement on all four faces
  const nW = Math.max(3, Math.min(12, Math.round(B / 280)));
  for (const face of [-1, 1]) {
    for (let k = 0; k < nW; k++) {
      const x = -xi + ((k + 1) * 2 * xi) / (nW + 1);
      rebar.push({
        id: `vert-z${face}-${k}`,
        points: [
          [x, cover, face * zi],
          [x, H - cover, face * zi],
        ],
        diameter: p.vertBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
      });
    }
  }
  // horizontal closed ties up the height
  const nTie = Math.max(2, Math.round((H - 2 * cover) / p.tieBars.spacing));
  for (let i = 0; i <= nTie; i++) {
    const y = cover + (i * (H - 2 * cover)) / nTie;
    rebar.push({
      id: `tie-${i}`,
      points: [
        [-B / 2 + cover, y, -zc], [B / 2 - cover, y, -zc],
        [B / 2 - cover, y, zc], [-B / 2 + cover, y, zc],
      ],
      diameter: p.tieBars.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  const loadArrows: ElementModel['render']['loads'] = [
    { pos: [0, H + 220, 0], dir: [0, -1, 0], magnitude: p.axialLoad,
      label: `N = ${(p.axialLoad / 1e3).toFixed(0)} kN` },
    { pos: [-B / 2 - 220, H - 200, 0], dir: [1, 0, 0], magnitude: p.lateralLoad,
      label: `V = ${(p.lateralLoad / 1e3).toFixed(0)} kN` },
  ];

  if (H / B > 5) notes.push('Slender pier (H/B > 5) — flexure dominates; the D-region is concentrated near the base.');

  return {
    truss: { nodes, members, loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    render: {
      solids, rebar, loads: loadArrows,
      bounds: {
        min: [-B / 2 - 500, -700, -t / 2],
        max: [B / 2 + 200, H + 500, t / 2],
      },
    },
    ties, struts, nodes: nodeSpecs, notes, info: [],
    totalLoad: p.axialLoad,
  };
}
