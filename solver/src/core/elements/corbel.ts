/**
 * Corbel / Bracket element module.
 *
 * A corbel projecting from a column — the classic strut-and-tie discontinuity
 * region: an inclined compression strut runs from the bearing into the column
 * and a horizontal tension tie along the top anchors it back. Modelled as a
 * trimmed D-region (the column boundary is the support); the elevation (column
 * stub + corbel) feeds the continuum CSFM solver.
 */

import type { ElementModel, BarGroupCount } from './types';
import { PALETTE } from './types';
import type { StirrupGroup } from './hammerhead';
import type { TrussNode, TrussMember, TrussLoad } from '../stm/truss';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface CorbelParams {
  columnWidth: number;     // cw (mm)
  columnHeight: number;    // modelled column stub height (mm)
  projection: number;      // corbel projection from the column face (mm)
  depthFace: number;       // corbel depth at the column face (mm)
  depthEdge: number;       // corbel depth at the outer edge (mm)
  thickness: number;       // out-of-plane width (mm)
  loadV: number;           // factored vertical load (N)
  loadDist: number;        // load distance from the column face, a_v (mm)
  bearingWidth: number;    // mm
  topBars: BarGroupCount;  // primary tension tie
  stirrup: StirrupGroup;   // closed horizontal framing bars
}

export function defaultCorbelParams(): CorbelParams {
  return {
    columnWidth: 600,
    columnHeight: 1600,
    projection: 500,
    depthFace: 700,
    depthEdge: 400,
    thickness: 400,
    loadV: 600e3,
    loadDist: 250,
    bearingWidth: 300,
    topBars: { dia: 25, count: 4 },
    stirrup: { dia: 12, spacing: 120, legs: 2 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildCorbel(p: CorbelParams): ElementModel {
  const notes: string[] = [];
  const cw = p.columnWidth;
  const Hc = p.columnHeight;
  const proj = p.projection;
  const cover = 60;

  /** corbel soffit elevation (mm) for 0 <= x <= projection */
  const soffit = (x: number): number => {
    const t = Math.min(1, Math.max(0, x / Math.max(1, proj)));
    return (Hc - p.depthFace) + (p.depthFace - p.depthEdge) * t;
  };
  const a = Math.min(proj, Math.max(0, p.loadDist));
  if (p.loadDist > proj) notes.push('Load lies beyond the corbel tip — clamped to the edge.');

  // ---- strut-and-tie model (trimmed D-region) ---------------------------
  // LOAD: bearing point; C_TOP / C_MID: the column boundary (supports)
  const nodes: TrussNode[] = [
    { id: 'LOAD', x: a, y: Hc, z: 0, fixed: [false, false, true], nodeType: 'CCT', label: 'Bearing' },
    { id: 'C_TOP', x: 0, y: Hc, z: 0, fixed: [true, true, true], nodeType: 'CCT', label: 'Tie anchorage' },
    { id: 'C_MID', x: 0, y: Hc - p.depthFace, z: 0, fixed: [true, true, true], nodeType: 'CCC', label: 'Strut-to-column node' },
  ];
  const Ec = 30000, Es = 200000;
  const members: TrussMember[] = [
    { id: 'TIE', ni: 'LOAD', nj: 'C_TOP', kind: 'tie',
      area: p.topBars.count * barArea(p.topBars.dia), E: Es },
    { id: 'STRUT', ni: 'LOAD', nj: 'C_MID', kind: 'strut',
      area: p.thickness * Math.min(p.thickness, 0.3 * p.depthFace), E: Ec },
  ];
  const loads: TrussLoad[] = [{ node: 'LOAD', fx: 0, fy: -p.loadV, fz: 0 }];

  const ties = [{
    memberId: 'TIE', barDiameter: p.topBars.dia, barCount: p.topBars.count,
    asProvided: p.topBars.count * barArea(p.topBars.dia), asRequired: 0,
    group: 'topBars',
  }];
  const struts = [{
    memberId: 'STRUT',
    width: Math.min(p.thickness, 0.3 * p.depthFace),
    thickness: p.thickness,
    type: 'bottle-reinforced' as const,
  }];
  const nodeSpecs = [
    { id: 'LOAD', type: 'CCT' as const, area: p.thickness * p.bearingWidth },
    { id: 'C_TOP', type: 'CCT' as const, area: p.thickness * cw },
    { id: 'C_MID', type: 'CCC' as const, area: p.thickness * cw },
  ];

  // ---- FE voxel domain (column stub + corbel) ---------------------------
  const xMin = -cw, xMax = proj;
  const W = xMax - xMin;
  const res = 22;
  const cell = Math.max(W, Hc) / res;
  const nx = Math.max(8, Math.round(W / cell));
  const ny = Math.max(8, Math.round(Hc / cell));
  const nz = Math.max(2, Math.round(p.thickness / cell));
  const dx = W / nx, dy = Hc / ny;
  const solid = (x: number, y: number): boolean =>
    x <= 0 ? true : y >= soffit(x) - dy * 0.5;
  const domain = {
    nx, ny, nz,
    dx, dy, dz: p.thickness / nz,
    origin: [xMin, 0, -p.thickness / 2] as [number, number, number],
    filled: (i: number, j: number) => solid(xMin + (i + 0.5) * dx, (j + 0.5) * dy),
  };
  const femLoads: FemLoad[] = [{ pos: [a, Hc, 0], f: [0, -p.loadV, 0] }];
  const femSupports: FemSupport[] = [{
    pos: [-cw / 2, 0, 0],
    fix: [true, true, false] as [boolean, boolean, boolean],
    radius: cw,
  }];

  // ---- 2D continuum CSFM problem ----------------------------------------
  const band = 0.2;
  const continuum = buildElevationContinuum({
    xMin, xMax,
    topY: () => Hc,
    botY: (x) => (x <= 0 ? 0 : soffit(x)),
    thickness: p.thickness,
    res: 28,
    loads: [{ x: a, P: p.loadV, width: p.bearingWidth }],
    supports: [{ x: -cw / 2, width: cw, fixX: true }],
    rhoTopBand: Math.min(0.08, (p.topBars.count * barArea(p.topBars.dia)) /
      (p.thickness * band * p.depthFace)),
    rhoBotBand: 0.0015,
    rhoStirrup: Math.min(0.04, (p.stirrup.legs * barArea(p.stirrup.dia) /
      p.stirrup.spacing) / p.thickness),
    bandFrac: band,
    barDia: p.topBars.dia,
  });

  // ---- render -----------------------------------------------------------
  const solids: ElementModel['render']['solids'] = [];
  // column
  solids.push({
    kind: 'box',
    center: [-cw / 2, Hc / 2, 0],
    size: [cw, Hc, p.thickness],
    color: PALETTE.column, opacity: 0.55, role: 'concrete',
  });
  // corbel — tapered prism (flat top, sloped soffit)
  solids.push({
    kind: 'prism',
    profile: [
      [0, Hc],
      [proj, Hc],
      [proj, Hc - p.depthEdge],
      [0, Hc - p.depthFace],
    ],
    zCenter: 0, zDepth: p.thickness,
    color: PALETTE.concrete, opacity: 0.55, role: 'concrete',
  });
  // bearing plate
  solids.push({
    kind: 'box',
    center: [a, Hc + 70, 0],
    size: [p.bearingWidth, 140, Math.min(p.thickness, 360)],
    color: PALETTE.bearing, opacity: 1, role: 'bearing',
  });

  // ---- reinforcement ----------------------------------------------------
  const rebar: ElementModel['render']['rebar'] = [];
  const zc = p.thickness / 2 - cover;
  // longitudinal / vertical bars stack clear of the closed hoops (hoop bar +
  // bar), so the rendered cylinders sit on top of one another, not intersect
  const ds = p.stirrup.dia;
  const zi = Math.max(8, zc - ds - p.topBars.dia);
  const across = (count: number): number[] => {
    const n = Math.max(2, Math.min(5, count));
    return Array.from({ length: n }, (_, i) => -zi + (i * 2 * zi) / (n - 1));
  };
  // primary tension tie along the top, hooked down at the outer end
  across(p.topBars.count).forEach((z, i) => {
    rebar.push({
      id: `tie-${i}`,
      points: [
        [-cw + cover, Hc - cover, z],
        [proj - cover, Hc - cover, z],
        [proj - cover, Hc - cover - Math.min(250, p.depthEdge - cover), z],
      ],
      diameter: p.topBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  // closed horizontal framing bars — each one stops where the sloped soffit
  // rises to meet it, so the lower (shorter) hoops stay inside the concrete
  const slope = (p.depthFace - p.depthEdge) / Math.max(1, proj);
  const innerX = -cw + cover;
  const nHoop = Math.max(2, Math.round(p.depthFace / p.stirrup.spacing));
  for (let i = 0; i < nHoop; i++) {
    const y = Hc - cover - (i + 0.5) * (p.depthFace - 2 * cover) / nHoop;
    // x where the soffit sits one cover below this bar level
    const xAtSoffit = slope > 1e-6
      ? ((y - cover) - (Hc - p.depthFace)) / slope
      : (y - cover >= Hc - p.depthFace ? proj : -1);
    const outerX = Math.min(proj - cover, xAtSoffit);
    if (outerX <= innerX + 100) continue;       // too low — no room for a hoop
    rebar.push({
      id: `hoop-${i}`,
      points: [
        [innerX, y, -zc], [outerX, y, -zc],
        [outerX, y, zc], [innerX, y, zc],
      ],
      diameter: p.stirrup.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  // column ties — closed stirrups around the column stub (the straight part
  // below the tapered bracket)
  const tieTop = Math.max(cover + 1, Hc - p.depthFace);
  const nCt = Math.max(2, Math.round((tieTop - cover) / p.stirrup.spacing));
  for (let i = 0; i <= nCt; i++) {
    const y = cover + (i * (tieTop - cover)) / nCt;
    rebar.push({
      id: `col-tie-${i}`,
      points: [
        [-cw + cover, y, -zc], [-cover, y, -zc],
        [-cover, y, zc], [-cw + cover, y, zc],
      ],
      diameter: p.stirrup.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  // column vertical bars — on all four faces, nested inside the column ties
  const gt = p.stirrup.dia + p.topBars.dia;
  const colXb = -cw + cover + gt, colXf = -cover - gt;   // back & front faces
  const nCz = Math.max(2, Math.min(5, Math.round(p.thickness / 150)));
  const nCx = Math.max(3, Math.min(6, Math.round(cw / 150)));
  for (const [tag, cx] of [['back', colXb], ['front', colXf]] as const) {
    for (let k = 0; k < nCz; k++) {
      const z = -zi + (k * 2 * zi) / (nCz - 1);
      rebar.push({
        id: `col-${tag}-${k}`,
        points: [[cx, cover, z], [cx, Hc - cover, z]],
        diameter: p.topBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
      });
    }
  }
  for (const [tag, cz] of [['nz', -zi], ['pz', zi]] as const) {
    for (let k = 1; k < nCx - 1; k++) {       // interior — corners already placed
      const x = colXb + (k * (colXf - colXb)) / (nCx - 1);
      rebar.push({
        id: `col-${tag}-${k}`,
        points: [[x, cover, cz], [x, Hc - cover, cz]],
        diameter: p.topBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
      });
    }
  }

  // anchorage bars hugging the corbel's tapered soffit, hooked up into the
  // corbel body at the column face and at the tip
  const hookU = Math.min(160, p.depthFace - 2 * cover);
  const yS0 = soffit(0) + cover;
  const ySp = soffit(proj) + cover;
  for (const [tag, z] of [['nz', -zi], ['mz', 0], ['pz', zi]] as const) {
    rebar.push({
      id: `soffit-${tag}`,
      points: [
        [cover, yS0 + hookU, z],
        [cover, yS0, z],
        [proj - cover, ySp, z],
        [proj - cover, ySp + hookU, z],
      ],
      diameter: p.topBars.dia, role: 'distribution', color: PALETTE.rebarMain,
    });
  }

  const loadArrows: ElementModel['render']['loads'] = [{
    pos: [a, Hc + 200, 0],
    dir: [0, -1, 0],
    magnitude: p.loadV,
    label: `${(p.loadV / 1e3).toFixed(0)} kN`,
  }];

  return {
    truss: { nodes, members, loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    render: {
      solids, rebar, loads: loadArrows,
      bounds: {
        min: [xMin, 0, -p.thickness / 2],
        max: [xMax, Hc + 320, p.thickness / 2],
      },
    },
    ties, struts, nodes: nodeSpecs, notes, info: [],
    totalLoad: p.loadV,
  };
}
