/**
 * Multi-Column Bent Cap element module.
 *
 * A cap beam continuous over several columns, carrying girder reactions on
 * bearings along its top. Modelled as a parallel-chord deep-beam strut-and-tie
 * truss. Column positions and girder loads are individually editable.
 */

import type { ElementModel, BarGroupCount, PointLoad } from './types';
import { PALETTE } from './types';
import type { StirrupGroup } from './hammerhead';
import { buildDeepBeamTruss } from './deepBeam';
import type { DeepBeamPanelLoad, DeepBeamSupport } from './deepBeam';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface MultiColumnBentParams {
  capDepth: number;        // Y (mm)
  capWidth: number;        // Z (mm)
  columns: number[];       // column centreline x-positions (mm) — editable
  columnWidth: number;     // X (mm)
  columnDepth: number;     // Z (mm)
  columnHeight: number;    // mm (render)
  overhang: number;        // cantilever beyond the end columns (mm)
  girders: PointLoad[];    // editable girder loads
  topBars: BarGroupCount;
  bottomBars: BarGroupCount;
  stirrup: StirrupGroup;
}

export function defaultMultiColumnBentParams(): MultiColumnBentParams {
  const columns = [-6000, 0, 6000];
  const n = 7, spc = 2400;
  const girders: PointLoad[] = [];
  for (let i = 0; i < n; i++) {
    girders.push({
      id: `g${i + 1}`,
      x: -((n - 1) * spc) / 2 + i * spc,
      load: 900e3,
    });
  }
  return {
    capDepth: 1500,
    capWidth: 1200,
    columns,
    columnWidth: 1000,
    columnDepth: 1000,
    columnHeight: 7000,
    overhang: 1500,
    girders,
    topBars: { dia: 32, count: 8 },
    bottomBars: { dia: 32, count: 8 },
    stirrup: { dia: 16, spacing: 200, legs: 4 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildMultiColumnBent(p: MultiColumnBentParams): ElementModel {
  const notes: string[] = [];
  const info: string[] = [];
  const cover = 80;
  const colX = [...p.columns].sort((a, b) => a - b);
  const xMin = Math.min(...colX) - p.overhang;
  const xMax = Math.max(...colX) + p.overhang;
  const capLength = xMax - xMin;
  const xMid = (xMin + xMax) / 2;

  const topY = () => p.capDepth - cover;
  const botY = () => cover;

  const supports: DeepBeamSupport[] = colX.map((x, i) => ({
    x, width: p.columnWidth, label: `Column ${i + 1}`,
  }));

  const loads: DeepBeamPanelLoad[] = [];
  p.girders.forEach((g, i) => {
    if (g.x < xMin - 1 || g.x > xMax + 1) {
      notes.push(`Girder ${i + 1} is outside the cap length — ignored.`);
      return;
    }
    loads.push({ x: g.x, P: g.load, label: `Girder ${i + 1}` });
  });
  if (loads.length === 0) notes.push('No girder loads defined.');

  const tr = buildDeepBeamTruss({
    topY, botY, width: p.capWidth, loads, supports,
    barDiameter: p.topBars.dia,
  });
  const totalLoad = loads.reduce((s, l) => s + l.P, 0);

  if (colX.length > 2) {
    info.push(
      `${colX.length}-column bent is statically indeterminate — the truss ` +
        'is solved by the stiffness method, distributing forces according to ' +
        'member stiffnesses. This is a valid lower-bound (safe) solution.',
    );
  }

  // ---- design data ------------------------------------------------------
  const nV = tr.members.filter((m) => m.id.startsWith('V')).length;
  const stirTrib = capLength / Math.max(1, nV);
  const asStirrupPerV =
    p.stirrup.legs * barArea(p.stirrup.dia) * (stirTrib / p.stirrup.spacing);

  const ties = tr.members
    .filter((m) => m.id.startsWith('TC') || m.id.startsWith('BC') || m.id.startsWith('V'))
    .map((m) => {
      const isV = m.id.startsWith('V');
      const isBot = m.id.startsWith('BC');
      const g = isBot ? p.bottomBars : p.topBars;
      const dia = isV ? p.stirrup.dia : g.dia;
      const as = isV ? asStirrupPerV : g.count * barArea(g.dia);
      const count = isV ? p.stirrup.legs : g.count;
      return {
        memberId: m.id, barDiameter: dia, barCount: count,
        asProvided: as, asRequired: 0,
        group: isV ? 'stirrup' : isBot ? 'bottomBars' : 'topBars',
      };
    });
  // A diagonal strut that lands on a bearing is limited by that bearing: its
  // width is the smaller of the geometric rule and the bearing length, the
  // bearing being the nodal zone through which the strut force enters.
  // Here the bearing is the column.
  const onBearing = new Set(
    tr.nodes
      .filter((nd) => nd.id.startsWith('B') && supports.some((s) => Math.abs(nd.x - s.x) < 1))
      .map((nd) => nd.id),
  );
  const struts = tr.members
    .filter((m) => m.id.startsWith('D'))
    .map((m) => {
      const w = Math.min(p.capWidth, 0.3 * p.capDepth);
      const bears = onBearing.has(m.ni) || onBearing.has(m.nj);
      return {
        memberId: m.id,
        width: bears ? Math.min(w, p.columnWidth) : w,
        thickness: p.capWidth,
        type: 'bottle-reinforced' as const,
      };
    });
  const nodeSpecs = tr.nodes.map((nd) => ({
    id: nd.id,
    type: (nd.id.startsWith('B') ? 'CCT' : 'CCC') as 'CCC' | 'CCT',
    area: p.capWidth * p.columnWidth,
  }));

  // ---- FE voxel domain --------------------------------------------------
  const res = 28;
  const cell = capLength / res;
  const nx = Math.max(12, Math.round(capLength / cell));
  const ny = Math.max(4, Math.round(p.capDepth / cell));
  const nz = Math.max(2, Math.round(p.capWidth / cell));
  const domain = {
    nx, ny, nz,
    dx: capLength / nx, dy: p.capDepth / ny, dz: p.capWidth / nz,
    origin: [xMin, 0, -p.capWidth / 2] as [number, number, number],
    filled: () => true,
  };
  const femLoads: FemLoad[] = loads.map((l) => ({
    pos: [l.x, p.capDepth, 0], f: [0, -l.P, 0],
  }));
  const femSupports: FemSupport[] = colX.map((x) => ({
    pos: [x, 0, 0],
    fix: [false, true, false] as [boolean, boolean, boolean],
    radius: p.columnWidth * 0.6,
  }));

  // ---- render -----------------------------------------------------------
  const solids: ElementModel['render']['solids'] = [];
  solids.push({
    kind: 'box',
    center: [xMid, p.capDepth / 2, 0],
    size: [capLength, p.capDepth, p.capWidth],
    color: PALETTE.concrete, opacity: 0.5, role: 'concrete',
  });
  colX.forEach((x) => {
    solids.push({
      kind: 'box',
      center: [x, -p.columnHeight / 2, 0],
      size: [p.columnWidth, p.columnHeight, p.columnDepth],
      color: PALETTE.column, opacity: 0.95, role: 'column',
    });
  });
  loads.forEach((l) => {
    solids.push({
      kind: 'box',
      center: [l.x, p.capDepth + 90, 0],
      size: [350, 180, Math.min(500, p.capWidth)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });

  // ---- reinforcement ----------------------------------------------------
  const rebar: ElementModel['render']['rebar'] = [];
  const zc = p.capWidth / 2 - cover;
  const x0 = xMin + cover, x1 = xMax - cover;
  const hook = Math.min(300, p.capDepth - 2 * cover);
  // longitudinal bars stack clear of the closed stirrups (stirrup bar +
  // longitudinal bar), so the rendered cylinders sit on top of one another
  const dS = p.stirrup.dia;
  const zi = Math.max(8, zc - dS - Math.max(p.topBars.dia, p.bottomBars.dia));
  const across = (count: number): number[] => {
    const n = Math.max(2, count > 6 ? 6 : count);
    return Array.from({ length: n }, (_, i) => -zi + (i * 2 * zi) / (n - 1));
  };
  const yTop = p.capDepth - cover - dS - p.topBars.dia;
  const yBot = cover + dS + p.bottomBars.dia;
  across(p.topBars.count).forEach((z, i) => {
    rebar.push({
      id: `top-${i}`,
      points: [
        [x0, yTop - hook, z], [x0, yTop, z],
        [x1, yTop, z], [x1, yTop - hook, z],
      ],
      diameter: p.topBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  across(p.bottomBars.count).forEach((z, i) => {
    rebar.push({
      id: `bot-${i}`,
      points: [
        [x0, yBot + hook, z], [x0, yBot, z],
        [x1, yBot, z], [x1, yBot + hook, z],
      ],
      diameter: p.bottomBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  const nStir = Math.max(2, Math.round((capLength - 2 * cover) / p.stirrup.spacing));
  for (let i = 0; i <= nStir; i++) {
    const x = x0 + (i * (x1 - x0)) / nStir;
    rebar.push({
      id: `stir-${i}`,
      points: [
        [x, cover, -zc], [x, p.capDepth - cover, -zc],
        [x, p.capDepth - cover, zc], [x, cover, zc],
      ],
      diameter: p.stirrup.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  const loadArrows: ElementModel['render']['loads'] = loads.map((l) => ({
    pos: [l.x, p.capDepth + 200, 0] as [number, number, number],
    dir: [0, -1, 0] as [number, number, number],
    magnitude: l.P,
    label: `${(l.P / 1e3).toFixed(0)} kN`,
  }));

  // ---- 2D continuum CSFM problem (elevation) ----------------------------
  const band = 0.16;
  const continuum = buildElevationContinuum({
    xMin, xMax,
    topY: () => p.capDepth,
    botY: () => 0,
    thickness: p.capWidth,
    res: 38,
    loads: loads.map((l) => ({ x: l.x, P: l.P, width: Math.max(350, capLength / 22) })),
    supports: colX.map((x, i) => ({ x, width: p.columnWidth, fixX: i === 0 })),
    rhoTopBand: Math.min(0.08, (p.topBars.count * barArea(p.topBars.dia)) /
      (p.capWidth * band * p.capDepth)),
    rhoBotBand: Math.min(0.08, (p.bottomBars.count * barArea(p.bottomBars.dia)) /
      (p.capWidth * band * p.capDepth)),
    rhoStirrup: Math.min(0.04, (p.stirrup.legs * barArea(p.stirrup.dia) /
      p.stirrup.spacing) / p.capWidth),
    bandFrac: band,
    barDia: p.bottomBars.dia,
  });

  return {
    truss: { nodes: tr.nodes, members: tr.members, loads: tr.loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    render: {
      solids, rebar, loads: loadArrows,
      bounds: {
        min: [xMin, -p.columnHeight, -p.capWidth / 2],
        max: [xMax, p.capDepth + 700, p.capWidth / 2],
      },
    },
    ties, struts, nodes: nodeSpecs, notes, info, totalLoad,
  };
}
