/**
 * Deep Beam element module.
 *
 * A simply-supported deep beam (span-to-depth ratio low enough that the whole
 * member is a D-region) carrying point loads on its top edge — the canonical
 * Compatible Stress Field / strut-and-tie example (CSFM book §5.2). Modelled
 * with the parallel-chord deep-beam truss; the elevation feeds the continuum
 * CSFM solver directly.
 */

import type { ElementModel, BarGroupCount, PointLoad } from './types';
import { PALETTE } from './types';
import type { StirrupGroup } from './hammerhead';
import { buildDeepBeamTruss } from './deepBeam';
import type { DeepBeamPanelLoad, DeepBeamSupport } from './deepBeam';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface DeepBeamParams {
  span: number;            // centre-to-centre of supports (mm)
  overhang: number;        // length beyond each support (mm)
  height: number;          // overall depth H (mm)
  thickness: number;       // out-of-plane width b (mm)
  supportWidth: number;    // bearing width (mm)
  loads: PointLoad[];      // editable top-edge loads
  topBars: BarGroupCount;
  bottomBars: BarGroupCount;
  stirrup: StirrupGroup;
}

export function defaultDeepBeamParams(): DeepBeamParams {
  return {
    span: 4000,
    overhang: 500,
    height: 2400,
    thickness: 400,
    supportWidth: 400,
    loads: [{ id: 'p1', x: 0, load: 3000e3 }],
    topBars: { dia: 16, count: 4 },
    bottomBars: { dia: 32, count: 8 },
    stirrup: { dia: 12, spacing: 250, legs: 2 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildDeepBeam(p: DeepBeamParams): ElementModel {
  const notes: string[] = [];
  const cover = 70;
  const H = p.height;
  const xL = -p.span / 2, xR = p.span / 2;
  const xMin = xL - p.overhang, xMax = xR + p.overhang;
  const capLength = xMax - xMin;

  const topY = () => H - cover;
  const botY = () => cover;

  const supports: DeepBeamSupport[] = [
    { x: xL, width: p.supportWidth, label: 'Support A' },
    { x: xR, width: p.supportWidth, label: 'Support B' },
  ];
  const loads: DeepBeamPanelLoad[] = [];
  p.loads.forEach((l, i) => {
    if (l.x < xMin - 1 || l.x > xMax + 1) {
      notes.push(`Load ${i + 1} is outside the beam — ignored.`);
      return;
    }
    loads.push({ x: l.x, P: l.load, label: `Load ${i + 1}` });
  });
  if (loads.length === 0) notes.push('No loads defined.');

  const tr = buildDeepBeamTruss({
    topY, botY, width: p.thickness, loads, supports,
    barDiameter: p.bottomBars.dia,
  });
  const totalLoad = loads.reduce((s, l) => s + l.P, 0);
  if (p.span / H > 4) {
    notes.push(
      `Span/depth = ${(p.span / H).toFixed(1)} > 4 — the member is slender; ` +
        'the CSFM strut-and-tie idealisation suits deep members best.',
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
  const onBearing = new Set(
    tr.nodes
      .filter((nd) => nd.id.startsWith('B') && supports.some((s) => Math.abs(nd.x - s.x) < 1))
      .map((nd) => nd.id),
  );
  const struts = tr.members
    .filter((m) => m.id.startsWith('D'))
    .map((m) => {
      const w = Math.min(p.thickness, 0.3 * H);
      const bears = onBearing.has(m.ni) || onBearing.has(m.nj);
      return {
        memberId: m.id,
        width: bears ? Math.min(w, p.supportWidth) : w,
        thickness: p.thickness,
        type: 'bottle-reinforced' as const,
      };
    });
  const nodeSpecs = tr.nodes.map((nd) => ({
    id: nd.id,
    type: (nd.id.startsWith('B') ? 'CCT' : 'CCC') as 'CCC' | 'CCT',
    area: p.thickness * p.supportWidth,
  }));

  // ---- FE voxel domain --------------------------------------------------
  const res = 28;
  const cell = capLength / res;
  const nx = Math.max(12, Math.round(capLength / cell));
  const ny = Math.max(6, Math.round(H / cell));
  const nz = Math.max(2, Math.round(p.thickness / cell));
  const domain = {
    nx, ny, nz,
    dx: capLength / nx, dy: H / ny, dz: p.thickness / nz,
    origin: [xMin, 0, -p.thickness / 2] as [number, number, number],
    filled: () => true,
  };
  const femLoads: FemLoad[] = loads.map((l) => ({ pos: [l.x, H, 0], f: [0, -l.P, 0] }));
  const femSupports: FemSupport[] = supports.map((s) => ({
    pos: [s.x, 0, 0],
    fix: [false, true, false] as [boolean, boolean, boolean],
    radius: p.supportWidth * 0.7,
  }));

  // ---- 2D continuum CSFM problem ----------------------------------------
  const band = 0.16;
  const continuum = buildElevationContinuum({
    xMin, xMax,
    topY: () => H, botY: () => 0,
    thickness: p.thickness,
    res: 36,
    loads: loads.map((l) => ({ x: l.x, P: l.P, width: Math.max(300, capLength / 16) })),
    supports: supports.map((s, i) => ({ x: s.x, width: s.width, fixX: i === 0 })),
    rhoTopBand: Math.min(0.08, (p.topBars.count * barArea(p.topBars.dia)) /
      (p.thickness * band * H)),
    rhoBotBand: Math.min(0.08, (p.bottomBars.count * barArea(p.bottomBars.dia)) /
      (p.thickness * band * H)),
    rhoStirrup: Math.min(0.04, (p.stirrup.legs * barArea(p.stirrup.dia) /
      p.stirrup.spacing) / p.thickness),
    bandFrac: band,
    barDia: p.bottomBars.dia,
  });

  // ---- render -----------------------------------------------------------
  const solids: ElementModel['render']['solids'] = [];
  solids.push({
    kind: 'box',
    center: [(xMin + xMax) / 2, H / 2, 0],
    size: [capLength, H, p.thickness],
    color: PALETTE.concrete, opacity: 0.5, role: 'concrete',
  });
  supports.forEach((s) => {
    solids.push({
      kind: 'box',
      center: [s.x, -110, 0],
      size: [p.supportWidth, 220, Math.min(p.thickness, 500)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });
  loads.forEach((l) => {
    solids.push({
      kind: 'box',
      center: [l.x, H + 90, 0],
      size: [340, 180, Math.min(p.thickness, 500)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });

  // ---- reinforcement ----------------------------------------------------
  const rebar: ElementModel['render']['rebar'] = [];
  const zc = p.thickness / 2 - cover;
  const x0 = xMin + cover, x1 = xMax - cover;
  const hook = Math.min(300, H - 2 * cover);
  // longitudinal bars stack clear of the closed stirrups (stirrup bar +
  // longitudinal bar), so the rendered cylinders sit on top of one another
  const dS = p.stirrup.dia;
  const gTop = dS + p.topBars.dia;
  const gBot = dS + p.bottomBars.dia;
  const zi = Math.max(8, zc - dS - Math.max(p.topBars.dia, p.bottomBars.dia));
  const across = (count: number): number[] => {
    const n = Math.max(2, Math.min(5, count));
    return Array.from({ length: n }, (_, i) => -zi + (i * 2 * zi) / (n - 1));
  };
  across(p.bottomBars.count).forEach((z, i) => {
    rebar.push({
      id: `bot-${i}`,
      points: [
        [x0, cover + hook, z], [x0, cover + gBot, z],
        [x1, cover + gBot, z], [x1, cover + hook, z],
      ],
      diameter: p.bottomBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  across(p.topBars.count).forEach((z, i) => {
    rebar.push({
      id: `top-${i}`,
      points: [[x0, H - cover - gTop, z], [x1, H - cover - gTop, z]],
      diameter: p.topBars.dia, role: 'distribution', color: PALETTE.rebarStirrup,
    });
  });
  const nStir = Math.max(2, Math.round((capLength - 2 * cover) / p.stirrup.spacing));
  for (let i = 0; i <= nStir; i++) {
    const x = x0 + (i * (x1 - x0)) / nStir;
    rebar.push({
      id: `stir-${i}`,
      points: [
        [x, cover, -zc], [x, H - cover, -zc], [x, H - cover, zc], [x, cover, zc],
      ],
      diameter: p.stirrup.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  const loadArrows: ElementModel['render']['loads'] = loads.map((l) => ({
    pos: [l.x, H + 200, 0] as [number, number, number],
    dir: [0, -1, 0] as [number, number, number],
    magnitude: l.P,
    label: `${(l.P / 1e3).toFixed(0)} kN`,
  }));

  return {
    truss: { nodes: tr.nodes, members: tr.members, loads: tr.loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    render: {
      solids, rebar, loads: loadArrows,
      bounds: {
        min: [xMin, -400, -p.thickness / 2],
        max: [xMax, H + 700, p.thickness / 2],
      },
    },
    ties, struts, nodes: nodeSpecs, notes, info: [], totalLoad,
  };
}
