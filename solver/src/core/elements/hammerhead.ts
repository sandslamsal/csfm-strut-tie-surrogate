/**
 * Hammerhead Bent Cap element module.
 *
 * A single-column pier with a cap cantilevering to both sides, carrying girder
 * reactions on bearings along its top. Modelled as a deep-beam strut-and-tie
 * truss whose chords follow the cap geometry — the top (tension) chord just
 * below the deck and the bottom (compression) chord just above the sloped
 * soffit — so all struts, ties, nodes and the FE stress field stay inside the
 * concrete. The single column is two support nodes at the column faces.
 */

import type { ElementModel, BarGroupCount, PointLoad } from './types';
import { PALETTE } from './types';
import { buildDeepBeamTruss } from './deepBeam';
import type { DeepBeamPanelLoad, DeepBeamSupport } from './deepBeam';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface StirrupGroup {
  dia: number;
  spacing: number;
  legs: number;
}

export interface HammerheadParams {
  capLength: number;       // total length X (mm)
  capDepthCenter: number;  // depth over the column (mm)
  capDepthTip: number;     // depth at the cantilever tips (mm)
  capWidth: number;        // Z (mm)
  columnWidth: number;     // X (mm)
  columnDepth: number;     // Z (mm)
  columnHeight: number;    // mm (render)
  bearings: PointLoad[];   // editable bearing loads
  topBars: BarGroupCount;     // top (tension) longitudinal bars
  bottomBars: BarGroupCount;  // bottom longitudinal bars
  stirrup: StirrupGroup;
}

export function defaultHammerheadParams(): HammerheadParams {
  const n = 6, spc = 1500;
  const bearings: PointLoad[] = [];
  for (let i = 0; i < n; i++) {
    bearings.push({
      id: `b${i + 1}`,
      x: -((n - 1) * spc) / 2 + i * spc,
      load: 1400e3,
    });
  }
  return {
    capLength: 9000,
    capDepthCenter: 2200,
    capDepthTip: 1100,
    capWidth: 1800,
    columnWidth: 1800,
    columnDepth: 1800,
    columnHeight: 6000,
    bearings,
    topBars: { dia: 32, count: 12 },
    bottomBars: { dia: 25, count: 6 },
    stirrup: { dia: 16, spacing: 200, legs: 4 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildHammerhead(p: HammerheadParams): ElementModel {
  const notes: string[] = [];
  const Dc = p.capDepthCenter;
  const Dt = p.capDepthTip;
  const L = p.capLength;
  const cw = p.columnWidth;
  const soffitTip = Math.max(0, Dc - Dt);
  const cover = 90;

  /** soffit elevation — flat over the column, sloped to the tips */
  const soffitY = (x: number): number => {
    const ax = Math.abs(x);
    if (ax <= cw / 2) return 0;
    const t = (ax - cw / 2) / Math.max(1, L / 2 - cw / 2);
    return soffitTip * Math.min(1, Math.max(0, t));
  };

  // chord elevations — kept inside the concrete
  const topY = () => Dc - cover;
  const botY = (x: number) => soffitY(x) + cover;

  // bearing loads on top
  const loads: DeepBeamPanelLoad[] = [];
  p.bearings.forEach((b, i) => {
    if (Math.abs(b.x) > L / 2) {
      notes.push(`Bearing ${i + 1} is outside the cap length — ignored.`);
      return;
    }
    loads.push({ x: b.x, P: b.load, label: `Bearing ${i + 1}` });
  });
  if (loads.length === 0) notes.push('No bearing loads defined.');

  const supports: DeepBeamSupport[] = [
    { x: -cw / 2, width: cw / 2, label: 'Column face L' },
    { x: +cw / 2, width: cw / 2, label: 'Column face R' },
  ];

  const tr = buildDeepBeamTruss({
    topY, botY, width: p.capWidth, loads, supports,
    barDiameter: p.topBars.dia,
  });
  const totalLoad = loads.reduce((s, l) => s + l.P, 0);

  // ---- design data ------------------------------------------------------
  const nV = tr.members.filter((m) => m.id.startsWith('V')).length;
  const stirTrib = L / Math.max(1, nV);
  const asStirrupPerV =
    p.stirrup.legs * barArea(p.stirrup.dia) * (stirTrib / p.stirrup.spacing);

  const ties = tr.members
    .filter((m) => m.id.startsWith('TC') || m.id.startsWith('V'))
    .map((m) => {
      const isV = m.id.startsWith('V');
      const dia = isV ? p.stirrup.dia : p.topBars.dia;
      const as = isV ? asStirrupPerV : p.topBars.count * barArea(p.topBars.dia);
      const count = isV ? p.stirrup.legs : p.topBars.count;
      return {
        memberId: m.id, barDiameter: dia, barCount: count,
        asProvided: as, asRequired: 0, group: isV ? 'stirrup' : 'topBars',
      };
    });
  const struts = tr.members
    .filter((m) => m.id.startsWith('D') || m.id.startsWith('BC'))
    .map((m) => ({
      memberId: m.id,
      width: Math.min(p.capWidth, 0.3 * Dc),
      thickness: p.capWidth,
      type: 'bottle-reinforced' as const,
    }));
  const nodeSpecs = tr.nodes.map((nd) => ({
    id: nd.id,
    type: (nd.id.startsWith('B') ? 'CCT' : 'CCC') as 'CCC' | 'CCT',
    area: p.capWidth * Math.min(cw, 0.25 * L),
  }));

  // ---- FE voxel domain (tapered — fills the cap above the soffit) -------
  const res = 26;
  const cell = L / res;
  const nx = Math.max(10, Math.round(L / cell));
  const ny = Math.max(5, Math.round(Dc / cell));
  const nz = Math.max(2, Math.round(p.capWidth / cell));
  const dx = L / nx, dy = Dc / ny;
  const domain = {
    nx, ny, nz,
    dx, dy, dz: p.capWidth / nz,
    origin: [-L / 2, 0, -p.capWidth / 2] as [number, number, number],
    filled: (i: number, j: number) => {
      const x = -L / 2 + (i + 0.5) * dx;
      const yc = (j + 0.5) * dy;
      return yc >= soffitY(x) - dy * 0.5;
    },
  };
  const femLoads: FemLoad[] = loads.map((l) => ({
    pos: [l.x, Dc, 0], f: [0, -l.P, 0],
  }));
  const femSupports: FemSupport[] = [
    { pos: [0, 0, 0], fix: [false, true, false] as [boolean, boolean, boolean],
      radius: cw * 0.6 },
  ];

  // ---- render -----------------------------------------------------------
  const solids: ElementModel['render']['solids'] = [];
  solids.push({
    kind: 'prism',
    profile: [
      [-L / 2, Dc], [L / 2, Dc], [L / 2, soffitTip],
      [cw / 2, 0], [-cw / 2, 0], [-L / 2, soffitTip],
    ],
    zCenter: 0, zDepth: p.capWidth,
    color: PALETTE.concrete, opacity: 0.5, role: 'concrete',
  });
  solids.push({
    kind: 'box',
    center: [0, -p.columnHeight / 2, 0],
    size: [cw, p.columnHeight, p.columnDepth],
    color: PALETTE.column, opacity: 0.95, role: 'column',
  });
  loads.forEach((l) => {
    solids.push({
      kind: 'box',
      center: [l.x, Dc + 90, 0],
      size: [400, 180, Math.min(600, p.capWidth)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });

  // ---- reinforcement ----------------------------------------------------
  const rebar: ElementModel['render']['rebar'] = [];
  const zc = p.capWidth / 2 - cover;
  // longitudinal bars stack clear of the closed stirrups — offset by the
  // stirrup bar plus the longitudinal bar, so the rendered cylinders sit on
  // top of one another rather than intersecting
  const dS = p.stirrup.dia;
  const gTop = dS + p.topBars.dia;
  const gBot = dS + p.bottomBars.dia;
  const zi = Math.max(8, zc - dS - Math.max(p.topBars.dia, p.bottomBars.dia));
  const placeAcross = (count: number): number[] => {
    const n = Math.max(2, count > 6 ? 6 : count);
    return Array.from({ length: n }, (_, i) => -zi + (i * 2 * zi) / (n - 1));
  };
  // top bars
  placeAcross(p.topBars.count).forEach((z, i) => {
    rebar.push({
      id: `top-${i}`,
      points: [[-L / 2 + cover, Dc - cover - gTop, z], [L / 2 - cover, Dc - cover - gTop, z]],
      diameter: p.topBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  // bottom bars along the soffit
  placeAcross(p.bottomBars.count).forEach((z, i) => {
    const pts: [number, number, number][] = [];
    for (let j = 0; j <= 24; j++) {
      const x = -L / 2 + cover + (j * (L - 2 * cover)) / 24;
      pts.push([x, soffitY(x) + cover + gBot, z]);
    }
    rebar.push({
      id: `bot-${i}`, points: pts,
      diameter: p.bottomBars.dia, role: 'distribution', color: PALETTE.rebarStirrup,
    });
  });
  // closed stirrups
  const nStir = Math.max(2, Math.round((L - 2 * cover) / p.stirrup.spacing));
  for (let i = 0; i <= nStir; i++) {
    const x = -L / 2 + cover + (i * (L - 2 * cover)) / nStir;
    const yb = soffitY(x) + cover;
    rebar.push({
      id: `stir-${i}`,
      points: [
        [x, yb, -zc], [x, Dc - cover, -zc], [x, Dc - cover, zc], [x, yb, zc],
      ],
      diameter: p.stirrup.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  const loadArrows: ElementModel['render']['loads'] = loads.map((l) => ({
    pos: [l.x, Dc + 200, 0] as [number, number, number],
    dir: [0, -1, 0] as [number, number, number],
    magnitude: l.P,
    label: `${(l.P / 1e3).toFixed(0)} kN`,
  }));

  // ---- 2D continuum CSFM problem (elevation) ----------------------------
  const band = 0.18;
  const continuum = buildElevationContinuum({
    xMin: -L / 2, xMax: L / 2,
    topY: () => Dc,
    botY: soffitY,
    thickness: p.capWidth,
    res: 36,
    loads: loads.map((l) => ({ x: l.x, P: l.P, width: Math.max(400, L / 14) })),
    supports: [
      { x: -cw / 2, width: cw / 2, fixX: true },
      { x: cw / 2, width: cw / 2, fixX: false },
    ],
    rhoTopBand: Math.min(0.08, (p.topBars.count * barArea(p.topBars.dia)) /
      (p.capWidth * band * Dc)),
    rhoBotBand: Math.min(0.08, (p.bottomBars.count * barArea(p.bottomBars.dia)) /
      (p.capWidth * band * Dc)),
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
        min: [-L / 2, -p.columnHeight, -p.capWidth / 2],
        max: [L / 2, Dc + 700, p.capWidth / 2],
      },
    },
    ties, struts, nodes: nodeSpecs, notes, info: [], totalLoad,
  };
}
