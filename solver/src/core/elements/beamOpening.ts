/**
 * Beam with a Web Opening element module.
 *
 * A simply-supported beam pierced by a rectangular web opening — the
 * discontinuity region used as the illustrative example in the CSFM book
 * (§3.4, Fig. 3.4). The force flow detours around the opening, putting the
 * chords above and below it into a local Vierendeel-type action.
 */

import type { ElementModel, BarGroupCount, PointLoad } from './types';
import { PALETTE } from './types';
import type { StirrupGroup } from './hammerhead';
import { buildDeepBeamTruss } from './deepBeam';
import type { DeepBeamPanelLoad, DeepBeamSupport } from './deepBeam';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface BeamOpeningParams {
  span: number;          // centre-to-centre of supports (mm)
  height: number;        // depth H (mm)
  thickness: number;     // out-of-plane width (mm)
  supportWidth: number;
  openX: number;         // opening centre, x (mm, 0 = mid-span)
  openY: number;         // opening centre, y from soffit (mm)
  openW: number;         // opening width (mm)
  openH: number;         // opening height (mm)
  loads: PointLoad[];
  topBars: BarGroupCount;
  bottomBars: BarGroupCount;
  stirrup: StirrupGroup;
}

export function defaultBeamOpeningParams(): BeamOpeningParams {
  return {
    span: 5000,
    height: 1600,
    thickness: 350,
    supportWidth: 350,
    openX: 0,
    openY: 800,
    openW: 1000,
    openH: 600,
    loads: [
      { id: 'p1', x: -1200, load: 700e3 },
      { id: 'p2', x: 1200, load: 700e3 },
    ],
    topBars: { dia: 25, count: 5 },
    bottomBars: { dia: 25, count: 5 },
    stirrup: { dia: 12, spacing: 180, legs: 2 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildBeamOpening(p: BeamOpeningParams): ElementModel {
  const notes: string[] = [];
  const cover = 60;
  const H = p.height;
  const xL = -p.span / 2, xR = p.span / 2;
  const capLength = p.span;
  // clamp the opening inside the beam
  const ow = Math.min(p.openW, p.span - 600);
  const oh = Math.min(p.openH, H - 400);
  const ox = Math.max(xL + ow / 2 + 200, Math.min(xR - ow / 2 - 200, p.openX));
  const oy = Math.max(oh / 2 + 150, Math.min(H - oh / 2 - 150, p.openY));
  const o = { x0: ox - ow / 2, x1: ox + ow / 2, y0: oy - oh / 2, y1: oy + oh / 2 };
  const hole = (x: number, y: number) =>
    x > o.x0 && x < o.x1 && y > o.y0 && y < o.y1;

  notes.push(
    'Force flow detours around the web opening — the chords above and below ' +
      'act in local bending; provide trim reinforcement framing the opening.',
  );

  const topY = () => H - cover;
  const botY = () => cover;
  const supports: DeepBeamSupport[] = [
    { x: xL + p.supportWidth / 2, width: p.supportWidth, label: 'Support A' },
    { x: xR - p.supportWidth / 2, width: p.supportWidth, label: 'Support B' },
  ];
  const loads: DeepBeamPanelLoad[] = [];
  p.loads.forEach((l, i) => {
    if (l.x < xL - 1 || l.x > xR + 1) {
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
  const struts = tr.members
    .filter((m) => m.id.startsWith('D'))
    .map((m) => ({
      memberId: m.id, width: Math.min(p.thickness, 0.3 * H),
      thickness: p.thickness, type: 'bottle-reinforced' as const,
    }));
  const nodeSpecs = tr.nodes.map((nd) => ({
    id: nd.id,
    type: (nd.id.startsWith('B') ? 'CCT' : 'CCC') as 'CCC' | 'CCT',
    area: p.thickness * p.supportWidth,
  }));

  // ---- FE voxel domain --------------------------------------------------
  const cell = capLength / 30;
  const nx = Math.max(14, Math.round(capLength / cell));
  const ny = Math.max(8, Math.round(H / cell));
  const nz = Math.max(2, Math.round(p.thickness / cell));
  const dxF = capLength / nx, dyF = H / ny;
  const domain = {
    nx, ny, nz, dx: dxF, dy: dyF, dz: p.thickness / nz,
    origin: [xL, 0, -p.thickness / 2] as [number, number, number],
    filled: (i: number, j: number) =>
      !hole(xL + (i + 0.5) * dxF, (j + 0.5) * dyF),
  };
  const femLoads: FemLoad[] = loads.map((l) => ({ pos: [l.x, H, 0], f: [0, -l.P, 0] }));
  const femSupports: FemSupport[] = supports.map((s) => ({
    pos: [s.x, 0, 0],
    fix: [false, true, false] as [boolean, boolean, boolean],
    radius: p.supportWidth * 0.8,
  }));

  // ---- 2D continuum CSFM problem ----------------------------------------
  const band = 0.14;
  const continuum = buildElevationContinuum({
    xMin: xL, xMax: xR, topY: () => H, botY: () => 0,
    thickness: p.thickness, res: 40,
    loads: loads.map((l) => ({ x: l.x, P: l.P, width: Math.max(280, capLength / 18) })),
    supports: supports.map((s, i) => ({ x: s.x, width: s.width, fixX: i === 0 })),
    rhoTopBand: Math.min(0.08, (p.topBars.count * barArea(p.topBars.dia)) /
      (p.thickness * band * H)),
    rhoBotBand: Math.min(0.08, (p.bottomBars.count * barArea(p.bottomBars.dia)) /
      (p.thickness * band * H)),
    rhoStirrup: Math.min(0.04, (p.stirrup.legs * barArea(p.stirrup.dia) /
      p.stirrup.spacing) / p.thickness),
    bandFrac: band,
    barDia: p.bottomBars.dia,
    hole,
  });

  // ---- render (four members framing the opening) ------------------------
  const solids: ElementModel['render']['solids'] = [];
  const box = (cx: number, cy: number, sx: number, sy: number) => {
    if (sx > 1 && sy > 1)
      solids.push({
        kind: 'box',
        center: [cx, cy, 0], size: [sx, sy, p.thickness],
        color: PALETTE.concrete, opacity: 0.5, role: 'concrete',
      });
  };
  box((xL + xR) / 2, o.y0 / 2, p.span, o.y0);                       // below opening
  box((xL + xR) / 2, (o.y1 + H) / 2, p.span, H - o.y1);             // above opening
  box((xL + o.x0) / 2, oy, o.x0 - xL, oh);                          // left of opening
  box((o.x1 + xR) / 2, oy, xR - o.x1, oh);                          // right of opening
  supports.forEach((s) => {
    solids.push({
      kind: 'box', center: [s.x, -110, 0],
      size: [p.supportWidth, 220, Math.min(p.thickness, 460)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });
  loads.forEach((l) => {
    solids.push({
      kind: 'box', center: [l.x, H + 90, 0],
      size: [300, 180, Math.min(p.thickness, 460)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });

  // ---- reinforcement ----------------------------------------------------
  const rebar: ElementModel['render']['rebar'] = [];
  const zc = p.thickness / 2 - cover;
  // longitudinal bars stack clear of the closed stirrups (stirrup bar +
  // longitudinal bar), so the rendered cylinders sit on top of one another
  const dS = p.stirrup.dia;
  const gTop = dS + p.topBars.dia;
  const gBot = dS + p.bottomBars.dia;
  const zi = Math.max(8, zc - dS - Math.max(p.topBars.dia, p.bottomBars.dia));
  const across = (n0: number): number[] => {
    const n = Math.max(2, Math.min(5, n0));
    return Array.from({ length: n }, (_, i) => -zi + (i * 2 * zi) / (n - 1));
  };
  across(p.bottomBars.count).forEach((z, i) => {
    rebar.push({
      id: `bot-${i}`,
      points: [[xL + cover, cover + gBot, z], [xR - cover, cover + gBot, z]],
      diameter: p.bottomBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  across(p.topBars.count).forEach((z, i) => {
    rebar.push({
      id: `top-${i}`,
      points: [[xL + cover, H - cover - gTop, z], [xR - cover, H - cover - gTop, z]],
      diameter: p.topBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  // trim bars framing the opening (a closed loop just outside the hole)
  const m = 70;
  across(2).forEach((z, i) => {
    rebar.push({
      id: `trim-${i}`,
      points: [
        [o.x0 - m, o.y0 - m, z], [o.x1 + m, o.y0 - m, z],
        [o.x1 + m, o.y1 + m, z], [o.x0 - m, o.y1 + m, z],
      ],
      diameter: p.stirrup.dia + 4, role: 'main-tie', color: PALETTE.rebarMain,
      closed: true,
    });
  });
  // stirrups (omitted across the opening width)
  const nStir = Math.max(2, Math.round((capLength - 2 * cover) / p.stirrup.spacing));
  for (let i = 0; i <= nStir; i++) {
    const x = xL + cover + (i * (capLength - 2 * cover)) / nStir;
    if (x > o.x0 - m && x < o.x1 + m) continue;
    rebar.push({
      id: `stir-${i}`,
      points: [
        [x, cover, -zc], [x, H - cover, -zc],
        [x, H - cover, zc], [x, cover, zc],
      ],
      diameter: p.stirrup.dia, role: 'stirrup', color: PALETTE.rebarStirrup, closed: true,
    });
  }

  const loadArrows: ElementModel['render']['loads'] = loads.map((l) => ({
    pos: [l.x, H + 200, 0] as [number, number, number],
    dir: [0, -1, 0] as [number, number, number],
    magnitude: l.P, label: `${(l.P / 1e3).toFixed(0)} kN`,
  }));

  return {
    truss: { nodes: tr.nodes, members: tr.members, loads: tr.loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    render: {
      solids, rebar, loads: loadArrows,
      bounds: {
        min: [xL, -400, -p.thickness / 2],
        max: [xR, H + 700, p.thickness / 2],
      },
    },
    ties, struts, nodes: nodeSpecs, notes, info: [], totalLoad,
  };
}
