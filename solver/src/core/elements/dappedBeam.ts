/**
 * Dapped-End Beam element module.
 *
 * A simply-supported beam with a re-entrant (dapped) corner at each end — the
 * end is notched so a reduced-depth nib bears on a lower support. The dap is a
 * classic discontinuity region (named in the CSFM book's introduction): a
 * vertical hanger tie at the re-entrant corner suspends the end reaction into
 * the full-depth beam.
 */

import type { ElementModel, BarGroupCount, PointLoad } from './types';
import { PALETTE } from './types';
import type { StirrupGroup } from './hammerhead';
import { buildDeepBeamTruss } from './deepBeam';
import type { DeepBeamPanelLoad, DeepBeamSupport } from './deepBeam';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import { buildElevationContinuum } from './continuum2d';

export interface DappedBeamParams {
  span: number;          // centre-to-centre of bearings (mm)
  height: number;        // full depth H (mm)
  thickness: number;     // out-of-plane width (mm)
  dapLength: number;     // length of the dap / nib (mm)
  nibDepth: number;      // depth of the reduced-depth nib (mm)
  supportWidth: number;  // bearing width (mm)
  loads: PointLoad[];    // top-edge loads
  topBars: BarGroupCount;
  bottomBars: BarGroupCount;
  hangerBars: BarGroupCount; // vertical hangers at the re-entrant corners
  stirrup: StirrupGroup;
}

export function defaultDappedBeamParams(): DappedBeamParams {
  return {
    span: 5000,
    height: 1300,
    thickness: 400,
    dapLength: 700,
    nibDepth: 650,
    supportWidth: 300,
    loads: [{ id: 'p1', x: 0, load: 1000e3 }],
    topBars: { dia: 20, count: 4 },
    bottomBars: { dia: 28, count: 6 },
    hangerBars: { dia: 16, count: 6 },
    stirrup: { dia: 12, spacing: 200, legs: 2 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildDappedBeam(p: DappedBeamParams): ElementModel {
  const notes: string[] = [];
  const cover = 60;
  const H = p.height;
  const Ld = p.dapLength;
  const dn = Math.min(p.nibDepth, H - 100);
  const xL = -p.span / 2 - Ld / 2;          // beam ends
  const xR = p.span / 2 + Ld / 2;
  const capLength = xR - xL;
  const reLx = xL + Ld;                      // re-entrant corner x (left)
  const reRx = xR - Ld;
  const nibTop = H;                          // nib is the top portion
  const nibSoffit = H - dn;

  /** soffit elevation — full depth in the span, raised to the nib at the ends */
  const botY = (x: number): number => (x < reLx || x > reRx ? nibSoffit : 0);
  const topY = () => H;

  const supports: DeepBeamSupport[] = [
    { x: xL + p.supportWidth, width: p.supportWidth, label: 'Bearing A (nib)' },
    { x: xR - p.supportWidth, width: p.supportWidth, label: 'Bearing B (nib)' },
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
  notes.push(
    'Dapped ends — provide a full vertical hanger tie at each re-entrant ' +
      'corner to suspend the bearing reaction into the full-depth beam.',
  );

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
    filled: (i: number, j: number) => {
      const x = xL + (i + 0.5) * dxF;
      return (j + 0.5) * dyF >= botY(x) - dyF * 0.5;
    },
  };
  const femLoads: FemLoad[] = loads.map((l) => ({ pos: [l.x, H, 0], f: [0, -l.P, 0] }));
  const femSupports: FemSupport[] = supports.map((s) => ({
    pos: [s.x, nibSoffit, 0],
    fix: [false, true, false] as [boolean, boolean, boolean],
    radius: p.supportWidth * 0.8,
  }));

  // ---- 2D continuum CSFM problem ----------------------------------------
  const band = 0.16;
  const continuum = buildElevationContinuum({
    xMin: xL, xMax: xR, topY, botY,
    thickness: p.thickness, res: 38,
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
  // dapped profile (8-point polygon — notched bottom corners)
  solids.push({
    kind: 'prism',
    profile: [
      [xL, H], [xR, H], [xR, nibSoffit], [reRx, nibSoffit],
      [reRx, 0], [reLx, 0], [reLx, nibSoffit], [xL, nibSoffit],
    ],
    zCenter: 0, zDepth: p.thickness,
    color: PALETTE.concrete, opacity: 0.5, role: 'concrete',
  });
  supports.forEach((s) => {
    solids.push({
      kind: 'box',
      center: [s.x, nibSoffit - 110, 0],
      size: [p.supportWidth, 220, Math.min(p.thickness, 460)],
      color: PALETTE.bearing, opacity: 1, role: 'bearing',
    });
  });
  loads.forEach((l) => {
    solids.push({
      kind: 'box',
      center: [l.x, H + 90, 0],
      size: [320, 180, Math.min(p.thickness, 460)],
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
  const zi = Math.max(8, zc - dS - Math.max(p.topBars.dia, p.bottomBars.dia,
    p.hangerBars.dia));
  const across = (n0: number): number[] => {
    const n = Math.max(2, Math.min(5, n0));
    return Array.from({ length: n }, (_, i) => -zi + (i * 2 * zi) / (n - 1));
  };
  // bottom main bars follow the stepped soffit
  across(p.bottomBars.count).forEach((z, i) => {
    const pts: [number, number, number][] = [];
    const N = 40;
    for (let k = 0; k <= N; k++) {
      const x = xL + cover + (k * (capLength - 2 * cover)) / N;
      pts.push([x, botY(x) + cover + gBot, z]);
    }
    rebar.push({ id: `bot-${i}`, points: pts,
      diameter: p.bottomBars.dia, role: 'main-tie', color: PALETTE.rebarMain });
  });
  // top bars
  across(p.topBars.count).forEach((z, i) => {
    rebar.push({
      id: `top-${i}`,
      points: [[xL + cover, H - cover - gTop, z], [xR - cover, H - cover - gTop, z]],
      diameter: p.topBars.dia, role: 'distribution', color: PALETTE.rebarStirrup,
    });
  });
  // vertical hanger ties at the two re-entrant corners
  [reLx + cover, reRx - cover].forEach((hx, c) => {
    const dz = (2 * zi) / Math.max(1, p.hangerBars.count - 1);
    for (let k = 0; k < p.hangerBars.count; k++) {
      const z = -zi + k * dz;
      rebar.push({
        id: `hang-${c}-${k}`,
        points: [[hx, cover, z], [hx, H - cover, z]],
        diameter: p.hangerBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
      });
    }
  });
  // stirrups
  const nStir = Math.max(2, Math.round((capLength - 2 * cover) / p.stirrup.spacing));
  for (let i = 0; i <= nStir; i++) {
    const x = xL + cover + (i * (capLength - 2 * cover)) / nStir;
    rebar.push({
      id: `stir-${i}`,
      points: [
        [x, botY(x) + cover, -zc], [x, H - cover, -zc],
        [x, H - cover, zc], [x, botY(x) + cover, zc],
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
