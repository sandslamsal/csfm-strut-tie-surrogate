/**
 * Pier Cap element module.
 *
 * Models a stepped reinforced-concrete pier cap: a wide cap band at the base,
 * a tapered transition, and a narrow stem at the top, with a concentrated load
 * on the stem and supports under the cap. This is the geometry of the
 * experimental pier-cap specimens of Geevar and Menon (2018), reported in the
 * CSFM book §6.5 (Kaufmann et al. 2020).
 *
 * Unlike the rectangular `pileCap` module, the diagonal-strut concrete area is
 * derived from the real confined section — out-of-plane thickness times the
 * loading-plate width — so the strut stress, and hence the concrete-crushing
 * failure mode, is captured faithfully (see validation/).
 *
 * Planar strut-and-tie model in the X-Y elevation (z = 0). Units: mm, N.
 */

import type { ElementModel, BarGroup, BarGroupCount } from './types';
import { PALETTE } from './types';
import type { TrussNode, TrussMember, TrussLoad } from '../stm/truss';
import type { FemLoad, FemSupport } from '../fem/fem3d';
import type { ContinuumProblem } from '../csfm/continuum';

export interface PierCapParams {
  capWidth: number;        // wide base width, X (mm)
  stemWidth: number;       // narrow stem width, X (mm)
  capBandHeight: number;   // constant-width cap band at the base, Y (mm)
  taperHeight: number;     // tapered transition, Y (mm)
  stemHeight: number;      // constant-width stem, Y (mm)
  thickness: number;       // out-of-plane thickness b, Z (mm)
  loadPlate: number;       // loading-plate width l_b on the stem (mm)
  columnLoad: number;      // applied vertical load (N, +ve compression)
  supports: number;        // number of supports under the cap base
  supportPlate: number;    // bearing-plate width per support (mm)
  edgeDistance: number;    // support centre to cap edge (mm)
  // reinforcement — Geevar & Menon (2018) / CSFM book §6.5 notation
  mainBars: BarGroupCount;  // primary bottom tension reinforcement  A_s1
  extraBars: BarGroupCount; // additional bottom layer  A_s2  (count 0 = none)
  skinBars: BarGroup;       // distributed horizontal reinforcement  A_h
  vertBars: BarGroup;       // distributed vertical reinforcement  A_v
}

/** Default parameters — the Geevar & Menon (2018) specimen S1 (CSFM book §6.5). */
export function defaultPierCapParams(): PierCapParams {
  return {
    capWidth: 1200,
    stemWidth: 640,
    capBandHeight: 160,
    taperHeight: 320,
    stemHeight: 500,
    thickness: 300,
    loadPlate: 130,
    columnLoad: 800e3,      // service design load — adequate for this section
    supports: 2,
    supportPlate: 150,
    edgeDistance: 136,
    // Geevar & Menon (2018) specimen S1: 9 No.10 primary bars, no second
    // layer; distributed horizontal + vertical reinforcement on both faces.
    mainBars: { dia: 10, count: 9 },
    extraBars: { dia: 8, count: 0 },
    skinBars: { dia: 8, spacing: 90 },
    vertBars: { dia: 8, spacing: 90 },
  };
}

const barArea = (d: number) => (Math.PI / 4) * d * d;

export function buildPierCap(p: PierCapParams): ElementModel {
  const notes: string[] = [];
  const info: string[] = [];
  const H = p.capBandHeight + p.taperHeight + p.stemHeight;

  /** concrete half-width of the stepped outline at elevation y */
  const halfWidthAt = (y: number): number => {
    if (y <= p.capBandHeight) return p.capWidth / 2;
    if (y >= p.capBandHeight + p.taperHeight) return p.stemWidth / 2;
    const t = (y - p.capBandHeight) / p.taperHeight;
    return (p.capWidth + t * (p.stemWidth - p.capWidth)) / 2;
  };

  // ---- support layout (single row along the cap base) -------------------
  const nSup = Math.max(2, Math.round(p.supports));
  const span = p.capWidth - 2 * p.edgeDistance;
  const supX: number[] = [];
  for (let i = 0; i < nSup; i++) {
    supX.push(nSup > 1 ? -span / 2 + (i * span) / (nSup - 1) : 0);
  }

  // ---- truss nodes ------------------------------------------------------
  const nodes: TrussNode[] = [];
  // load / nodal zone at the stem top
  nodes.push({
    id: 'LOAD', x: 0, y: H, z: 0,
    fixed: [false, false, true],
    nodeType: 'CCC', label: 'Load nodal zone',
  });
  supX.forEach((x, i) => {
    nodes.push({
      id: `SUP${i}`, x, y: 0, z: 0,
      fixed: [i === 0, true, true],     // vertical reaction; SUP0 also locks X
      nodeType: 'CCT', label: `Support ${i + 1}`,
    });
  });

  // ---- members ----------------------------------------------------------
  const Ec = 30000;   // MPa, stiffness only
  const Es = 200000;
  const members: TrussMember[] = [];
  // diagonal struts: load -> each support. Concrete area = b x loading plate
  // (the real confined section), NOT a pile diameter.
  const strutArea = p.thickness * p.loadPlate;
  supX.forEach((_, i) => {
    members.push({
      id: `S${i}`, ni: 'LOAD', nj: `SUP${i}`,
      kind: 'strut', area: strutArea, E: Ec,
    });
  });
  // bottom tie chain along the cap base — carries the primary (A_s1) plus
  // any additional (A_s2) bottom reinforcement
  const tieSteel = p.mainBars.count * barArea(p.mainBars.dia)
    + p.extraBars.count * barArea(p.extraBars.dia);
  for (let i = 0; i < nSup - 1; i++) {
    members.push({
      id: `T${i}`, ni: `SUP${i}`, nj: `SUP${i + 1}`,
      kind: 'tie', area: tieSteel, E: Es,
    });
  }

  // ---- loads ------------------------------------------------------------
  const loads: TrussLoad[] = [{ node: 'LOAD', fx: 0, fy: -p.columnLoad, fz: 0 }];

  // ---- design data (ties / struts / nodal zones) ------------------------
  const ties = members
    .filter((m) => m.kind === 'tie')
    .map((m) => ({
      memberId: m.id,
      barDiameter: p.mainBars.dia,
      barCount: p.mainBars.count + p.extraBars.count,
      asProvided: tieSteel,
      asRequired: 0,
      group: 'mainBars',
    }));
  const struts = members
    .filter((m) => m.kind === 'strut')
    .map((m) => ({
      memberId: m.id,
      width: p.loadPlate,
      thickness: p.thickness,
      type: 'bottle-reinforced' as const,
    }));
  const nodeSpecs = [
    { id: 'LOAD', type: 'CCC' as const, area: p.loadPlate * p.thickness },
    ...supX.map((_, i) => ({
      id: `SUP${i}`, type: 'CCT' as const,
      area: p.supportPlate * p.thickness,
    })),
  ];

  if (p.stemWidth >= p.capWidth) {
    notes.push('Stem width is not narrower than the cap — check the geometry.');
  }
  info.push(
    'Diagonal-strut concrete area taken as out-of-plane thickness x loading-'
    + 'plate width (real confined section).',
  );

  // ---- 2D continuum CSFM problem (stepped elevation) --------------------
  const continuum = buildPierCapContinuum(p, H, halfWidthAt, supX, tieSteel);

  // ---- FE voxel domain --------------------------------------------------
  const res = 16;
  const cell = Math.max(p.capWidth, H) / res;
  const nx = Math.max(4, Math.round(p.capWidth / cell));
  const ny = Math.max(4, Math.round(H / cell));
  const nz = Math.max(2, Math.round(p.thickness / cell));
  const domain = {
    nx, ny, nz,
    dx: p.capWidth / nx, dy: H / ny, dz: p.thickness / nz,
    origin: [-p.capWidth / 2, 0, -p.thickness / 2] as [number, number, number],
    filled: (i: number, j: number) => {
      const x = -p.capWidth / 2 + (i + 0.5) * (p.capWidth / nx);
      const y = (j + 0.5) * (H / ny);
      return Math.abs(x) <= halfWidthAt(y);
    },
  };
  const femLoads: FemLoad[] = [{ pos: [0, H, 0], f: [0, -p.columnLoad, 0] }];
  const femSupports: FemSupport[] = supX.map((x) => ({
    pos: [x, 0, 0],
    fix: [false, true, false] as [boolean, boolean, boolean],
    radius: p.supportPlate * 0.6,
  }));

  // ---- render geometry --------------------------------------------------
  const render = buildRender(p, H, supX, halfWidthAt);

  return {
    truss: { nodes, members, loads },
    fem: { domain, loads: femLoads, supports: femSupports },
    continuum,
    continuumOrient: 'x',
    render,
    ties,
    struts,
    nodes: nodeSpecs,
    notes,
    info,
    totalLoad: p.columnLoad,
  };
}

/** Build the 2D plane-stress continuum problem over the stepped outline. */
function buildPierCapContinuum(
  p: PierCapParams,
  H: number,
  halfWidthAt: (y: number) => number,
  supX: number[],
  tieSteel: number,
): ContinuumProblem {
  const nx = 40;
  const dx = p.capWidth / nx;
  const ny = Math.max(8, Math.round(H / dx));
  const dy = H / ny;
  const band = Math.max(dy, p.capBandHeight);
  const rhoBot = Math.min(0.08, tieSteel / (p.thickness * band));

  return {
    nx, ny, dx, dy,
    origin: [-p.capWidth / 2, 0],
    thickness: p.thickness,
    filled: (i, j) => {
      const x = -p.capWidth / 2 + (i + 0.5) * dx;
      const y = (j + 0.5) * dy;
      return y >= 0 && y <= H && Math.abs(x) <= halfWidthAt(y);
    },
    rhoX: (_, j) => ((j + 0.5) * dy <= band ? rhoBot : 0.0012),
    rhoY: () => 0.0015,
    loads: distributeLoad(p.columnLoad, p.loadPlate, H, dx),
    supports: supX.map((x, i) => ({
      pos: [x, 0] as [number, number],
      radius: p.supportPlate / 2,
      fix: [i === 0, true] as [boolean, boolean],
    })),
    barDia: p.mainBars.dia,
  };
}

/** Spread a concentrated load over its bearing plate as point loads. */
function distributeLoad(
  P: number, width: number, y: number, dx: number,
): ContinuumProblem['loads'] {
  const n = Math.max(1, Math.round(width / dx));
  const out: ContinuumProblem['loads'] = [];
  for (let k = 0; k < n; k++) {
    const x = -width / 2 + (k + 0.5) * (width / n);
    out.push({ pos: [x, y], f: [0, -P / n] });
  }
  return out;
}

function buildRender(
  p: PierCapParams, H: number, supX: number[],
  halfWidthAt: (y: number) => number,
): ElementModel['render'] {
  const cw = p.capWidth / 2, sw = p.stemWidth / 2;
  const yTaper = p.capBandHeight + p.taperHeight;
  // stepped outline polygon in the X-Y elevation
  const profile: [number, number][] = [
    [-cw, 0], [cw, 0],
    [cw, p.capBandHeight], [sw, yTaper], [sw, H],
    [-sw, H], [-sw, yTaper], [-cw, p.capBandHeight],
  ];
  const solids: ElementModel['render']['solids'] = [
    {
      kind: 'prism', profile, zCenter: 0, zDepth: p.thickness,
      color: PALETTE.concrete, opacity: 0.55, role: 'concrete',
    },
  ];
  // support bearing blocks under the cap base
  supX.forEach((x) => {
    solids.push({
      kind: 'box',
      center: [x, -100, 0],
      size: [p.supportPlate, 200, p.thickness],
      color: PALETTE.bearing, opacity: 0.9, role: 'bearing',
    });
  });
  // loading-plate bearing block on top of the stem — shows where N is applied
  solids.push({
    kind: 'box',
    center: [0, H + 70, 0],
    size: [p.loadPlate, 140, Math.min(p.thickness, p.stemWidth)],
    color: PALETTE.bearing, opacity: 1, role: 'bearing',
  });

  // ---- reinforcement: an orthogonal mesh conforming to the stepped outline,
  //      plus the concentrated primary bars and a perimeter stirrup ---------
  const cover = 40;
  const rebar: ElementModel['render']['rebar'] = [];
  const zHalf = p.thickness / 2 - cover;
  const faces = [-zHalf, zHalf];
  const yTaperTop = p.capBandHeight + p.taperHeight;

  /** half-width of the cage (outline less cover) at elevation y */
  const cageHalf = (y: number): number => Math.max(0, halfWidthAt(y) - cover);
  /** top elevation of the cage above x (stepped outline less cover) */
  const cageTop = (x: number): number => {
    if (Math.abs(x) <= p.stemWidth / 2) return H - cover;
    const t = (2 * Math.abs(x) - p.capWidth) / (p.stemWidth - p.capWidth);
    return p.capBandHeight + t * p.taperHeight - cover;
  };
  /** evenly spread `n` bars across the section depth (z) */
  const acrossZ = (n: number): number[] => {
    const m = Math.max(2, Math.min(6, n));
    return Array.from({ length: m }, (_, i) => -zHalf + (i * 2 * zHalf) / (m - 1));
  };

  // cap + taper: horizontal distributed bars A_h every s_h; stem (upper
  // straight part): closed stirrups looping the section instead
  const sh = Math.max(60, p.skinBars.spacing);
  const nH = Math.max(1, Math.floor((H - 2 * cover) / sh));
  const swI = p.stemWidth / 2 - cover;
  for (let i = 0; i <= nH; i++) {
    const y = cover + (i * (H - 2 * cover)) / nH;
    if (y <= yTaperTop) {
      const hw = cageHalf(y);
      if (hw < 30) continue;
      faces.forEach((z, k) => {
        rebar.push({
          id: `h-${i}-${k}`,
          points: [[-hw, y, z], [hw, y, z]],
          diameter: p.skinBars.dia,
          role: 'distribution',
          color: PALETTE.rebarStirrup,
        });
      });
    } else {
      rebar.push({
        id: `stem-stir-${i}`,
        points: [
          [-swI, y, -zHalf], [swI, y, -zHalf],
          [swI, y, zHalf], [-swI, y, zHalf],
        ],
        diameter: p.skinBars.dia,
        role: 'stirrup',
        color: PALETTE.rebarStirrup,
        closed: true,
      });
    }
  }

  // vertical mesh — distributed bars A_v every s_v across the cap width, each
  // rising from the base to the stepped outline; laid clear of the horizontal
  // mesh (horizontal bar + vertical bar) so the two layers overlay, not cross
  const vGap = p.skinBars.dia + p.vertBars.dia;
  const vFaces = [-zHalf + vGap, zHalf - vGap];
  const sv = Math.max(60, p.vertBars.spacing);
  const nV = Math.max(1, Math.round((p.capWidth - 2 * cover) / sv));
  for (let i = 0; i <= nV; i++) {
    const x = -p.capWidth / 2 + cover + (i * (p.capWidth - 2 * cover)) / nV;
    const yTop = cageTop(x);
    if (yTop < cover + 60) continue;
    vFaces.forEach((z, k) => {
      rebar.push({
        id: `v-${i}-${k}`,
        points: [[x, cover, z], [x, yTop, z]],
        diameter: p.vertBars.dia,
        role: 'distribution',
        color: PALETTE.rebarStirrup,
      });
    });
  }

  // closed perimeter stirrup following the stepped outline, on both faces
  const cwI = p.capWidth / 2 - cover;
  const loopOutline: [number, number][] = [
    [-cwI, cover], [cwI, cover],
    [cwI, p.capBandHeight], [swI, yTaperTop],
    [swI, H - cover], [-swI, H - cover],
    [-swI, yTaperTop], [-cwI, p.capBandHeight],
  ];
  faces.forEach((z, k) => {
    rebar.push({
      id: `loop-${k}`,
      points: loopOutline.map(([x, y]) => [x, y, z] as [number, number, number]),
      diameter: p.vertBars.dia,
      role: 'stirrup',
      color: PALETTE.rebarStirrup,
      closed: true,
    });
  });

  // primary bottom reinforcement A_s1 — concentrated horizontal bars along the
  // cap base, hooked up at each end; A_s2 a second layer just above
  const xHalf = p.capWidth / 2 - cover;
  const hookH = Math.min(110, p.capBandHeight - cover);
  const mainBar = (y: number, z: number): [number, number, number][] => ([
    [-xHalf, y + hookH, z], [-xHalf, y, z],
    [xHalf, y, z], [xHalf, y + hookH, z],
  ]);
  acrossZ(p.mainBars.count).forEach((z, i) => {
    rebar.push({
      id: `s1-${i}`, points: mainBar(cover, z),
      diameter: p.mainBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
    });
  });
  if (p.extraBars.count > 0) {
    const yE = cover + p.mainBars.dia + 30;
    acrossZ(p.extraBars.count).forEach((z, i) => {
      rebar.push({
        id: `s2-${i}`, points: mainBar(yE, z),
        diameter: p.extraBars.dia, role: 'main-tie', color: PALETTE.rebarMain,
      });
    });
  }

  // surface skin bars along each inclined (tapered) face — longitudinal bars
  // hugging the taper surface, spread across the thickness (the inner bar sits
  // on the skin between the two outer face bars); they turn a little into the
  // cap band and the stem
  const yTb = p.capBandHeight;
  const yTt = yTaperTop;
  const skinRows = [-zHalf, 0, zHalf];   // on the skin, across the thickness
  ([-1, 1] as const).forEach((sgn) => {
    skinRows.forEach((z, k) => {
      rebar.push({
        id: `skin-${sgn > 0 ? 'p' : 'n'}-${k}`,
        points: [
          [sgn * (halfWidthAt(yTb) - cover), Math.max(cover, yTb - 90), z],
          [sgn * (halfWidthAt(yTb) - cover), yTb, z],
          [sgn * (halfWidthAt(yTt) - cover), yTt, z],
          [sgn * (halfWidthAt(yTt) - cover), Math.min(H - cover, yTt + 90), z],
        ],
        diameter: p.skinBars.dia,
        role: 'distribution',
        color: PALETTE.rebarMain,
      });
    });
  });

  // closed-loop stirrups across the tapered area — closed ties wrapping the
  // cross-section at intervals down the taper
  const nTap = Math.max(2, Math.round(p.taperHeight / Math.max(70, p.vertBars.spacing)));
  for (let i = 1; i < nTap; i++) {
    const y = yTb + (i * p.taperHeight) / nTap;
    const hw = cageHalf(y);
    if (hw < 30) continue;
    rebar.push({
      id: `tstir-${i}`,
      points: [
        [-hw, y, -zHalf], [hw, y, -zHalf],
        [hw, y, zHalf], [-hw, y, zHalf],
      ],
      diameter: p.vertBars.dia,
      role: 'stirrup',
      color: PALETTE.rebarStirrup,
      closed: true,
    });
  }

  // column (stem) longitudinal reinforcement — vertical bars at the stem skin,
  // running the full stem height and continuing along the tapered side down
  // into the cap band; spread across the thickness
  const colX = p.stemWidth / 2 - cover;
  ([-1, 1] as const).forEach((sgn) => {
    skinRows.forEach((z, k) => {
      rebar.push({
        id: `col-${sgn > 0 ? 'p' : 'n'}-${k}`,
        points: [
          [sgn * (p.capWidth / 2 - cover), p.capBandHeight, z],
          [sgn * colX, yTt, z],
          [sgn * colX, H - cover, z],
        ],
        diameter: p.mainBars.dia,
        role: 'main-tie',
        color: PALETTE.rebarMain,
      });
    });
  });

  // load arrow sitting just above the loading plate, like the other elements
  const loads: ElementModel['render']['loads'] = [{
    pos: [0, H + 200, 0],
    dir: [0, -1, 0],
    magnitude: p.columnLoad,
    label: `N = ${(p.columnLoad / 1e3).toFixed(0)} kN`,
  }];

  return {
    solids, rebar, loads,
    bounds: {
      min: [-p.capWidth / 2, -300, -p.thickness / 2],
      max: [p.capWidth / 2, H + 360, p.thickness / 2],
    },
  };
}
