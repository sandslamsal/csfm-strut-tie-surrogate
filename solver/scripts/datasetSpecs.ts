/**
 * Shared design-space specifications and CSFM sweep for the dataset.
 *
 * Extracted from exportDataset.ts so that both the in-domain training-data
 * generator (exportDataset.ts) and the out-of-domain extrapolation generator
 * (exportExtrapolation.ts) draw on a single, identical definition of each
 * archetype's design space and the analysis sweep. This module has no
 * top-level side effects (it never runs a sweep on import).
 */

import { analyze } from '../src/core/analysis.ts';
import type { AnalysisOptions } from '../src/core/analysis.ts';
import type { ElementModel } from '../src/core/elements/types.ts';
import type { ConcreteMaterial, SteelMaterial } from '../src/core/materials.ts';

import { buildDeepBeam, defaultDeepBeamParams } from '../src/core/elements/deepBeamElement.ts';
import { buildHammerhead, defaultHammerheadParams } from '../src/core/elements/hammerhead.ts';
import {
  buildMultiColumnBent,
  defaultMultiColumnBentParams,
} from '../src/core/elements/multiColumnBent.ts';
import { buildPileCap, defaultPileCapParams } from '../src/core/elements/pileCap.ts';
import { buildPierCap, defaultPierCapParams } from '../src/core/elements/pierCap.ts';

/* ----------------------------------------------------------------------- */
/* Seeded RNG (mulberry32) + Latin-Hypercube sampler                         */
/* ----------------------------------------------------------------------- */

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Latin-Hypercube sample: n rows, d columns, each entry in [0,1). */
export function latinHypercube(n: number, d: number, rng: () => number): number[][] {
  const rows: number[][] = Array.from({ length: n }, () => new Array(d).fill(0));
  for (let j = 0; j < d; j++) {
    const perm = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) {
      const k = Math.floor(rng() * (i + 1));
      [perm[i], perm[k]] = [perm[k], perm[i]];
    }
    for (let i = 0; i < n; i++) rows[i][j] = (perm[i] + rng()) / n;
  }
  return rows;
}

/* ----------------------------------------------------------------------- */
/* Design space                                                             */
/*                                                                          */
/* Ranges below are the realized design space of the dataset.              */
/* ----------------------------------------------------------------------- */

/** Snap a continuous value to the nearest standard bar diameter (mm). */
const BAR_DIA = [20, 25, 28, 32, 36];
function snapDia(v: number): number {
  return BAR_DIA.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), BAR_DIA[0]);
}
export const lerp = (u: number, lo: number, hi: number) => lo + u * (hi - lo);
export const lerpInt = (u: number, lo: number, hi: number) => Math.round(lerp(u, lo, hi));

/** Build a concrete material from a sampled f'c (MPa). */
function concrete(fc: number): ConcreteMaterial {
  return { fc, density: 2400, lambda: 1.0 };
}
/** Build a steel material from a sampled f_y (MPa); f_t, eps_u approximated. */
function steel(fy: number): SteelMaterial {
  return { fy, ft: 1.2 * fy, Es: 200000, epsU: 0.08, grade: `fy${Math.round(fy)}` };
}

/**
 * A sampling spec: ordered dimension names, and a function that turns one
 * unit-hypercube row into (a) the design parameter record and (b) the
 * analysis-ready ElementModel.
 */
export interface ArchetypeSpec {
  archetype: string;
  dims: string[];
  build: (u: number[]) => { params: Record<string, number>; model: ElementModel };
}

const SHARED_DIMS = ['fck', 'fy'];

export const SPECS: ArchetypeSpec[] = [
  {
    archetype: 'deepBeam',
    dims: ['span', 'height', 'thickness', 'supportWidth', 'P', 'botDia', 'botCount',
      ...SHARED_DIMS],
    build: (u) => {
      const span = lerp(u[0], 2000, 6000);
      const height = lerp(u[1], 1400, 3600);
      const thickness = lerp(u[2], 300, 700);
      const supportWidth = lerp(u[3], 300, 600);
      const P = lerp(u[4], 1000e3, 6000e3);
      const botDia = snapDia(lerp(u[5], 20, 36));
      const botCount = lerpInt(u[6], 4, 10);
      const fck = lerp(u[7], 21, 69);
      const fy = lerp(u[8], 280, 690);
      const p = defaultDeepBeamParams();
      p.span = span;
      p.height = height;
      p.thickness = thickness;
      p.supportWidth = supportWidth;
      p.loads = [{ id: 'p1', x: 0, load: P }];
      p.bottomBars = { dia: botDia, count: botCount };
      return {
        params: { span, height, thickness, supportWidth, P, botDia, botCount, fck, fy },
        model: buildDeepBeam(p),
      };
    },
  },
  {
    archetype: 'hammerhead',
    dims: ['capLength', 'capDepthCenter', 'capDepthTip', 'capWidth', 'columnWidth',
      'bearingLoad', 'topDia', 'topCount', ...SHARED_DIMS],
    build: (u) => {
      const capLength = lerp(u[0], 6000, 12000);
      const capDepthCenter = lerp(u[1], 1600, 3000);
      const capDepthTip = lerp(u[2], 800, capDepthCenter - 400);
      const capWidth = lerp(u[3], 1200, 2400);
      const columnWidth = lerp(u[4], 1200, 2400);
      const bearingLoad = lerp(u[5], 600e3, 2000e3);
      const topDia = snapDia(lerp(u[6], 20, 36));
      const topCount = lerpInt(u[7], 8, 16);
      const fck = lerp(u[8], 21, 69);
      const fy = lerp(u[9], 280, 690);
      const p = defaultHammerheadParams();
      p.capLength = capLength;
      p.capDepthCenter = capDepthCenter;
      p.capDepthTip = capDepthTip;
      p.capWidth = capWidth;
      p.columnWidth = columnWidth;
      p.columnDepth = columnWidth;
      // bearings span a fixed fraction of the cap so the strut-and-tie
      // topology is invariant across designs (required by the fixed-size MLP)
      const nb = 6;
      const span = 0.84 * capLength;
      p.bearings = Array.from({ length: nb }, (_, i) => ({
        id: `b${i + 1}`,
        x: -span / 2 + (i * span) / (nb - 1),
        load: bearingLoad,
      }));
      p.topBars = { dia: topDia, count: topCount };
      return {
        params: { capLength, capDepthCenter, capDepthTip, capWidth, columnWidth,
          bearingLoad, topDia, topCount, fck, fy },
        model: buildHammerhead(p),
      };
    },
  },
  {
    archetype: 'multiColumnBent',
    dims: ['capDepth', 'capWidth', 'columnWidth', 'overhang', 'girderLoad',
      'topDia', 'topCount', ...SHARED_DIMS],
    build: (u) => {
      const capDepth = lerp(u[0], 1000, 2200);
      const capWidth = lerp(u[1], 900, 1600);
      const columnWidth = lerp(u[2], 700, 1300);
      const overhang = lerp(u[3], 800, 2400);
      const girderLoad = lerp(u[4], 400e3, 1400e3);
      const topDia = snapDia(lerp(u[5], 20, 36));
      const topCount = lerpInt(u[6], 6, 12);
      const fck = lerp(u[7], 21, 69);
      const fy = lerp(u[8], 280, 690);
      const p = defaultMultiColumnBentParams();
      p.capDepth = capDepth;
      p.capWidth = capWidth;
      p.columnWidth = columnWidth;
      p.columnDepth = columnWidth;
      p.overhang = overhang;
      // girders at fixed x positions (columns at -6000/0/6000) — never
      // coincident with a column — so the truss topology is invariant
      const gx = [-5400, -3240, -1080, 1080, 3240, 5400];
      p.girders = gx.map((x, i) => ({ id: `g${i + 1}`, x, load: girderLoad }));
      p.topBars = { dia: topDia, count: topCount };
      p.bottomBars = { dia: topDia, count: topCount };
      return {
        params: { capDepth, capWidth, columnWidth, overhang, girderLoad,
          topDia, topCount, fck, fy },
        model: buildMultiColumnBent(p),
      };
    },
  },
  {
    archetype: 'pileCap',
    dims: ['capLength', 'capWidth', 'capDepth', 'columnWidth', 'columnLoad',
      'botDia', 'botSpacing', ...SHARED_DIMS],
    build: (u) => {
      const capLength = lerp(u[0], 2400, 4200);
      const capWidth = lerp(u[1], 2400, 4200);
      const capDepth = lerp(u[2], 900, 1800);
      const columnWidth = lerp(u[3], 500, 900);
      const columnLoad = lerp(u[4], 3000e3, 10000e3);
      const botDia = snapDia(lerp(u[5], 20, 36));
      const botSpacing = lerp(u[6], 120, 250);
      const fck = lerp(u[7], 21, 69);
      const fy = lerp(u[8], 280, 690);
      const p = defaultPileCapParams();
      p.capLength = capLength;
      p.capWidth = capWidth;
      p.capDepth = capDepth;
      p.columnWidth = columnWidth;
      p.columnDepth = columnWidth;
      p.columnLoad = columnLoad;
      p.bottomX = { dia: botDia, spacing: botSpacing };
      p.bottomZ = { dia: botDia, spacing: botSpacing };
      return {
        params: { capLength, capWidth, capDepth, columnWidth, columnLoad,
          botDia, botSpacing, fck, fy },
        model: buildPileCap(p),
      };
    },
  },
  {
    // Stepped pier cap (CSFM book §6.5 geometry). Ranges bracket the
    // Geevar & Menon experimental specimens so the surrogate can be
    // validated directly against them.
    archetype: 'pierCap',
    dims: ['capWidth', 'stemWidth', 'capBandHeight', 'taperHeight',
      'stemHeight', 'thickness', 'loadPlate', 'columnLoad', 'supportPlate',
      'edgeDistance', 'mainDia', 'mainCount', ...SHARED_DIMS],
    build: (u) => {
      const capWidth = lerp(u[0], 900, 1600);
      const stemWidth = capWidth * lerp(u[1], 0.40, 0.65);
      const capBandHeight = lerp(u[2], 120, 240);
      const taperHeight = lerp(u[3], 240, 420);
      const stemHeight = lerp(u[4], 380, 620);
      const thickness = lerp(u[5], 300, 700);
      const loadPlate = lerp(u[6], 100, 220);
      const columnLoad = lerp(u[7], 1500e3, 5000e3);
      const supportPlate = lerp(u[8], 110, 200);
      const edgeDistance = lerp(u[9], 100, 200);
      const mainDia = lerp(u[10], 8, 16);
      const mainCount = lerpInt(u[11], 6, 28);
      const fck = lerp(u[12], 21, 69);
      const fy = lerp(u[13], 280, 690);
      const p = defaultPierCapParams();
      p.capWidth = capWidth;
      p.stemWidth = stemWidth;
      p.capBandHeight = capBandHeight;
      p.taperHeight = taperHeight;
      p.stemHeight = stemHeight;
      p.thickness = thickness;
      p.loadPlate = loadPlate;
      p.columnLoad = columnLoad;
      p.supports = 2;
      p.supportPlate = supportPlate;
      p.edgeDistance = edgeDistance;
      p.mainBars = { dia: mainDia, count: mainCount };
      return {
        params: { capWidth, stemWidth, capBandHeight, taperHeight, stemHeight,
          thickness, loadPlate, columnLoad, supportPlate, edgeDistance,
          mainDia, mainCount, fck, fy },
        model: buildPierCap(p),
      };
    },
  },
];

/* ----------------------------------------------------------------------- */
/* Sweep                                                                     */
/* ----------------------------------------------------------------------- */

/** A truss member enriched with the section properties the CSFM uses. */
export type EnrichedMember = ElementModel['truss']['members'][number] & {
  /** concrete cross-sectional area used by the CSFM strut law (mm^2) */
  concreteArea: number;
  /** reinforcement area used by the CSFM tie law (mm^2) */
  steelArea: number;
  /** representative tie bar diameter (mm) */
  barDia: number;
};

export interface DesignRecord {
  id: string;
  archetype: string;
  params: Record<string, number>;
  truss: {
    nodes: ElementModel['truss']['nodes'];
    members: EnrichedMember[];
    loads: ElementModel['truss']['loads'];
  };
  labels: {
    failureLoadFactor: number;
    failureMode: string;
    memberForces: Record<string, number>;
    curve: { loadFactor: number; displacement: number }[];
    /** CSFM member states sampled along the load path — supervision targets */
    csfmStates: {
      loadFactor: number;
      strains: Record<string, number>;
      forces: Record<string, number>;
    }[];
  };
}

/** Sub-sample the CSFM load steps to at most `n` evenly-spaced states. */
function sampleCsfmStates(
  steps: { loadFactor: number; members: { id: string; force: number; strain: number }[] }[],
  n = 12,
): DesignRecord['labels']['csfmStates'] {
  if (steps.length === 0) return [];
  const pick = steps.length <= n
    ? steps.map((_, i) => i)
    : Array.from({ length: n }, (_, i) => Math.round((i * (steps.length - 1)) / (n - 1)));
  return [...new Set(pick)].map((i) => {
    const st = steps[i];
    const strains: Record<string, number> = {};
    const forces: Record<string, number> = {};
    for (const m of st.members) { strains[m.id] = m.strain; forces[m.id] = m.force; }
    return { loadFactor: st.loadFactor, strains, forces };
  });
}

/**
 * Optional per-row transform applied to a unit-hypercube sample before it is
 * passed to `spec.build`. The identity transform reproduces the in-domain
 * training sweep exactly; the extrapolation generator supplies a transform
 * that maps each coordinate into a shell just outside [0,1].
 */
export type SampleTransform = (u: number[], dims: string[]) => number[];

const identityTransform: SampleTransform = (u) => u;

export interface SweepResult {
  designs: DesignRecord[];
  attempted: number;
  rejected: number;
}

/**
 * Sweep the CSFM solver over a Latin-Hypercube design of experiments for each
 * spec, returning the accepted designs. `transform` (default identity) lets a
 * caller remap the unit samples; with the identity transform the output is
 * byte-identical to the original exportDataset sweep.
 */
export function sweep(
  specs: ArchetypeSpec[],
  nPer: number,
  rng: () => number,
  transform: SampleTransform = identityTransform,
): SweepResult {
  const designs: DesignRecord[] = [];
  let attempted = 0;
  let rejected = 0;

  for (const spec of specs) {
    const samples = latinHypercube(nPer, spec.dims.length, rng);
    let kept = 0;
    for (let i = 0; i < samples.length; i++) {
      attempted++;
      let built;
      try {
        built = spec.build(transform(samples[i], spec.dims));
      } catch {
        rejected++;
        continue;
      }
      const opt: AnalysisOptions = {
        concrete: concrete(built.params.fck),
        steel: steel(built.params.fy),
        code: 'ACI318-19',
        exposure: 'exterior',
        runCsfm: true,
        runFem: false,
      };

      let result;
      try {
        result = analyze(built.model, opt);
      } catch {
        rejected++;
        continue;
      }
      // reject unstable / failed / non-converged CSFM analyses
      if (!result.truss.stable || result.errors.length > 0 || result.csfm === null) {
        rejected++;
        continue;
      }

      const memberForces: Record<string, number> = {};
      for (const m of result.truss.members) memberForces[m.id] = m.force;

      // Enrich each member with the section properties the CSFM assigns
      // (mirrors src/core/analysis.ts so the Python physics loss is faithful).
      const struts = built.model.struts;
      const strutMap = new Map(struts.map((s) => [s.memberId, s.width * s.thickness]));
      const tieMap = new Map(result.ties.map((t) => [t.memberId, t]));
      const repConcreteArea = struts.length
        ? Math.max(...struts.map((s) => s.width * s.thickness))
        : 1e6;
      const repSteelArea = result.ties.length
        ? Math.max(...result.ties.map((t) => t.asProvided))
        : 4000;
      const members: EnrichedMember[] = built.model.truss.members.map((m) => {
        const t = tieMap.get(m.id);
        return {
          ...m,
          concreteArea: strutMap.get(m.id) ?? repConcreteArea,
          steelArea: t ? t.asProvided : repSteelArea,
          barDia: t ? t.barDiameter : 25,
        };
      });

      designs.push({
        id: `${spec.archetype}-${String(kept).padStart(4, '0')}`,
        archetype: spec.archetype,
        params: built.params,
        truss: {
          nodes: built.model.truss.nodes,
          members,
          loads: built.model.truss.loads,
        },
        labels: {
          failureLoadFactor: result.csfm.failureLoadFactor,
          failureMode: result.csfm.failureMode,
          memberForces,
          curve: result.csfm.curve,
          csfmStates: sampleCsfmStates(result.csfm.steps),
        },
      });
      kept++;
    }
    console.log(`${spec.archetype}: kept ${kept}/${nPer}`);
  }

  return { designs, attempted, rejected };
}
