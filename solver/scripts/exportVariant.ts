/**
 * Experiment E1: a CHANGED LOADING CONFIGURATION.
 *
 * The training deep beam carries one midspan point load. This variant carries
 * two point loads at +-span/4 (four-point loading), which changes the strut
 * and tie topology (four panel points instead of three: 8 nodes, 13 members
 * against 6 nodes, 9 members). Everything else -- the parameter ranges, the
 * sampler, the solver and its settings -- is identical to datasetSpecs.ts, so
 * the cost of the new configuration is measured on nothing but the
 * configuration change.
 *
 * Run (Node 22+):
 *   npx tsx scripts/exportVariant.ts --n 750 --seed 1 --out <path>
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import { buildDeepBeam, defaultDeepBeamParams } from '../src/core/elements/deepBeamElement.ts';
import { lerp, lerpInt, mulberry32, sweep } from './datasetSpecs.ts';
import type { ArchetypeSpec } from './datasetSpecs.ts';

const BAR_DIA = [20, 25, 28, 32, 36];
const snapDia = (v: number) =>
  BAR_DIA.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), BAR_DIA[0]);

const SPEC2P: ArchetypeSpec = {
  archetype: 'deepBeam2P',
  dims: ['span', 'height', 'thickness', 'supportWidth', 'P', 'botDia', 'botCount', 'fck', 'fy'],
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
    // two equal loads at the quarter points: the SAME total load as the
    // training configuration, delivered as two point loads
    p.loads = [
      { id: 'p1', x: -span / 4, load: P / 2 },
      { id: 'p2', x: +span / 4, load: P / 2 },
    ];
    p.bottomBars = { dia: botDia, count: botCount };
    return {
      params: { span, height, thickness, supportWidth, P, botDia, botCount, fck, fy },
      model: buildDeepBeam(p),
    };
  },
};

function arg(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}
const N_PER = parseInt(arg('--n', '750'), 10);
const SEED = parseInt(arg('--seed', '1'), 10);
const OUT = arg('--out', '../data/dataset_deepBeam2P.json');

const t0 = Date.now();
const rng = mulberry32(SEED);
const { designs, attempted, rejected } = sweep([SPEC2P], N_PER, rng);
const elapsed = (Date.now() - t0) / 1000;
const dataset = {
  meta: {
    generatedAt: new Date().toISOString(), seed: SEED, perArchetype: N_PER,
    attempted, accepted: designs.length, rejected, sweepSeconds: elapsed,
    archetypes: [{ archetype: SPEC2P.archetype, dims: SPEC2P.dims }],
  },
  designs,
};
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(dataset, null, 1));
const d0 = designs[0];
console.log(`nodes ${d0.truss.nodes.length}, members ${d0.truss.members.length}`);
console.log(`Wrote ${designs.length} designs (${rejected} rejected of ${attempted}) in ${elapsed.toFixed(1)} s -> ${OUT}`);
