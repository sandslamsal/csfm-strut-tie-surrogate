/**
 * Out-of-domain extrapolation-set generator.
 *
 * Re-uses the exact archetype design-space specs of scripts/datasetSpecs.ts,
 * but maps every Latin-Hypercube coordinate into a thin SHELL just outside the
 * training range [lo, hi] on each design parameter, so that every generated
 * design lies outside the box the surrogate was trained on. The CSFM solver
 * still labels each design, giving an honest out-of-domain test set on which
 * the trained surrogate's generalisation (and its domain-of-validity flag) can
 * be measured.
 *
 * The unit sample u in [0,1] is remapped per coordinate to a shell band:
 *   u in [0, 0.5)  ->  -delta * (1 - 2u)      (just below lo)
 *   u in [0.5, 1]  ->  1 + delta * (2u - 1)   (just above hi)
 * so lerp(u', lo, hi) lands in [lo - delta*(hi-lo), lo) u (hi, hi + delta*(hi-lo)].
 * Snapped/integer parameters (bar diameter, bar counts) saturate at their
 * physical limits, which is the intended behaviour.
 *
 * Run (Node 22+):
 *   npx tsx scripts/exportExtrapolation.ts --n 200 --seed 7 --delta 0.2
 *
 * Options:
 *   --n      <int>    designs sampled PER archetype (default 200)
 *   --seed   <int>    RNG seed (default 7; distinct from the training seed)
 *   --delta  <float>  shell width as a fraction of each range (default 0.2)
 *   --out    <path>   output JSON (default ../data/dataset_extrap.json, run from solver/)
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import { SPECS, mulberry32, sweep } from './datasetSpecs.ts';
import type { SampleTransform } from './datasetSpecs.ts';

/* ----------------------------------------------------------------------- */
/* CLI                                                                      */
/* ----------------------------------------------------------------------- */

function arg(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}
const N_PER = parseInt(arg('--n', '200'), 10);
const SEED = parseInt(arg('--seed', '7'), 10);
const DELTA = parseFloat(arg('--delta', '0.2'));
const OUT = arg('--out', '../data/dataset_extrap.json');

/* ----------------------------------------------------------------------- */
/* Shell transform                                                          */
/* ----------------------------------------------------------------------- */

const shell: SampleTransform = (u) =>
  u.map((x) => (x < 0.5 ? -DELTA * (1 - 2 * x) : 1 + DELTA * (2 * x - 1)));

/* ----------------------------------------------------------------------- */
/* Sweep + write                                                            */
/* ----------------------------------------------------------------------- */

const rng = mulberry32(SEED);
const { designs, attempted, rejected } = sweep(SPECS, N_PER, rng, shell);

const dataset = {
  meta: {
    generatedAt: new Date().toISOString(),
    kind: 'extrapolation',
    seed: SEED,
    delta: DELTA,
    perArchetype: N_PER,
    attempted,
    accepted: designs.length,
    rejected,
    archetypes: SPECS.map((s) => ({ archetype: s.archetype, dims: s.dims })),
  },
  designs,
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(dataset, null, 1));

console.log(
  `\nWrote ${designs.length} out-of-domain designs ` +
    `(${rejected} rejected of ${attempted}, delta=${DELTA}) -> ${OUT}`,
);
