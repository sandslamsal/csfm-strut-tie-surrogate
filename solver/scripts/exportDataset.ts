/**
 * Dataset-export script.
 *
 * Sweeps the validated CSFM solver (`analyze`) over a Latin-Hypercube design
 * of experiments across the D-region archetypes and writes a single JSON
 * dataset consumed by the Python + PyTorch training code.
 *
 * The design-space specifications and the sweep itself live in
 * scripts/datasetSpecs.ts, shared with the out-of-domain extrapolation
 * generator (scripts/exportExtrapolation.ts). This is the only TypeScript
 * stage of the pipeline; everything downstream (training, evaluation) is
 * Python (see ../pinn/).
 *
 * Run (Node 22+):
 *   npx tsx scripts/exportDataset.ts --n 200 --seed 1
 *
 * Options:
 *   --n     <int>   designs sampled PER archetype (default 200)
 *   --seed  <int>   RNG seed for reproducibility (default 1)
 *   --out   <path>  output JSON path (default ../data/dataset.json, run from solver/)
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import { SPECS, mulberry32, sweep } from './datasetSpecs.ts';

/* ----------------------------------------------------------------------- */
/* CLI                                                                      */
/* ----------------------------------------------------------------------- */

function arg(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}
const N_PER = parseInt(arg('--n', '200'), 10);
const SEED = parseInt(arg('--seed', '1'), 10);
const OUT = arg('--out', '../data/dataset.json');

/* ----------------------------------------------------------------------- */
/* Sweep + write                                                            */
/* ----------------------------------------------------------------------- */

const rng = mulberry32(SEED);
const { designs, attempted, rejected } = sweep(SPECS, N_PER, rng);

const dataset = {
  meta: {
    generatedAt: new Date().toISOString(),
    seed: SEED,
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
  `\nWrote ${designs.length} designs (${rejected} rejected of ${attempted}) -> ${OUT}`,
);
