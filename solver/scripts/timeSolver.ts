/**
 * Reference-solver per-design timing.
 *
 * Measures the wall-clock time of a single CSFM analysis on this machine, so
 * the surrogate speed-up is quoted against a like-for-like, same-machine solver
 * time rather than a literature figure. Builds a batch of valid deep-beam
 * designs (the archetype used for the surrogate timing) and times analyze()
 * per design, after warm-up.
 *
 * Run:  npx tsx scripts/timeSolver.ts --n 60
 */

import { performance } from 'node:perf_hooks';
import { analyze } from '../src/core/analysis.ts';
import type { AnalysisOptions } from '../src/core/analysis.ts';
import { SPECS, mulberry32, latinHypercube } from './datasetSpecs.ts';

function arg(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}
const N = parseInt(arg('--n', '60'), 10);

console.log(`reference CSFM solver timing, n~=${N} designs/archetype (single core)`);
console.log(
  `${'archetype'.padEnd(16)} ${'mean'.padStart(8)} ${'median'.padStart(8)} ` +
    `${'min'.padStart(7)} ${'max'.padStart(7)}  (ms/design)`,
);

for (const spec of SPECS) {
  const rng = mulberry32(123);
  const rows = latinHypercube(N + 20, spec.dims.length, rng);

  const jobs: { model: ReturnType<typeof spec.build>['model']; opt: AnalysisOptions }[] = [];
  for (const u of rows) {
    let built;
    try {
      built = spec.build(u);
    } catch {
      continue;
    }
    const opt: AnalysisOptions = {
      concrete: { fc: built.params.fck, density: 2400, lambda: 1.0 },
      steel: { fy: built.params.fy, ft: 1.2 * built.params.fy, Es: 200000,
        epsU: 0.08, grade: `fy${Math.round(built.params.fy)}` },
      code: 'ACI318-19',
      exposure: 'exterior',
      runCsfm: true,
      runFem: false,
    };
    try {
      const r = analyze(built.model, opt); // warm-up / validity filter
      if (r.truss.stable && r.errors.length === 0 && r.csfm !== null) {
        jobs.push({ model: built.model, opt });
      }
    } catch {
      /* skip */
    }
    if (jobs.length >= N) break;
  }

  for (let i = 0; i < Math.min(5, jobs.length); i++) analyze(jobs[i].model, jobs[i].opt);

  const times: number[] = [];
  for (const j of jobs) {
    const t0 = performance.now();
    analyze(j.model, j.opt);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const median = times[Math.floor(times.length / 2)];
  console.log(
    `${spec.archetype.padEnd(16)} ${mean.toFixed(2).padStart(8)} ` +
      `${median.toFixed(2).padStart(8)} ${times[0].toFixed(2).padStart(7)} ` +
      `${times[times.length - 1].toFixed(2).padStart(7)}`,
  );
}
