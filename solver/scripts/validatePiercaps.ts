/**
 * Solver check — repo CSFM solvers vs. the Geevar & Menon (2018)
 * concrete pier-cap experiments reported in Kaufmann et al. (2020) §6.5.
 *
 * Builds each specimen S1-S5 with the dedicated `pierCap` element and the mean
 * measured material properties, then runs BOTH repo solvers:
 *   - the truss strut-and-tie CSFM (analysis.ts / nonlinearCsfm.ts)
 *   - the 2D continuum stress-field CSFM (runContinuumCsfm)
 * and compares against the measured test load and the book's CSFM (M0).
 *
 * The book's M0 is itself a continuum analysis, so the continuum column is the
 * apples-to-apples comparison.
 *
 * Both solvers run at gamma = 1, so feeding mean material properties suffices.
 *
 * Run:  npx tsx scripts/validatePiercaps.ts
 */

import { readFileSync } from 'node:fs';

import { analyze } from '../src/core/analysis.ts';
import type { AnalysisOptions } from '../src/core/analysis.ts';
import { buildPierCap } from '../src/core/elements/pierCap.ts';
import type { PierCapParams } from '../src/core/elements/pierCap.ts';
import { runContinuumCsfm } from '../src/core/csfm/continuum.ts';
import type { ConcreteMaterial, SteelMaterial } from '../src/core/materials.ts';

const DATA = '../validation/piercaps_geevar_menon_2018.json';

interface BarSpec { count: number; barDia_mm: number }
interface Specimen {
  id: string;
  loadPlate_lb_mm: number;
  reinforcement: {
    primary_As1: BarSpec;
    additional_As2: BarSpec | null;
  };
  materials: {
    mainReinf: { fy_MPa: number; ft_MPa: number; epsU_permille: number };
    concrete: { fc_MPa: number; epsC0_permille: number };
  };
  measured: { Pu_perSupport_kN: number; Pu_total_kN: number; failureMode: string };
  bookCsfmPrediction_perSupport_kN: { M0: number };
}

const dataset = JSON.parse(readFileSync(DATA, 'utf8'));
const geom = dataset.geometry.reportedFigureDimensions_mm as Record<string, number>;
const bands = geom.heightBands as unknown as number[];
const specimens = dataset.specimens as Specimen[];

function stats(rs: number[]): { mean: number; cov: number } {
  const mean = rs.reduce((a, b) => a + b, 0) / rs.length;
  const cov = Math.sqrt(
    rs.reduce((a, b) => a + (b - mean) ** 2, 0) / rs.length) / mean;
  return { mean, cov };
}
const pad = (v: string | number, w: number) => String(v).padEnd(w);
const W = [6, 9, 9, 9, 9, 9, 9, 9];

console.log('Solver check — pier caps (Geevar & Menon 2018)');
console.log('Dedicated pierCap element; loads in kN (total = 4x per-support).\n');
console.log(
  ['spec', 'measured', 'truss', 'continuum', 'book-M0',
    'exp/trus', 'exp/cont', 'exp/book'].map((s, i) => pad(s, W[i])).join(''),
);

const rTruss: number[] = [];
const rCont: number[] = [];
for (const s of specimens) {
  const concrete: ConcreteMaterial = {
    fc: s.materials.concrete.fc_MPa, density: 2400, lambda: 1.0,
  };
  const steel: SteelMaterial = {
    fy: s.materials.mainReinf.fy_MPa,
    ft: s.materials.mainReinf.ft_MPa,
    Es: 200000,
    epsU: s.materials.mainReinf.epsU_permille / 1000,
    grade: 'mean',
  };
  const as2 = s.reinforcement.additional_As2;
  const expTotal = s.measured.Pu_total_kN;

  const params: PierCapParams = {
    capWidth: geom.topCapWidth,
    stemWidth: geom.lowerStemWidth,
    capBandHeight: bands[0],
    taperHeight: bands[1],
    stemHeight: bands[2],
    thickness: geom.outOfPlaneThickness_b,
    loadPlate: s.loadPlate_lb_mm,
    columnLoad: expTotal * 1e3,            // reference = measured total load
    supports: 2,
    supportPlate: 150,
    edgeDistance: geom.dimension_a,
    mainBars: {
      dia: s.reinforcement.primary_As1.barDia_mm,
      count: s.reinforcement.primary_As1.count,
    },
    extraBars: as2
      ? { dia: as2.barDia_mm, count: as2.count }
      : { dia: 8, count: 0 },
    skinBars: { dia: 8, spacing: 150 },
    vertBars: { dia: 8, spacing: 150 },
  };
  const model = buildPierCap(params);
  const opt: AnalysisOptions = {
    concrete, steel, code: 'ACI318-19', exposure: 'exterior',
    runCsfm: true, runFem: false,
  };

  // truss strut-and-tie CSFM
  let trussPred = NaN;
  try {
    const r = analyze(model, opt);
    if (r.csfm) trussPred = r.csfm.failureLoadFactor * expTotal;
  } catch { /* leave NaN */ }

  // 2D continuum stress-field CSFM (apples-to-apples with the book M0)
  let contPred = NaN;
  try {
    const c = runContinuumCsfm(model.continuum, concrete, steel);
    contPred = c.failureLoadFactor * expTotal;
  } catch { /* leave NaN */ }

  const bookTotal = s.bookCsfmPrediction_perSupport_kN.M0 * 4;
  const eT = expTotal / trussPred;
  const eC = expTotal / contPred;
  if (isFinite(eT)) rTruss.push(eT);
  if (isFinite(eC)) rCont.push(eC);

  console.log([
    pad(s.id, W[0]),
    pad(expTotal.toFixed(0), W[1]),
    pad(isFinite(trussPred) ? trussPred.toFixed(0) : '—', W[2]),
    pad(isFinite(contPred) ? contPred.toFixed(0) : '—', W[3]),
    pad(bookTotal.toFixed(0), W[4]),
    pad(isFinite(eT) ? eT.toFixed(2) : '—', W[5]),
    pad(isFinite(eC) ? eC.toFixed(2) : '—', W[6]),
    pad((expTotal / bookTotal).toFixed(2), W[7]),
  ].join(''));
}

console.log();
if (rTruss.length) {
  const t = stats(rTruss);
  console.log(`truss STM   exp/calc:  mean ${t.mean.toFixed(2)}  CoV ${t.cov.toFixed(2)}`);
}
if (rCont.length) {
  const c = stats(rCont);
  console.log(`continuum   exp/calc:  mean ${c.mean.toFixed(2)}  CoV ${c.cov.toFixed(2)}`);
}
console.log('book CSFM M0 exp/calc:  mean 0.96  CoV 0.13  (Kaufmann et al. 2020, Table 6.17)');
