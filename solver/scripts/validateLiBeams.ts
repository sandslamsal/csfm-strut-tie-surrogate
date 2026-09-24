/**
 * A second experimental series for the reference solver: the eight
 * full-scale deep beams of Li, Wu, Zhang and
 * Xie, Materials 15 (2022) 6017 (open access). 2200 x 200 x 900 mm,
 * l0/h = 2, four-point loading, varying the shear-span ratio a/h0
 * (LDB1-3), the longitudinal reinforcement ratio (LDB4, 2, 5) and the
 * stirrup ratio (LDB6, 2, 7, 8). LDB1-7 failed by strut crushing, LDB8 (no
 * stirrups) by diagonal splitting. Mean measured materials, gamma = 1.
 *
 * Run:  npx tsx scripts/validateLiBeams.ts
 */
import { analyze } from '../src/core/analysis.ts';
import type { AnalysisOptions } from '../src/core/analysis.ts';
import { buildDeepBeam } from '../src/core/elements/deepBeamElement.ts';
import type { DeepBeamParams } from '../src/core/elements/deepBeamElement.ts';
import type { ConcreteMaterial, SteelMaterial } from '../src/core/materials.ts';

// geometry common to all eight specimens (paper, Table 3 and Figure 4)
const L_TOT = 2200, H = 900, B = 200;
const L0 = 2 * H;                    // effective span 1800 mm
const OVERHANG = (L_TOT - L0) / 2;   // 200 mm
const H0 = 807;                      // effective depth
const PLATE = 80;                    // 80 x 200 mm bearing plates
// materials (paper, Tables 1 and 2): prism strength; HTRB600 20 mm bars
const concrete: ConcreteMaterial = { fc: 42.9, density: 2400, lambda: 1.0 };
const steel: SteelMaterial = { fy: 653.7, ft: 823.3, Es: 196600, epsU: 0.075, grade: 'HTRB600-mean' };
// HRB400E 8 mm web bars at fy 456.8: the element carries one steel grade, so
// the stirrup diameter is scaled to preserve the yield FORCE at the tie grade
const D_STIR_EFF = 8 * Math.sqrt(456.8 / 653.7);
const A_STIR = 2 * (Math.PI / 4) * 8 ** 2;             // two legs, mm^2

interface Spec { id: string; lam: number; rho_s: number; rho_sv: number; Pu: number; mode: string }
const SPECS: Spec[] = [
  { id: 'LDB1', lam: 0.3, rho_s: 0.0105, rho_sv: 0.0033, Pu: 1169.0, mode: 'strut crushing' },
  { id: 'LDB2', lam: 0.6, rho_s: 0.0105, rho_sv: 0.0033, Pu: 1000.0, mode: 'strut crushing' },
  { id: 'LDB3', lam: 0.9, rho_s: 0.0105, rho_sv: 0.0033, Pu: 943.0, mode: 'strut crushing' },
  { id: 'LDB4', lam: 0.6, rho_s: 0.0067, rho_sv: 0.0033, Pu: 823.0, mode: 'strut crushing' },
  { id: 'LDB5', lam: 0.6, rho_s: 0.0127, rho_sv: 0.0033, Pu: 1193.5, mode: 'strut crushing' },
  { id: 'LDB6', lam: 0.6, rho_s: 0.0105, rho_sv: 0.0050, Pu: 1024.0, mode: 'strut crushing' },
  { id: 'LDB7', lam: 0.6, rho_s: 0.0105, rho_sv: 0.0025, Pu: 950.0, mode: 'strut crushing' },
  { id: 'LDB8', lam: 0.6, rho_s: 0.0105, rho_sv: 0.0, Pu: 940.0, mode: 'diagonal splitting' },
];

const pad = (v: string | number, w: number) => String(v).padEnd(w);
console.log('Reference solver vs Li et al. (2022) deep beams');
console.log(`geometry ${L_TOT}x${B}x${H} mm, l0 = ${L0} mm, fc = ${concrete.fc} MPa, tie fy = ${steel.fy} MPa\n`);
console.log(['spec', 'a/h0', 'rho_s', 'rho_sv', 'P_exp', 'P_calc', 'exp/calc', 'mode (solver)', 'mode (test)']
  .map((s, i) => pad(s, [6, 6, 7, 7, 8, 8, 9, 22, 20][i])).join(''));

const ratios: number[] = [];
const rows: Record<string, unknown>[] = [];
for (const s of SPECS) {
  // shear span a is measured from the support EDGE to the load (paper, note to Table 3)
  const a = s.lam * H0;
  const xLoad = L0 / 2 - (PLATE / 2 + a);
  const As = s.rho_s * B * H0;                            // mm^2
  const count = As / ((Math.PI / 4) * 20 ** 2);           // equivalent 20 mm bars
  const spacing = s.rho_sv > 0 ? A_STIR / (s.rho_sv * B) : 1e9;
  const params: DeepBeamParams = {
    span: L0, overhang: OVERHANG, height: H, thickness: B, supportWidth: PLATE,
    loads: [
      { id: 'p1', x: -xLoad, load: s.Pu * 1e3 / 2 },
      { id: 'p2', x: +xLoad, load: s.Pu * 1e3 / 2 },
    ],
    topBars: { dia: 12, count: 2 },
    bottomBars: { dia: 20, count },
    stirrup: { dia: D_STIR_EFF, spacing, legs: 2 },
  };
  const model = buildDeepBeam(params);
  const opt: AnalysisOptions = {
    concrete, steel, code: 'ACI318-19', exposure: 'exterior', runCsfm: true, runFem: false,
  };
  const r = analyze(model, opt);
  const lamF = r.csfm ? r.csfm.failureLoadFactor : NaN;
  const Pcalc = lamF * s.Pu;                               // reference load = measured total
  const ratio = s.Pu / Pcalc;
  if (isFinite(ratio)) ratios.push(ratio);
  const mode = r.csfm ? r.csfm.failureMode : 'n/a';
  // sensitivity: the web-width rule, strut width min(b, 0.3 h) regardless of
  // the bearing, which ignores the nodal zone at the 80 mm plates
  const modelWeb = buildDeepBeam(params);
  for (const st of modelWeb.struts) st.width = Math.min(B, 0.3 * H);
  const rWeb = analyze(modelWeb, opt);
  const Pweb = rWeb.csfm ? rWeb.csfm.failureLoadFactor * s.Pu : NaN;
  // measured bearing stress at the ultimate load: each plate carries P/2
  const bearing = (s.Pu * 1e3 / 2) / (PLATE * B);
  const strut = model.struts.find((st) => st.memberId === 'D0')!;
  rows.push({ id: s.id, lam: s.lam, rho_s: s.rho_s, rho_sv: s.rho_sv, Pu: s.Pu, Pcalc, ratio, mode, testMode: s.mode,
    nodes: model.truss.nodes.length, members: model.truss.members.length,
    strutWidth: strut.width, PcalcWebWidth: Pweb, ratioWebWidth: s.Pu / Pweb,
    bearingStressAtPu_MPa: bearing, bearingStressOverFc: bearing / concrete.fc });
  console.log([s.id, s.lam.toFixed(1), (100 * s.rho_s).toFixed(2), (100 * s.rho_sv).toFixed(2),
    s.Pu.toFixed(0), Pcalc.toFixed(0), ratio.toFixed(2), mode, s.mode]
    .map((v, i) => pad(v, [6, 6, 7, 7, 8, 8, 9, 22, 20][i])).join(''));
}
const mean = ratios.reduce((x, y) => x + y, 0) / ratios.length;
const cov = Math.sqrt(ratios.reduce((x, y) => x + (y - mean) ** 2, 0) / ratios.length) / mean;
console.log(`\nexp/calc  mean ${mean.toFixed(2)}  CoV ${cov.toFixed(2)}  (n = ${ratios.length})`);
// sensitivity to the longitudinal reinforcement (LDB4 -> LDB2 -> LDB5, same a/h0 and web)
const byId = Object.fromEntries(rows.map((r) => [r.id as string, r]));
const s4 = byId.LDB4, s5 = byId.LDB5;
console.log(`capacity rise LDB4 -> LDB5: measured ${((s5.Pu as number) / (s4.Pu as number) - 1) * 100 | 0}%, `
  + `solver ${(((s5.Pcalc as number) / (s4.Pcalc as number) - 1) * 100).toFixed(0)}%`);
const webRatios = rows.map((r) => r.ratioWebWidth as number);
const webMean = webRatios.reduce((x, y) => x + y, 0) / webRatios.length;
const bear = rows.map((r) => r.bearingStressOverFc as number);
console.log(`\nstrut width in the element: ${rows.map((r) => (r.strutWidth as number).toFixed(0)).join(', ')} mm`);
console.log(`web-width rule (min(b, 0.3h), bearing ignored): exp/calc mean ${webMean.toFixed(2)}`);
console.log(`measured bearing stress at P_u: ${Math.min(...bear).toFixed(2)}-${Math.max(...bear).toFixed(2)} f_c, mean ${(bear.reduce((x, y) => x + y, 0) / bear.length).toFixed(2)} f_c`);
console.log(`eta_fc (30/fc)^(1/3) = ${Math.pow(30 / concrete.fc, 1 / 3).toFixed(3)}; eta_fc * fc = ${(Math.pow(30 / concrete.fc, 1 / 3) * concrete.fc).toFixed(1)} MPa`);
console.log(JSON.stringify({ mean, cov, rows }, null, 1));
