/**
 * Single-design CSFM verification for the design-optimisation demo
 * (pinn/design_optimization.py).
 *
 * Reads a JSON request {archetype, us: number[][]} from stdin, where each `u`
 * is a unit-hypercube design vector in the SAME ordering as the archetype's
 * `dims` in datasetSpecs.ts. Each design is rebuilt with the exact
 * training-time `SPECS[].build(u)` and run through the reference CSFM solver
 * `analyze()`. Prints a JSON array of
 *   { params, stable, failureLoadFactor, failureMode, memberForces }
 * to stdout. This lets the Python optimiser verify a handful of surrogate-
 * chosen designs against the ground-truth solver without re-implementing it.
 *
 * Run:  echo '{"archetype":"deepBeam","us":[[...]]}' | npx tsx scripts/verifyDesign.ts
 */
import { SPECS } from './datasetSpecs.ts';
import { analyze } from '../src/core/analysis.ts';
import type { AnalysisOptions } from '../src/core/analysis.ts';
import type { ConcreteMaterial, SteelMaterial } from '../src/core/materials.ts';

// Mirror the (module-private) material builders in datasetSpecs.ts exactly.
function concrete(fc: number): ConcreteMaterial {
  return { fc, density: 2400, lambda: 1.0 };
}
function steel(fy: number): SteelMaterial {
  return { fy, ft: 1.2 * fy, Es: 200000, epsU: 0.08, grade: `fy${Math.round(fy)}` };
}

function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let buf = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (d) => (buf += d));
    process.stdin.on('end', () => resolve(buf));
  });
}

const main = async () => {
  const req = JSON.parse(await readStdin()) as { archetype: string; us: number[][] };
  const spec = SPECS.find((s) => s.archetype === req.archetype);
  if (!spec) {
    console.error(`unknown archetype: ${req.archetype}`);
    process.exit(1);
  }

  const out = req.us.map((u) => {
    let built;
    try {
      built = spec.build(u);
    } catch (e) {
      return { params: null, stable: false, failureLoadFactor: null, failureMode: `build error: ${e}`, memberForces: {} };
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
    } catch (e) {
      return { params: built.params, stable: false, failureLoadFactor: null, failureMode: `analyze error: ${e}`, memberForces: {} };
    }
    const stable = result.truss.stable && result.errors.length === 0 && result.csfm !== null;
    const memberForces: Record<string, number> = {};
    for (const m of result.truss.members) memberForces[m.id] = m.force;
    // CSFM failure-state member forces (last load step) -- the surrogate's
    // force-head target, used for the force-state comparison in the demo.
    const failureForces: Record<string, number> = {};
    if (result.csfm && result.csfm.steps.length) {
      const last = result.csfm.steps[result.csfm.steps.length - 1];
      for (const m of last.members) failureForces[m.id] = m.force;
    }
    // Applied-load scale F0 = ||P_ref|| (matches data.py ref_load norm), used
    // to denormalise the surrogate's member-force head.
    let f0sq = 0;
    for (const ld of built.model.truss.loads) {
      f0sq += (ld.fx ?? 0) ** 2 + (ld.fy ?? 0) ** 2 + (ld.fz ?? 0) ** 2;
    }
    return {
      params: built.params,
      stable,
      failureLoadFactor: result.csfm ? result.csfm.failureLoadFactor : null,
      failureMode: result.csfm ? result.csfm.failureMode : null,
      f0: Math.max(Math.sqrt(f0sq), 1.0),
      memberForces,
      failureForces,
    };
  });

  console.log(JSON.stringify(out));
};

main();
