/**
 * Builds a 2D plane-stress problem (an elevation of the element) for the
 * continuum CSFM solver. Reinforcement is smeared into top / bottom bands and
 * a distributed stirrup ratio, following the cracked-membrane idealisation.
 */

import type { ContinuumProblem } from '../csfm/continuum';

export interface ElevationSpec {
  xMin: number;
  xMax: number;
  topY: (x: number) => number;     // top surface elevation (mm)
  botY: (x: number) => number;     // soffit elevation (mm)
  thickness: number;               // out-of-plane width (mm)
  res: number;                     // target cells along x
  /** downward loads applied on the top surface */
  loads: { x: number; P: number; width: number }[];
  /** supports at the soffit (vertical reaction; one may also fix X) */
  supports: { x: number; width: number; fixX: boolean }[];
  rhoTopBand: number;              // smeared ratio of the top reinforcement
  rhoBotBand: number;              // smeared ratio of the bottom reinforcement
  rhoStirrup: number;              // smeared vertical (stirrup) ratio
  bandFrac: number;                // band thickness as a fraction of the depth
  /** representative reinforcing-bar diameter (mm) — for crack spacing */
  barDia: number;
  /** optional cut-out / opening — true where there is NO concrete */
  hole?: (x: number, y: number) => boolean;
}

export function buildElevationContinuum(spec: ElevationSpec): ContinuumProblem {
  const span = spec.xMax - spec.xMin;
  const nx = Math.max(16, Math.min(44, Math.round(spec.res)));
  const dx = span / nx;

  // vertical extent
  let yMax = 0;
  for (let i = 0; i <= nx; i++) {
    const x = spec.xMin + i * dx;
    yMax = Math.max(yMax, spec.topY(x));
  }
  const ny = Math.max(8, Math.min(34, Math.round(yMax / dx)));
  const dy = yMax / ny;

  const filled = (i: number, j: number): boolean => {
    const x = spec.xMin + (i + 0.5) * dx;
    const y = (j + 0.5) * dy;
    if (y < spec.botY(x) - dy * 0.5 || y > spec.topY(x) + dy * 0.5) return false;
    if (spec.hole && spec.hole(x, y)) return false;
    return true;
  };

  const rhoX = (i: number, j: number): number => {
    const x = spec.xMin + (i + 0.5) * dx;
    const y = (j + 0.5) * dy;
    const b = spec.botY(x), t = spec.topY(x);
    const band = Math.max(dy, spec.bandFrac * (t - b));
    if (y >= t - band) return spec.rhoTopBand;
    if (y <= b + band) return spec.rhoBotBand;
    return 0.0012;                  // nominal skin reinforcement
  };
  const rhoY = (): number => spec.rhoStirrup;

  // distribute each load over its bearing width as point loads on the top
  const loads: ContinuumProblem['loads'] = [];
  for (const ld of spec.loads) {
    const n = Math.max(1, Math.round(ld.width / dx));
    const per = ld.P / n;
    for (let k = 0; k < n; k++) {
      const x = ld.x - ld.width / 2 + (k + 0.5) * (ld.width / n);
      loads.push({ pos: [x, spec.topY(x)], f: [0, -per] });
    }
  }

  const supports: ContinuumProblem['supports'] = spec.supports.map((s) => ({
    pos: [s.x, spec.botY(s.x)],
    radius: Math.max(dx, s.width / 2) * 1.1,
    fix: [s.fixX, true] as [boolean, boolean],
  }));

  return {
    nx, ny, dx, dy,
    origin: [spec.xMin, 0],
    thickness: spec.thickness,
    filled,
    rhoX,
    rhoY,
    loads,
    supports,
    barDia: spec.barDia,
  };
}
