/**
 * CSFM crack-width calculation — Section 3.5 of Kaufmann et al. (2020).
 *
 * The crack width is obtained by integrating the reinforcement strains over the
 * crack spacing — Eq. (3.11):
 *
 *   w_b = integral_{s_r} (eps_m - eps_cm) dx = (eps_m - lambda*f_ctm/(2 Es)) * s_r
 *
 * and projected onto the crack direction — Eq. (3.10):
 *
 *   w = w_b / cos(theta_r + theta_b - pi/2)
 *
 * For crack-width verification the book uses lambda = 1.0 (maximum spacing) and
 * multiplies the strains computed with lambda = 0.67 by a factor 1.0/0.67 = 1.5.
 */

import type { ConcreteMaterial, SteelMaterial } from '../materials';
import { concreteFct } from '../materials';
import { averageStrain, crackSpacingMax, LAMBDA_AVG, LAMBDA_MAX } from './tensionStiffening';

const STRAIN_AMPLIFICATION = LAMBDA_MAX / LAMBDA_AVG; // = 1.5

/**
 * Crack width in the bar direction, w_b — Eq. (3.11).
 * @param sigmaSr steel stress at the crack (MPa)
 * @param phi     bar diameter (mm)
 * @param rhoEff  effective reinforcement ratio
 */
export function crackWidthBarDir(
  conc: ConcreteMaterial,
  steel: SteelMaterial,
  sigmaSr: number,
  phi: number,
  rhoEff: number,
): { wb: number; sr: number; epsM: number } {
  // strains computed with the average spacing, then amplified by 1.5
  const { epsM } = averageStrain(conc, steel, sigmaSr, phi, rhoEff, LAMBDA_AVG);
  const epsMamp = epsM * STRAIN_AMPLIFICATION;
  // crack spacing for crack-width check uses lambda = 1.0
  const sr = LAMBDA_MAX * crackSpacingMax(conc, phi, rhoEff);
  const fctm = concreteFct(conc);
  const epsCm = (LAMBDA_MAX * fctm) / (2 * steel.Es); // mean concrete strain
  const wb = Math.max(0, (epsMamp - epsCm) * sr);
  return { wb, sr, epsM: epsMamp };
}

/**
 * Crack width projected onto the principal crack direction — Eq. (3.10).
 * @param thetaR principal direction of the cracks (rad)
 * @param thetaB bar inclination (rad)
 */
export function projectedCrackWidth(
  wb: number,
  thetaR: number,
  thetaB: number,
): number {
  const c = Math.cos(thetaR + thetaB - Math.PI / 2);
  if (Math.abs(c) < 1e-6) return wb; // bar aligned with crack normal
  return wb / Math.abs(c);
}

/** Allowable crack width (mm) for an exposure class. */
export function allowableCrackWidth(exposure: ExposureClass): number {
  switch (exposure) {
    case 'interior':       return 0.40; // dry / protected
    case 'exterior':       return 0.30; // humidity, exterior exposure
    case 'aggressive':     return 0.20; // de-icing, marine, corrosive
    case 'water-retaining':return 0.15;
  }
}

export type ExposureClass = 'interior' | 'exterior' | 'aggressive' | 'water-retaining';

export const EXPOSURE_LABELS: Record<ExposureClass, string> = {
  interior: 'Interior / dry',
  exterior: 'Exterior exposure',
  aggressive: 'Aggressive / marine',
  'water-retaining': 'Water-retaining',
};
