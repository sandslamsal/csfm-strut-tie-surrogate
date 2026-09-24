/**
 * CSFM tension stiffening — Section 3.3.4 of Kaufmann et al. (2020).
 *
 * Tension stiffening is introduced by modifying the bare-bar stress-strain law
 * so that the average strain eps_m of the embedded bar is captured. Two regimes:
 *
 *  - Stabilized cracking  : Tension Chord Model (TCM), Marti et al. 1998 — Eq.(3.3)
 *  - Non-stabilized crack : Pull-Out Model (POM)                         — Eq.(3.6)
 *
 * The maximum theoretical crack spacing s_r0 follows from Eq. (3.2); the actual
 * spacing is s_r = lambda * s_r0 with lambda = 0.67 (average behaviour) or
 * lambda = 1.0 (conservative, used for crack-width verification).
 */

import type { ConcreteMaterial, SteelMaterial } from '../materials';
import { concreteEc, concreteFct } from '../materials';
import { hardeningModulus, yieldStrain } from './constitutive';

export const LAMBDA_AVG = 0.67; // average crack spacing factor
export const LAMBDA_MAX = 1.0;  // conservative crack spacing for crack widths

/** Bond shear stresses of the stepped rigid-plastic TCM bond law. */
export function bondStresses(conc: ConcreteMaterial): { tb0: number; tb1: number } {
  const fctm = concreteFct(conc);
  return { tb0: 2 * fctm, tb1: fctm }; // tau_b0 (sigma_s<=fy), tau_b1 (sigma_s>fy)
}

/**
 * Minimum reinforcement ratio for stabilized cracking — Eq. (3.5):
 *   rho_cr = f_ct / (f_y - (n-1) f_ct),   n = Es/Ec
 * Below rho_cr cracking is non-stabilized (use POM).
 */
export function rhoCr(conc: ConcreteMaterial, steel: SteelMaterial): number {
  const fct = concreteFct(conc);
  const n = steel.Es / concreteEc(conc);
  const denom = steel.fy - (n - 1) * fct;
  return denom > 0 ? fct / denom : 1;
}

/**
 * Maximum theoretical crack spacing of the tension chord — Eq. (3.2):
 *   s_r0 = phi * f_ct * (1 - rho_eff) / (2 * tau_b0 * rho_eff)
 * @param phi    bar diameter (mm)
 * @param rhoEff effective reinforcement ratio of the tension chord
 */
export function crackSpacingMax(
  conc: ConcreteMaterial,
  phi: number,
  rhoEff: number,
): number {
  const fct = concreteFct(conc);
  const { tb0 } = bondStresses(conc);
  const r = Math.max(rhoEff, 1e-4);
  return (phi * fct * (1 - r)) / (2 * tb0 * r);
}

/**
 * Diameter of the maximum concrete area that one bar can activate — Eq. (3.4):
 *   phi_c,eff = phi * sqrt(f_t / f_ct)
 */
export function effectiveConcreteDiameter(
  conc: ConcreteMaterial,
  steel: SteelMaterial,
  phi: number,
): number {
  return phi * Math.sqrt(steel.ft / concreteFct(conc));
}

/**
 * Average strain eps_m of an embedded bar for STABILIZED cracking — Eq. (3.3),
 * Tension Chord Model. sigmaSr is the steel stress at the crack (MPa).
 */
export function epsMStabilized(
  conc: ConcreteMaterial,
  steel: SteelMaterial,
  sigmaSr: number,
  phi: number,
  sr: number,
): number {
  const { tb0, tb1 } = bondStresses(conc);
  const Es = steel.Es;
  const fy = steel.fy;
  const ft = steel.ft;
  const Esh = hardeningModulus(steel);
  const epsY = yieldStrain(steel);

  if (sigmaSr <= fy) {
    // elastic chord
    return Math.max(0, sigmaSr / Es - (tb0 * sr) / (Es * phi));
  }
  const sigmaLim2 = fy + (2 * tb1 * sr) / phi;
  if (sigmaSr <= sigmaLim2) {
    const term1 =
      ((sigmaSr - fy) ** 2 * phi) /
      (4 * Esh * tb1 * sr) *
      (1 - (Esh * tb0) / (Es * tb1));
    const term2 = ((sigmaSr - fy) / Es) * (tb0 / tb1);
    const term3 = epsY - (tb0 * sr) / (Es * phi);
    return Math.max(0, term1 + term2 + term3);
  }
  // fully yielded chord
  return Math.max(
    0,
    fy / Es + (sigmaSr - fy) / Esh - (tb1 * sr) / (Esh * phi),
  );
}

/**
 * Average strain eps_m for NON-STABILIZED cracking — Eq. (3.6), Pull-Out Model.
 * Used for reinforcement ratios below rho_cr (e.g. light stirrups).
 */
export function epsMNonStabilized(
  conc: ConcreteMaterial,
  steel: SteelMaterial,
  sigmaSr: number,
): number {
  const { tb0, tb1 } = bondStresses(conc);
  const Es = steel.Es;
  const fy = steel.fy;
  const ft = steel.ft;
  const Esh = hardeningModulus(steel);
  const ratio = tb1 / tb0;

  if (sigmaSr <= fy) {
    const denom = 2 * Es * (ft + fy * (ratio - 1));
    return Math.max(0, (sigmaSr ** 2 * ratio) / denom);
  }
  const num =
    (fy / Es) * (sigmaSr + fy * (ratio / 2 - 1)) +
    (sigmaSr - fy) ** 2 / (2 * Esh);
  const denom = ft + fy * (ratio - 1);
  return Math.max(0, num / denom);
}

/**
 * Convenience: pick the appropriate tension-stiffening model and return eps_m.
 */
export function averageStrain(
  conc: ConcreteMaterial,
  steel: SteelMaterial,
  sigmaSr: number,
  phi: number,
  rhoEff: number,
  lambda = LAMBDA_AVG,
): { epsM: number; sr: number; stabilized: boolean } {
  const stabilized = rhoEff >= rhoCr(conc, steel);
  const sr = lambda * crackSpacingMax(conc, phi, rhoEff);
  const epsM = stabilized
    ? epsMStabilized(conc, steel, sigmaSr, phi, sr)
    : epsMNonStabilized(conc, steel, sigmaSr);
  return { epsM, sr, stabilized };
}
