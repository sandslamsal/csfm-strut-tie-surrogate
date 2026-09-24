/**
 * CSFM constitutive models — Kaufmann, Mata-Falcon, Weber, Galkovski (2020),
 * "Compatible Stress Field Design of Structural Concrete", Section 3.3.
 *
 * Concrete (3.3.1): uniaxial parabola-rectangle compression law per EN 1992-1-1,
 *   tensile strength neglected for strength. Effective compressive strength of
 *   cracked concrete obtained from the compression-softening factor k_c2 and the
 *   brittleness factor eta_fc — Eq. (3.1):
 *
 *      f_cd = k_c * f_ck / gamma_c = eta_fc * k_c2 * f_ck / gamma_c
 *
 * Reinforcement (3.3.2): idealized bilinear stress-strain law for the bare bar.
 *
 * All inputs/outputs SI: stress in MPa, strain dimensionless.
 */

import type { ConcreteMaterial, SteelMaterial } from '../materials';
import { etaFc } from '../materials';

/* ----------------------------------------------------------------------- */
/* Concrete in compression — parabola-rectangle (EN 1992-1-1 §3.1.7)         */
/* ----------------------------------------------------------------------- */

export interface ParabolaRectParams {
  epsC2: number;   // strain at peak
  epsCu2: number;  // ultimate strain
  n: number;       // exponent of the parabola
}

/** EN 1992-1-1 Table 3.1 parameters as a function of f_ck (MPa). */
export function parabolaRectParams(fck: number): ParabolaRectParams {
  if (fck <= 50) return { epsC2: 0.0020, epsCu2: 0.0035, n: 2.0 };
  const epsC2 = (2.0 + 0.085 * Math.pow(fck - 50, 0.53)) / 1000;
  const epsCu2 = (2.6 + 35 * Math.pow((90 - fck) / 100, 4)) / 1000;
  const n = 1.4 + 23.4 * Math.pow((90 - fck) / 100, 4);
  return { epsC2, epsCu2, n: Math.max(1.4, n) };
}

/**
 * Compression-softening factor k_c2 of cracked concrete as a function of the
 * principal (transverse) tensile strain eps1 — CSFM book Fig. 3.1e.
 *
 * The book uses a generalization of the fib MC2010 compression-field law,
 * removing the 0.65 cap used for shear verifications. This is a calibrated
 * implementation of that "Considered" curve: k_c2 = 1 up to first cracking,
 * then a hyperbolic decay milder than the MCFT (Vecchio & Collins 1986),
 * because the CSFM works with maximum stresses at the cracks.
 */
export function softeningKc2(eps1: number): number {
  if (eps1 <= 0) return 1.0;
  return Math.min(1.0, 1.0 / (0.8 + 140 * eps1));
}

/**
 * Effective (design) compressive strength of cracked concrete — Eq. (3.1).
 * @param fck   characteristic cylinder strength (MPa)
 * @param eps1  principal tensile strain (transverse) — drives softening
 * @param gammaC partial safety factor for concrete (1.0 for nominal/phi-format)
 */
export function effectiveFcd(
  conc: ConcreteMaterial,
  eps1: number,
  gammaC = 1.0,
): { kc2: number; etaFc: number; kc: number; fcd: number } {
  const kc2 = softeningKc2(eps1);
  const eta = etaFc(conc);
  const kc = eta * kc2;
  return { kc2, etaFc: eta, kc, fcd: (kc * conc.fc) / gammaC };
}

/** Concrete compressive stress (MPa, positive in compression) at strain eps>=0. */
export function concreteStress(
  conc: ConcreteMaterial,
  epsCompression: number,
  fcEffective?: number,
): number {
  const fc = fcEffective ?? conc.fc;
  const { epsC2, epsCu2, n } = parabolaRectParams(conc.fc);
  const e = Math.max(0, epsCompression);
  if (e >= epsCu2) return fc;          // plastic plateau (CSFM keeps it ~plastic)
  if (e >= epsC2) return fc;
  return fc * (1 - Math.pow(1 - e / epsC2, n));
}

/* ----------------------------------------------------------------------- */
/* Reinforcing steel — idealized bilinear law (EN 1992-1-1 / ACI)            */
/* ----------------------------------------------------------------------- */

/** Bare-bar stress (MPa) at average strain eps for the bilinear model. */
export function steelStressBare(steel: SteelMaterial, eps: number): number {
  const s = Math.abs(eps);
  const epsY = steel.fy / steel.Es;
  if (s <= epsY) return Math.sign(eps) * steel.Es * s;
  // hardening branch from (epsY, fy) to (epsU, ft)
  const Esh = (steel.ft - steel.fy) / (steel.epsU - epsY);
  const stress = steel.fy + Esh * (s - epsY);
  return Math.sign(eps) * Math.min(stress, steel.ft);
}

/** Steel hardening modulus E_sh = (ft - fy) / (eps_u - fy/Es). */
export function hardeningModulus(steel: SteelMaterial): number {
  const epsY = steel.fy / steel.Es;
  return (steel.ft - steel.fy) / (steel.epsU - epsY);
}

/** Yield strain. */
export function yieldStrain(steel: SteelMaterial): number {
  return steel.fy / steel.Es;
}
