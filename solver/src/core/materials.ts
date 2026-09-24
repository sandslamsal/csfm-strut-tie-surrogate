/**
 * Material models for concrete and reinforcing steel.
 *
 * All values in SI base units (MPa, N, mm). Helpers derive code-specific
 * quantities for ACI 318-19 and AASHTO LRFD, and the characteristic values
 * used by the CSFM constitutive models (Kaufmann et al. 2020, Ch. 3.3).
 */

export type DesignCode = 'ACI318-19' | 'AASHTO-LRFD';

export interface ConcreteMaterial {
  /** specified compressive strength f'c (MPa) — characteristic cylinder strength */
  fc: number;
  /** density (kg/m^3) — normal weight ~2400 */
  density: number;
  /** lightweight modification factor lambda (ACI 19.2.4) */
  lambda: number;
}

export interface SteelMaterial {
  /** specified yield strength fy (MPa) */
  fy: number;
  /** ultimate / tensile strength ft (MPa) */
  ft: number;
  /** modulus of elasticity Es (MPa) */
  Es: number;
  /** strain at rupture eps_u */
  epsU: number;
  /** ductility class label */
  grade: string;
}

/* ----------------------------------------------------------------------- */
/* Standard material library                                                */
/* ----------------------------------------------------------------------- */

export const CONCRETE_PRESETS: Record<string, ConcreteMaterial> = {
  "f'c 21 MPa (3 ksi)":         { fc: 20.7, density: 2400, lambda: 1.0 },
  "f'c 28 MPa (4 ksi)":         { fc: 27.6, density: 2400, lambda: 1.0 },
  "f'c 34 MPa (5 ksi)":         { fc: 34.5, density: 2400, lambda: 1.0 },
  "f'c 41 MPa (6 ksi)":         { fc: 41.4, density: 2400, lambda: 1.0 },
  "f'c 48 MPa (7 ksi)":         { fc: 48.3, density: 2400, lambda: 1.0 },
  "f'c 55 MPa (8 ksi)":         { fc: 55.2, density: 2400, lambda: 1.0 },
  "f'c 69 MPa (10 ksi) HSC":    { fc: 68.9, density: 2450, lambda: 1.0 },
  "f'c 83 MPa (12 ksi) HSC":    { fc: 82.7, density: 2450, lambda: 1.0 },
  "Lightweight 28 MPa":         { fc: 27.6, density: 1850, lambda: 0.75 },
  "Sand-lightweight 34 MPa":    { fc: 34.5, density: 1950, lambda: 0.85 },
};

export const STEEL_PRESETS: Record<string, SteelMaterial> = {
  'Grade 40 (280 MPa)': { fy: 280, ft: 420, Es: 200000, epsU: 0.12, grade: 'A615 Gr.40' },
  'Grade 60 (420 MPa)': { fy: 420, ft: 620, Es: 200000, epsU: 0.09, grade: 'A615 Gr.60' },
  'Grade 75 (520 MPa)': { fy: 517, ft: 690, Es: 200000, epsU: 0.07, grade: 'A615 Gr.75' },
  'Grade 80 (550 MPa)': { fy: 550, ft: 725, Es: 200000, epsU: 0.06, grade: 'A706 Gr.80' },
  'Grade 100 (690 MPa)':{ fy: 690, ft: 860, Es: 200000, epsU: 0.05, grade: 'A615 Gr.100' },
  'B500B (500 MPa)':    { fy: 500, ft: 540, Es: 200000, epsU: 0.05, grade: 'B500B' },
  'B500C (500 MPa)':    { fy: 500, ft: 575, Es: 200000, epsU: 0.075, grade: 'B500C' },
};

/* ----------------------------------------------------------------------- */
/* Derived concrete properties                                              */
/* ----------------------------------------------------------------------- */

/** Modulus of elasticity Ec (MPa). ACI 318-19 Eq. 19.2.2.1.b for normal range. */
export function concreteEc(c: ConcreteMaterial): number {
  // Ec = 4700 * lambda * sqrt(f'c)  (MPa) — SI form of 57000*sqrt(f'c [psi])
  return 4700 * c.lambda * Math.sqrt(c.fc);
}

/**
 * Concrete tensile strength f_ct (MPa) used by the CSFM tension-stiffening
 * models. The book neglects tensile strength for strength but uses it for
 * stiffness. fib MC2010 mean axial tensile strength is used.
 */
export function concreteFct(c: ConcreteMaterial): number {
  if (c.fc <= 50) return 0.3 * Math.pow(c.fc, 2 / 3);
  return 2.12 * Math.log(1 + (c.fc + 8) / 10);
}

/** Modulus of rupture fr (MPa) — ACI 318-19 Eq. 19.2.3.1. */
export function concreteFr(c: ConcreteMaterial): number {
  return 0.62 * c.lambda * Math.sqrt(c.fc);
}

/**
 * eta_fc brittleness factor — fib Model Code 2010, Eq. (3.1) of the CSFM book:
 *   eta_fc = (30 / f_ck)^(1/3) <= 1
 * Accounts for the increased brittleness of higher-strength concrete.
 */
export function etaFc(c: ConcreteMaterial): number {
  return Math.min(1, Math.cbrt(30 / c.fc));
}

/* ----------------------------------------------------------------------- */
/* Strength reduction / resistance factors                                  */
/* ----------------------------------------------------------------------- */

export interface SafetyFactors {
  /** phi for strut-and-tie components (ACI 318-19 21.2.1: 0.75) */
  phiSTM: number;
  /** phi for bearing (ACI 318-19 21.2.1: 0.65) */
  phiBearing: number;
  /** phi for shear */
  phiShear: number;
  /** label */
  label: string;
}

export function safetyFactors(code: DesignCode): SafetyFactors {
  if (code === 'AASHTO-LRFD') {
    // AASHTO LRFD 5.5.4.2: resistance factor for STM and compression in
    // anchorage zones; 0.70 for tension-controlled / strut-and-tie.
    return { phiSTM: 0.70, phiBearing: 0.70, phiShear: 0.90, label: 'AASHTO LRFD' };
  }
  // ACI 318-19 Table 21.2.1
  return { phiSTM: 0.75, phiBearing: 0.65, phiShear: 0.75, label: 'ACI 318-19' };
}

export function defaultConcrete(): ConcreteMaterial {
  return { ...CONCRETE_PRESETS["f'c 34 MPa (5 ksi)"] };
}

export function defaultSteel(): SteelMaterial {
  return { ...STEEL_PRESETS['Grade 60 (420 MPa)'] };
}
