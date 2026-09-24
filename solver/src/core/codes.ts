/**
 * Design-code strut-and-tie provisions.
 *
 *  - ACI 318-19, Chapter 23 (Strut-and-Tie Method)
 *  - AASHTO LRFD Bridge Design Specifications, Article 5.8.2
 *
 * Both yield an effective concrete strength f_ce for struts and nodal zones and
 * a tie strength from the reinforcement. Strength reduction (phi) factors are
 * applied separately by the verification layer.
 */

import type { ConcreteMaterial, DesignCode, SteelMaterial } from './materials';

/* ----------------------------------------------------------------------- */
/* Strut / node classification                                              */
/* ----------------------------------------------------------------------- */

export type StrutType =
  | 'prismatic'         // uniform cross-section, uncracked compression
  | 'bottle-reinforced' // bottle-shaped with crack-control reinforcement
  | 'bottle-plain'      // bottle-shaped without reinforcement
  | 'tension-member';   // strut located in a tension member

export type NodeType =
  | 'CCC'  // bounded by struts and bearing areas only
  | 'CCT'  // anchoring one tie
  | 'CTT'; // anchoring two or more ties

export const STRUT_LABELS: Record<StrutType, string> = {
  prismatic: 'Prismatic strut (βs = 1.0)',
  'bottle-reinforced': 'Bottle-shaped, reinforced (βs = 0.75)',
  'bottle-plain': 'Bottle-shaped, plain (βs = 0.40)',
  'tension-member': 'Strut in tension member (βs = 0.40)',
};

export const NODE_LABELS: Record<NodeType, string> = {
  CCC: 'CCC node — struts/bearing only (βn = 1.0)',
  CCT: 'CCT node — anchoring one tie (βn = 0.80)',
  CTT: 'CTT node — anchoring ≥2 ties (βn = 0.60)',
};

/* ----------------------------------------------------------------------- */
/* ACI 318-19                                                                */
/* ----------------------------------------------------------------------- */

/** ACI 318-19 Table 23.4.3 — strut coefficient βs. */
function aciBetaS(t: StrutType): number {
  switch (t) {
    case 'prismatic': return 1.0;
    case 'bottle-reinforced': return 0.75;
    case 'bottle-plain': return 0.40;
    case 'tension-member': return 0.40;
  }
}

/** ACI 318-19 Table 23.9.2 — nodal-zone coefficient βn. */
function aciBetaN(n: NodeType): number {
  switch (n) {
    case 'CCC': return 1.0;
    case 'CCT': return 0.80;
    case 'CTT': return 0.60;
  }
}

/* ----------------------------------------------------------------------- */
/* AASHTO LRFD 5.8.2                                                          */
/* ----------------------------------------------------------------------- */

/**
 * AASHTO LRFD 5.8.2.5.3a — limiting compressive stress in a strut, including
 * the strain-softening effect (MCFT):
 *   f_cu = f'c / (0.8 + 170 * eps_1)  <=  0.85 f'c
 *   eps_1 = eps_s + (eps_s + 0.002) * cot^2(alpha_s)
 * where eps_s is the tensile strain in the tie crossing the strut and alpha_s
 * is the smallest angle between strut and adjoining tie.
 */
export function aashtoEps1(epsTie: number, alphaStrutTieRad: number): number {
  const cot = 1 / Math.tan(Math.max(alphaStrutTieRad, 0.05));
  return epsTie + (epsTie + 0.002) * cot * cot;
}

export function aashtoFcu(fc: number, eps1: number): number {
  return Math.min(0.85 * fc, fc / (0.8 + 170 * Math.max(eps1, 0)));
}

/* ----------------------------------------------------------------------- */
/* Unified API                                                               */
/* ----------------------------------------------------------------------- */

export interface StrutCapacity {
  /** effective concrete strength (MPa) */
  fce: number;
  /** efficiency factor relative to f'c */
  efficiency: number;
  /** human-readable basis */
  basis: string;
}

/**
 * Effective compressive strength of a strut.
 * @param strutAngle smallest strut-to-tie angle (rad) — AASHTO only
 * @param epsTie     tie strain crossing the strut — AASHTO only
 */
export function strutStrength(
  code: DesignCode,
  conc: ConcreteMaterial,
  type: StrutType,
  strutAngle = Math.PI / 4,
  epsTie = 0.002,
): StrutCapacity {
  if (code === 'ACI318-19') {
    const betaS = aciBetaS(type);
    const fce = 0.85 * betaS * conc.fc;
    return {
      fce,
      efficiency: 0.85 * betaS,
      basis: `ACI 318-19 §23.4.3: f_ce = 0.85·βs·f'c, βs = ${betaS.toFixed(2)}`,
    };
  }
  // AASHTO LRFD
  const eps1 = aashtoEps1(epsTie, strutAngle);
  const fce = aashtoFcu(conc.fc, eps1);
  return {
    fce,
    efficiency: fce / conc.fc,
    basis: `AASHTO LRFD §5.8.2.5.3: f_cu = f'c/(0.8+170·ε₁), ε₁ = ${eps1.toFixed(4)}`,
  };
}

/** Effective compressive strength of a nodal zone. */
export function nodeStrength(
  code: DesignCode,
  conc: ConcreteMaterial,
  node: NodeType,
): StrutCapacity {
  if (code === 'ACI318-19') {
    const betaN = aciBetaN(node);
    const fce = 0.85 * betaN * conc.fc;
    return {
      fce,
      efficiency: 0.85 * betaN,
      basis: `ACI 318-19 §23.9.2: f_ce = 0.85·βn·f'c, βn = ${betaN.toFixed(2)}`,
    };
  }
  // AASHTO LRFD 5.8.2.5.3b — nodal zone limits (m factor)
  const m = node === 'CCC' ? 0.85 : node === 'CCT' ? 0.75 : 0.65;
  return {
    fce: m * conc.fc,
    efficiency: m,
    basis: `AASHTO LRFD §5.8.2.5.3: f_cu = m·f'c, m = ${m.toFixed(2)} (${node})`,
  };
}

/** Tie nominal strength F_nt = A_st · f_y. */
export function tieStrength(steel: SteelMaterial, asProvided: number): number {
  return asProvided * steel.fy;
}
