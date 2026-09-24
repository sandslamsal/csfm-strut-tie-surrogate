/**
 * Verification of the structural element — Section 3.5 of Kaufmann et al. (2020).
 *
 * Combines the strut-and-tie solution with the CSFM constitutive models and the
 * design-code provisions to perform Ultimate Limit State (ULS) and Serviceability
 * Limit State (SLS) checks for every component (struts, ties, nodes).
 *
 * The partial-safety / phi format is used (book §3.5.1): factored demands are
 * compared against phi-reduced resistances.
 */

import type { ConcreteMaterial, DesignCode, SteelMaterial } from '../materials';
import { safetyFactors } from '../materials';
import {
  NodeType,
  StrutType,
  nodeStrength,
  strutStrength,
  tieStrength,
} from '../codes';
import type { MemberResult } from '../stm/truss';
import { crackWidthBarDir } from './crackWidth';
import type { ExposureClass } from './crackWidth';
import { allowableCrackWidth } from './crackWidth';

export type CheckStatus = 'ok' | 'warn' | 'fail';

export interface ComponentCheck {
  id: string;
  label: string;
  kind: 'strut' | 'tie' | 'node';
  demand: number;        // N (magnitude)
  capacity: number;      // N (phi-reduced)
  dcr: number;           // demand / capacity ratio
  status: CheckStatus;
  detail: string;        // explanation / code basis
}

export interface TieDesign {
  memberId: string;
  barDiameter: number;   // mm
  barCount: number;
  asProvided: number;    // mm^2
  asRequired: number;    // mm^2
  /** reinforcement group this tie belongs to (for optimization) */
  group?: string;
}

export interface StrutGeometry {
  memberId: string;
  width: number;         // mm — strut width
  thickness: number;     // mm — out-of-plane / member thickness
  type: StrutType;
}

export interface VerificationInput {
  code: DesignCode;
  concrete: ConcreteMaterial;
  steel: SteelMaterial;
  members: MemberResult[];
  ties: TieDesign[];
  struts: StrutGeometry[];
  nodes: { id: string; type: NodeType; area: number }[];
  exposure: ExposureClass;
}

export interface VerificationResult {
  checks: ComponentCheck[];
  governing: ComponentCheck | null;
  maxDcr: number;
  crackChecks: CrackCheck[];
  pass: boolean;
}

export interface CrackCheck {
  memberId: string;
  width: number;         // mm
  allowable: number;     // mm
  status: CheckStatus;
}

function statusFromDcr(dcr: number): CheckStatus {
  if (dcr <= 1.0) return 'ok';
  if (dcr <= 1.05) return 'warn';
  return 'fail';
}

/** Run the full ULS + SLS verification. */
export function verify(input: VerificationInput): VerificationResult {
  const phi = safetyFactors(input.code);
  const checks: ComponentCheck[] = [];

  const tieMap = new Map(input.ties.map((t) => [t.memberId, t]));
  const strutMap = new Map(input.struts.map((s) => [s.memberId, s]));

  for (const m of input.members) {
    const demand = Math.abs(m.force);
    if (m.kind === 'tie') {
      const tie = tieMap.get(m.id);
      if (!tie) continue;
      const Fnt = tieStrength(input.steel, tie.asProvided);
      const cap = phi.phiSTM * Fnt;
      const dcr = cap > 0 ? demand / cap : Infinity;
      checks.push({
        id: m.id,
        label: `Tie ${m.id}`,
        kind: 'tie',
        demand,
        capacity: cap,
        dcr,
        status: statusFromDcr(dcr),
        detail:
          `${tie.barCount}⌀${tie.barDiameter} (As = ${tie.asProvided.toFixed(0)} mm²); ` +
          `φFnt = ${phi.phiSTM}·As·fy`,
      });
    } else {
      const sg = strutMap.get(m.id);
      if (!sg) continue;
      // strut-to-tie angle approximated as 45° unless refined elsewhere
      const cap0 = strutStrength(input.code, input.concrete, sg.type);
      const area = sg.width * sg.thickness;
      const Fns = cap0.fce * area;
      const cap = phi.phiSTM * Fns;
      const dcr = cap > 0 ? demand / cap : Infinity;
      checks.push({
        id: m.id,
        label: `Strut ${m.id}`,
        kind: 'strut',
        demand,
        capacity: cap,
        dcr,
        status: statusFromDcr(dcr),
        detail: `${cap0.basis}; A_cs = ${area.toFixed(0)} mm²`,
      });
    }
  }

  // nodal zone checks
  for (const nd of input.nodes) {
    // governing demand at a node = largest member force framing in
    const framing = input.members.filter((m) => m.ni === nd.id || m.nj === nd.id);
    if (framing.length === 0) continue;
    const demand = Math.max(...framing.map((m) => Math.abs(m.force)));
    const ns = nodeStrength(input.code, input.concrete, nd.type);
    const Fnn = ns.fce * nd.area;
    const cap = phi.phiBearing * Fnn;
    const dcr = cap > 0 ? demand / cap : Infinity;
    checks.push({
      id: `N-${nd.id}`,
      label: `Node ${nd.id}`,
      kind: 'node',
      demand,
      capacity: cap,
      dcr,
      status: statusFromDcr(dcr),
      detail: `${ns.basis}; A_nz = ${nd.area.toFixed(0)} mm²`,
    });
  }

  // SLS crack-width checks for ties (service force ~ demand / 1.5 average factor)
  const allowable = allowableCrackWidth(input.exposure);
  const crackChecks: CrackCheck[] = [];
  for (const m of input.members) {
    if (m.kind !== 'tie') continue;
    const tie = tieMap.get(m.id);
    if (!tie || tie.asProvided <= 0) continue;
    // service-level steel stress at the crack: factored force / ~1.45 load factor
    const serviceForce = Math.abs(m.force) / 1.45;
    const sigmaSr = serviceForce / tie.asProvided;
    // effective reinforcement ratio: bar area over its tributary concrete area
    const rhoEff = estimateRhoEff(tie);
    const { wb } = crackWidthBarDir(
      input.concrete,
      input.steel,
      sigmaSr,
      tie.barDiameter,
      rhoEff,
    );
    crackChecks.push({
      memberId: m.id,
      width: wb,
      allowable,
      status: wb <= allowable ? 'ok' : wb <= 1.1 * allowable ? 'warn' : 'fail',
    });
  }

  const maxDcr = checks.reduce((a, c) => Math.max(a, c.dcr), 0);
  const governing =
    checks.length > 0
      ? checks.reduce((a, c) => (c.dcr > a.dcr ? c : a))
      : null;
  const pass =
    checks.every((c) => c.status !== 'fail') &&
    crackChecks.every((c) => c.status !== 'fail');

  return { checks, governing, maxDcr, crackChecks, pass };
}

/** Effective reinforcement ratio of a tie's tension chord (Eq. 3.4 area basis). */
function estimateRhoEff(tie: TieDesign): number {
  // tributary concrete = a square of side phi_c,eff per bar (Fig. 3.3)
  const barArea = Math.PI * (tie.barDiameter / 2) ** 2;
  const cEff = 4 * tie.barDiameter; // ~ phi_c,eff for typical fc/fct
  const concPerBar = cEff * cEff;
  return Math.min(0.08, barArea / concPerBar);
}
