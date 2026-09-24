/**
 * Compatible Stress Field analysis — incremental, compatibility-based solution.
 *
 * This is the "compatible" extension of the static strut-and-tie model that
 * gives the CSFM its name (Kaufmann et al. 2020, Ch. 3): the classical lower-
 * bound truss is complemented with kinematic considerations. Here it is applied
 * at the member level — a nonlinear space-truss analysis in which:
 *
 *   - struts use the concrete law with compression softening k_c2(eps_1),
 *     where eps_1 is the transverse tensile strain inferred from the ties
 *     anchored at the strut ends (Eq. 3.1, Fig. 3.1e);
 *   - ties use the reinforcement law including tension stiffening (Eq. 3.3/3.6).
 *
 * The applied load is incremented; at every step the member secant stiffnesses
 * are iterated to equilibrium + compatibility, yielding a load-deformation
 * response and the failure load factor — the deformation-capacity information
 * that classical STM cannot provide.
 */

import type { ConcreteMaterial, SteelMaterial } from '../materials';
import { effectiveFcd, parabolaRectParams, steelStressBare } from './constitutive';
import { averageStrain } from './tensionStiffening';
import { solveTruss } from '../stm/truss';
import type { TrussLoad, TrussMember, TrussNode } from '../stm/truss';

export interface CsfmMemberState {
  id: string;
  kind: 'strut' | 'tie';
  force: number;          // N
  strain: number;         // average strain (mm/mm)
  utilization: number;    // |demand| / capacity at this load level
  /** for struts: k_c2 softening factor in effect */
  kc2?: number;
}

export interface CsfmStep {
  loadFactor: number;
  displacement: number;   // mm — magnitude of the largest nodal displacement
  members: CsfmMemberState[];
  converged: boolean;
}

export interface CsfmAnalysisInput {
  nodes: TrussNode[];
  members: TrussMember[];
  loads: TrussLoad[];     // reference load set (multiplied by load factor)
  concrete: ConcreteMaterial;
  steel: SteelMaterial;
  /** strut cross-sectional areas, keyed by member id (mm^2) */
  strutAreas: Record<string, number>;
  /** tie steel areas, keyed by member id (mm^2) */
  tieAreas: Record<string, number>;
  /** tie representative bar diameter, keyed by member id (mm) */
  tieBarDia: Record<string, number>;
  /** tie effective reinforcement ratio, keyed by member id */
  tieRhoEff: Record<string, number>;
}

export interface CsfmAnalysisResult {
  steps: CsfmStep[];
  failureLoadFactor: number;
  failureMode: string;
  curve: { loadFactor: number; displacement: number }[];
}

const MAX_LF = 3.0;        // search up to 3x the reference load
const LF_STEPS = 60;
const MAX_INNER = 25;

/**
 * Run the incremental compatible-stress-field analysis.
 */
export function runCsfmAnalysis(input: CsfmAnalysisInput): CsfmAnalysisResult {
  const { nodes, members, loads, concrete, steel } = input;
  const { epsCu2 } = parabolaRectParams(concrete.fc);
  const Ec0 = 4700 * Math.sqrt(concrete.fc);

  const steps: CsfmStep[] = [];
  let failureLoadFactor = MAX_LF;
  let failureMode = 'No failure detected within the analysed load range.';

  // member secant moduli, updated across load steps
  const Esec = new Map<string, number>();
  for (const m of members) {
    Esec.set(m.id, m.kind === 'strut' || m.kind === 'auto' ? Ec0 : steel.Es);
  }

  for (let s = 1; s <= LF_STEPS; s++) {
    const lf = (MAX_LF * s) / LF_STEPS;
    const scaledLoads = loads.map((l) => ({
      node: l.node,
      fx: l.fx * lf,
      fy: l.fy * lf,
      fz: l.fz * lf,
    }));

    let converged = false;
    let memberStates: CsfmMemberState[] = [];
    let maxDisp = 0;
    const prevForce = new Map<string, number>();
    const RELAX = 0.5; // under-relaxation of the secant update for stability

    for (let inner = 0; inner < MAX_INNER; inner++) {
      // build truss with current secant areas*moduli
      const trussMembers: TrussMember[] = members.map((m) => {
        // area follows the member's *current* role (compression vs tension)
        const f = prevForce.get(m.id);
        const inTension = f !== undefined ? f > 0 : m.kind === 'tie';
        const area = inTension
          ? input.tieAreas[m.id] ?? m.area
          : input.strutAreas[m.id] ?? m.area;
        return { ...m, area, E: Esec.get(m.id) ?? m.E };
      });

      let result;
      try {
        result = solveTruss(nodes, trussMembers, scaledLoads);
      } catch {
        failureLoadFactor = lf;
        failureMode = 'Truss became a mechanism (instability) at this load.';
        return finalize(steps, failureLoadFactor, failureMode);
      }

      // convergence is judged on member forces (robust to zero-force members
      // whose strut/tie role would otherwise flip on numerical noise)
      const maxF = Math.max(1, ...result.members.map((m) => Math.abs(m.force)));
      let change = 0;

      memberStates = result.members.map((mr) => {
        const isStrut = mr.force < 0;
        const dia = input.tieBarDia[mr.id] ?? 25;
        const rhoEff = input.tieRhoEff[mr.id] ?? 0.01;
        const old = Esec.get(mr.id) ?? Ec0;

        if (isStrut) {
          const area = input.strutAreas[mr.id] ?? 1e5;
          const stress = Math.abs(mr.force) / area; // compressive (MPa)
          const eps1 = transverseStrain(result.members, mr, input, steel);
          const { fcd, kc2 } = effectiveFcd(concrete, eps1, 1.0);
          const strain = -stress / Ec0;
          const target = Math.max(0.05 * Ec0, secantConcrete(stress, fcd, Ec0));
          Esec.set(mr.id, old + RELAX * (target - old));
          return {
            id: mr.id, kind: 'strut' as const, force: mr.force, strain,
            utilization: stress / Math.max(fcd, 1e-3), kc2,
          };
        }
        const area = input.tieAreas[mr.id] ?? 1e3;
        const sigmaSr = mr.force / area; // MPa, tension
        const { epsM } = averageStrain(concrete, steel, sigmaSr, dia, rhoEff);
        const target =
          epsM > 1e-9 ? Math.max(0.02 * steel.Es, sigmaSr / epsM) : steel.Es;
        Esec.set(mr.id, old + RELAX * (target - old));
        return {
          id: mr.id, kind: 'tie' as const, force: mr.force, strain: epsM,
          utilization: sigmaSr / steel.fy,
        };
      });

      result.members.forEach((mr) => {
        const prev = prevForce.get(mr.id);
        if (prev !== undefined) {
          change = Math.max(change, Math.abs(mr.force - prev) / maxF);
        }
        prevForce.set(mr.id, mr.force);
      });

      maxDisp = Math.max(
        ...Object.values(result.displacements).map((d) => Math.hypot(...d)),
      );

      if (inner > 0 && change < 1e-3) {
        converged = true;
        break;
      }
    }

    steps.push({ loadFactor: lf, displacement: maxDisp, members: memberStates, converged });

    // failure detection
    const crushed = memberStates.find(
      (m) => m.kind === 'strut' && m.utilization >= 1.0,
    );
    // A tie ruptures when the steel stress at a crack reaches the tensile
    // strength f_t (CSFM: sigma_sr = f_t). Its utilisation is sigma_sr / f_y.
    // With tension stiffening the average strain is then still below eps_u,
    // so the strain limit alone would let sigma_sr exceed f_t; it is kept
    // only as a secondary check.
    const ruptured = memberStates.find(
      (m) => m.kind === 'tie'
        && (m.utilization * steel.fy >= steel.ft || m.strain >= steel.epsU),
    );
    if (crushed) {
      failureLoadFactor = lf;
      failureMode = `Concrete crushing in strut ${crushed.id} (effective compressive strength reached).`;
      break;
    }
    if (ruptured) {
      failureLoadFactor = lf;
      failureMode = `Reinforcement rupture in tie ${ruptured.id} (σ_sr ≥ f_t).`;
      break;
    }
    if (!converged) {
      failureLoadFactor = lf;
      failureMode = 'Loss of convergence — softening branch / capacity reached.';
      break;
    }
  }

  return finalize(steps, failureLoadFactor, failureMode);
}

function finalize(
  steps: CsfmStep[],
  flf: number,
  mode: string,
): CsfmAnalysisResult {
  return {
    steps,
    failureLoadFactor: flf,
    failureMode: mode,
    curve: steps.map((s) => ({ loadFactor: s.loadFactor, displacement: s.displacement })),
  };
}

/** Secant modulus of the parabola-rectangle law at a given compressive stress. */
function secantConcrete(stress: number, fcd: number, Ec0: number): number {
  const r = Math.min(0.999, stress / Math.max(fcd, 1e-3));
  // strain from inverted parabola eps = epsC2*(1 - sqrt(1-r))
  const eps = 0.002 * (1 - Math.sqrt(1 - r));
  return eps > 1e-9 ? stress / eps : Ec0;
}

/**
 * Transverse tensile strain eps_1 acting on a strut — taken as the largest
 * average tie strain among ties anchored at the strut's two end nodes. This is
 * the mechanism by which compression softening enters the truss model.
 */
function transverseStrain(
  all: { id: string; ni: string; nj: string; force: number }[],
  strut: { id: string; ni: string; nj: string },
  input: CsfmAnalysisInput,
  steel: SteelMaterial,
): number {
  let eps1 = 0;
  for (const m of all) {
    if (m.force <= 0) continue; // ties only
    if (
      m.ni === strut.ni || m.nj === strut.ni ||
      m.ni === strut.nj || m.nj === strut.nj
    ) {
      const area = input.tieAreas[m.id] ?? 1e3;
      const sigma = m.force / area;
      eps1 = Math.max(eps1, sigma / steel.Es);
    }
  }
  return eps1;
}
