/**
 * Analysis pipeline — orchestrates the full CSFM workflow for an element:
 *
 *   1. Solve the 3D strut-and-tie model (lower-bound static solution).
 *   2. Auto-design the tie reinforcement (book §3.4 — reinforcement design).
 *   3. ULS + SLS verification against the design code (book §3.5).
 *   4. Incremental Compatible-Stress-Field analysis (book Ch. 3) — load-
 *      deformation response and failure load factor.
 *   5. Linear-elastic FE force-path field (book §3.4.2).
 *
 * Steps 4 and 5 are optional (toggled by the UI) as they are heavier.
 */

import type { ConcreteMaterial, DesignCode, SteelMaterial } from './materials';
import { safetyFactors } from './materials';
import { solveTruss } from './stm/truss';
import type { TrussResult } from './stm/truss';
import { verify } from './csfm/verification';
import type { VerificationResult, TieDesign } from './csfm/verification';
import type { ExposureClass } from './csfm/crackWidth';
import { runCsfmAnalysis } from './csfm/nonlinearCsfm';
import type { CsfmAnalysisResult } from './csfm/nonlinearCsfm';
import { solveFem } from './fem/fem3d';
import type { FemResult } from './fem/fem3d';
import { concreteEc } from './materials';
import type { ElementModel } from './elements/types';

export interface AnalysisOptions {
  concrete: ConcreteMaterial;
  steel: SteelMaterial;
  code: DesignCode;
  exposure: ExposureClass;
  runCsfm: boolean;
  runFem: boolean;
}

/** Reinforcement optimization result — CSFM book §3.4.3, Eq. (3.7)-(3.8). */
export interface OptiMember {
  id: string;
  kind: 'tie' | 'strut';
  asReq: number;       // required steel area (mm^2)
  asProv: number;      // provided steel area (mm^2)
  length: number;      // member length (mm)
  utilization: number; // asReq / asProv
}
export interface OptimizationResult {
  members: OptiMember[];
  volumeProvided: number;   // mm^3 of steel
  volumeRequired: number;   // mm^3 of steel
  /** potential saving as a fraction of the provided volume */
  saving: number;
  /** total provided / required steel mass (kg, density 7850 kg/m^3) */
  massProvided: number;
  massRequired: number;
}

export interface AnalysisResult {
  truss: TrussResult;
  ties: TieDesign[];
  verification: VerificationResult;
  optimization: OptimizationResult;
  csfm: CsfmAnalysisResult | null;
  fem: FemResult | null;
  errors: string[];
  warnings: string[];
  /** informational notes — no action required */
  info: string[];
}

/** Run the complete analysis for an element model. */
export function analyze(model: ElementModel, opt: AnalysisOptions): AnalysisResult {
  const errors: string[] = [];
  const warnings: string[] = [...model.notes];
  const info: string[] = [...model.info];
  const phi = safetyFactors(opt.code);

  // ---- 1. strut-and-tie solution ----------------------------------------
  let truss: TrussResult;
  try {
    truss = solveTruss(model.truss.nodes, model.truss.members, model.truss.loads);
  } catch (e) {
    errors.push(e instanceof Error ? e.message : String(e));
    return {
      truss: emptyTruss(),
      ties: model.ties,
      verification: emptyVerification(),
      optimization: emptyOptimization(),
      csfm: null,
      fem: null,
      errors,
      warnings,
      info,
    };
  }

  // ---- 2. tie reinforcement: keep the user-provided bars, compute the ---
  //         statically required area (book §3.4 — reinforcement design)
  const forceById = new Map(truss.members.map((m) => [m.id, m.force]));
  const designedTies: TieDesign[] = model.ties.map((t) => {
    const F = Math.abs(forceById.get(t.memberId) ?? 0);
    const asReq = F / (phi.phiSTM * opt.steel.fy);
    return { ...t, asRequired: asReq };
  });

  // ---- reinforcement optimization (Eq. 3.7: min sum l_si * A_si) --------
  const lengthById = new Map(truss.members.map((m) => [m.id, m.length]));
  const optiMembers: OptiMember[] = designedTies.map((t) => {
    const len = lengthById.get(t.memberId) ?? 0;
    return {
      id: t.memberId,
      kind: 'tie' as const,
      asReq: t.asRequired,
      asProv: t.asProvided,
      length: len,
      utilization: t.asProvided > 0 ? t.asRequired / t.asProvided : Infinity,
    };
  });
  const volProv = optiMembers.reduce((s, m) => s + m.asProv * m.length, 0);
  const volReq = optiMembers.reduce((s, m) => s + m.asReq * m.length, 0);
  const optimization: OptimizationResult = {
    members: optiMembers,
    volumeProvided: volProv,
    volumeRequired: volReq,
    saving: volProv > 0 ? Math.max(0, 1 - volReq / volProv) : 0,
    massProvided: (volProv * 7850) / 1e9,
    massRequired: (volReq * 7850) / 1e9,
  };

  // ---- 3. verification --------------------------------------------------
  const verification = verify({
    code: opt.code,
    concrete: opt.concrete,
    steel: opt.steel,
    members: truss.members,
    ties: designedTies,
    struts: model.struts,
    nodes: model.nodes,
    exposure: opt.exposure,
  });

  // ---- 4. compatible stress field analysis ------------------------------
  let csfm: CsfmAnalysisResult | null = null;
  if (opt.runCsfm) {
    try {
      // Every truss member can carry compression (concrete) OR tension (steel)
      // — its role may flip from the static STM classification. Both a
      // concrete cross-section and a steel area are therefore assigned to
      // every member so the compatible-stress-field iteration is well posed.
      const strutMap = new Map(model.struts.map((s) => [s.memberId, s.width * s.thickness]));
      const tieMap = new Map(designedTies.map((t) => [t.memberId, t]));
      const repConcreteArea = model.struts.length
        ? Math.max(...model.struts.map((s) => s.width * s.thickness))
        : 1e6;
      const repSteelArea = designedTies.length
        ? Math.max(...designedTies.map((t) => t.asProvided))
        : 4000;

      const strutAreas: Record<string, number> = {};
      const tieAreas: Record<string, number> = {};
      const tieBarDia: Record<string, number> = {};
      const tieRhoEff: Record<string, number> = {};
      model.truss.members.forEach((m) => {
        strutAreas[m.id] = strutMap.get(m.id) ?? repConcreteArea;
        const t = tieMap.get(m.id);
        tieAreas[m.id] = t ? t.asProvided : repSteelArea;
        tieBarDia[m.id] = t ? t.barDiameter : 25;
        const dia = tieBarDia[m.id];
        const barA = (Math.PI / 4) * dia ** 2;
        // effective reinforcement ratio per bar over its tributary concrete
        // area (~(4·phi)^2 per Fig. 3.3 of the CSFM book)
        tieRhoEff[m.id] = Math.min(0.06, barA / (16 * dia * dia));
      });
      csfm = runCsfmAnalysis({
        nodes: model.truss.nodes,
        members: model.truss.members,
        loads: model.truss.loads,
        concrete: opt.concrete,
        steel: opt.steel,
        strutAreas,
        tieAreas,
        tieBarDia,
        tieRhoEff,
      });
    } catch (e) {
      warnings.push(
        'CSFM incremental analysis could not complete: ' +
          (e instanceof Error ? e.message : String(e)),
      );
    }
  }

  // ---- 5. linear FE force-path field ------------------------------------
  let fem: FemResult | null = null;
  if (opt.runFem) {
    try {
      fem = solveFem(
        model.fem.domain,
        model.fem.loads,
        model.fem.supports,
        concreteEc(opt.concrete),
        0.2,
      );
      if (!fem.converged) {
        warnings.push('FE force-path solver did not fully converge (CG iteration limit).');
      }
    } catch (e) {
      warnings.push(
        'FE force-path analysis failed: ' +
          (e instanceof Error ? e.message : String(e)),
      );
    }
  }

  return { truss, ties: designedTies, verification, optimization, csfm, fem, errors, warnings, info };
}

function emptyOptimization(): OptimizationResult {
  return {
    members: [], volumeProvided: 0, volumeRequired: 0, saving: 0,
    massProvided: 0, massRequired: 0,
  };
}

function emptyTruss(): TrussResult {
  return {
    members: [],
    reactions: {},
    displacements: {},
    maxStrut: 0,
    maxTie: 0,
    stable: false,
    message: 'Not solved.',
  };
}

function emptyVerification(): VerificationResult {
  return { checks: [], governing: null, maxDcr: 0, crackChecks: [], pass: false };
}
