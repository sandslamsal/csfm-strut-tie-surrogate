/**
 * Common types for the 3D D-region element modules.
 *
 * Each element module (pile cap, hammerhead bent, multi-column bent) turns a
 * parametric definition into:
 *   - a 3D strut-and-tie model (space truss),
 *   - a voxelised continuum domain for the linear-elastic FE force-path tool,
 *   - render geometry for the 3D viewport,
 *   - design data (ties / struts / nodes) for the verification layer.
 *
 * Coordinate system: X = width, Y = up (height), Z = depth. Units: mm, N.
 */

import type { TrussLoad, TrussMember, TrussNode } from '../stm/truss';
import type { VoxelDomain, FemLoad, FemSupport } from '../fem/fem3d';
import type { TieDesign, StrutGeometry } from '../csfm/verification';
import type { ContinuumProblem } from '../csfm/continuum';
import type { NodeType } from '../codes';

export type ElementType =
  | 'pile-cap'
  | 'pier-cap'
  | 'hammerhead'
  | 'multi-column-bent'
  | 'deep-beam'
  | 'corbel'
  | 'dapped-beam'
  | 'beam-opening'
  | 'cantilever-pier';

export const ELEMENT_LABELS: Record<ElementType, string> = {
  'pile-cap': '3D Pile Cap',
  'pier-cap': 'Pier Cap',
  hammerhead: 'Hammerhead Bent Cap',
  'multi-column-bent': 'Multi-Column Bent Cap',
  'deep-beam': 'Deep Beam',
  corbel: 'Corbel / Bracket',
  'dapped-beam': 'Dapped-End Beam',
  'beam-opening': 'Beam with Web Opening',
  'cantilever-pier': 'Cantilever Wall Pier',
};

/** A render primitive for the Three.js viewport. */
export type SolidRole = 'concrete' | 'column' | 'pile' | 'bearing' | 'load';

export type RenderSolid =
  | {
      kind: 'box';
      center: [number, number, number];
      size: [number, number, number];
      color: string;
      opacity?: number;
      role: SolidRole;
    }
  | {
      kind: 'cylinder';
      base: [number, number, number];
      height: number;
      radius: number;
      axis: 'x' | 'y' | 'z';
      color: string;
      opacity?: number;
      role: SolidRole;
    }
  | {
      /** polygon profile in the X-Y (elevation) plane, extruded along Z */
      kind: 'prism';
      profile: [number, number][];
      zCenter: number;
      zDepth: number;
      color: string;
      opacity?: number;
      role: SolidRole;
    };

/** A reinforcing bar (or bar group) polyline for the 3D viewport. */
export interface RenderRebar {
  id: string;
  points: [number, number, number][];
  diameter: number;            // mm
  role: 'main-tie' | 'stirrup' | 'skin' | 'distribution';
  color: string;
  /** closed polyline (stirrups / ties) */
  closed?: boolean;
}

/** An editable point load / bearing reaction along an element. */
export interface PointLoad {
  id: string;
  x: number;      // position along the element length (mm, model X)
  load: number;   // factored vertical load (N, +ve downward)
}

/** A reinforcement group defined by bar diameter and centre-to-centre spacing. */
export interface BarGroup {
  dia: number;      // bar diameter (mm)
  spacing: number;  // centre-to-centre spacing (mm)
}

/** A reinforcement group defined by bar diameter and a fixed bar count. */
export interface BarGroupCount {
  dia: number;
  count: number;
}

/** An applied load shown as an arrow in the viewport. */
export interface LoadArrow {
  pos: [number, number, number];   // point of application (mm)
  dir: [number, number, number];   // unit direction of the force
  magnitude: number;               // N
  label: string;
}

export interface RenderModel {
  solids: RenderSolid[];
  rebar: RenderRebar[];
  loads: LoadArrow[];
  /** overall bounding box, for camera framing */
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

export interface NodeSpec {
  id: string;
  type: NodeType;
  area: number;                // mm^2 — nodal-zone bearing area
}

/** The complete analysis-ready model produced by an element module. */
export interface ElementModel {
  truss: {
    nodes: TrussNode[];
    members: TrussMember[];
    loads: TrussLoad[];
  };
  fem: {
    domain: VoxelDomain;
    loads: FemLoad[];
    supports: FemSupport[];
  };
  /** 2D plane-stress problem for the continuum CSFM solver (book Ch. 3.6) */
  continuum: ContinuumProblem;
  /** which 2D plane the continuum slice lies in — 'x' = X–Y elevation,
   *  'z' = Z–Y section. Defaults to 'x' when omitted. */
  continuumOrient?: 'x' | 'z';
  render: RenderModel;
  ties: TieDesign[];
  struts: StrutGeometry[];
  nodes: NodeSpec[];
  /** warnings — issues the user should act on */
  notes: string[];
  /** informational notes — modelling remarks, no action needed */
  info: string[];
  /** total factored vertical load applied (N) — for reporting */
  totalLoad: number;
}

/**
 * Colour palette — follows the CSFM book convention: compression fields RED,
 * tension fields BLUE (book Figs. 3.5 / 3.6), concrete light grey.
 */
export const PALETTE = {
  concrete: '#c4c8cf',
  column: '#a6abb4',
  pile: '#9a9ea7',
  bearing: '#566070',
  strut: '#d23b30',      // compression — red
  tie: '#2f7fc4',        // tension — blue
  rebarMain: '#2f7fc4',
  rebarStirrup: '#27ae60',
  load: '#f4b740',
};
