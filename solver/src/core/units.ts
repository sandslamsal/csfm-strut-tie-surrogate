/**
 * Unit system management.
 *
 * INTERNAL CONVENTION: every quantity stored in the project model and used by
 * the analysis engine is in SI base units:
 *   length  -> mm
 *   force   -> N
 *   stress  -> MPa  (= N/mm^2)
 *   moment  -> N*mm
 * Conversions defined here are only applied at the UI boundary (input/output).
 */

export type UnitSystem = 'SI' | 'US';

export type QuantityKind =
  | 'length'      // mm        <-> in
  | 'length_m'    // mm        <-> ft
  | 'crack'       // mm        <-> in   (small lengths — crack widths)
  | 'force'       // N         <-> kip
  | 'force_kN'    // N         <-> kip
  | 'stress'      // MPa       <-> ksi
  | 'stress_psi'  // MPa       <-> psi
  | 'moment'      // N*mm      <-> kip*ft
  | 'area'        // mm^2      <-> in^2
  | 'density'     // kg/m^3    <-> pcf
  | 'mass'        // kg        <-> lb
  | 'dimensionless';

interface UnitDef {
  /** factor to multiply an internal (SI) value by to get the display value */
  factor: number;
  label: string;
}

const SI_UNITS: Record<QuantityKind, UnitDef> = {
  length:        { factor: 1, label: 'mm' },
  length_m:      { factor: 1e-3, label: 'm' },
  crack:         { factor: 1, label: 'mm' },
  force:         { factor: 1e-3, label: 'kN' },
  force_kN:      { factor: 1e-3, label: 'kN' },
  stress:        { factor: 1, label: 'MPa' },
  stress_psi:    { factor: 1, label: 'MPa' },
  moment:        { factor: 1e-6, label: 'kN·m' },
  area:          { factor: 1, label: 'mm²' },
  density:       { factor: 1, label: 'kg/m³' },
  mass:          { factor: 1, label: 'kg' },
  dimensionless: { factor: 1, label: '' },
};

// 1 in = 25.4 mm ; 1 kip = 4448.2216 N ; 1 ksi = 6.894757 MPa
const IN = 25.4;
const KIP = 4448.2216;
const KSI = 6.894757;

const US_UNITS: Record<QuantityKind, UnitDef> = {
  length:        { factor: 1 / IN, label: 'in' },
  length_m:      { factor: 1 / (IN * 12), label: 'ft' },
  crack:         { factor: 1 / IN, label: 'in' },
  force:         { factor: 1 / KIP, label: 'kip' },
  force_kN:      { factor: 1 / KIP, label: 'kip' },
  stress:        { factor: 1 / KSI, label: 'ksi' },
  stress_psi:    { factor: 1 / (KSI / 1000), label: 'psi' },
  moment:        { factor: 1 / (KIP * IN * 12), label: 'kip·ft' },
  area:          { factor: 1 / (IN * IN), label: 'in²' },
  density:       { factor: 0.06242797, label: 'pcf' },     // kg/m³ -> lb/ft³
  mass:          { factor: 2.2046226, label: 'lb' },        // kg -> lb
  dimensionless: { factor: 1, label: '' },
};

export function unitTable(sys: UnitSystem): Record<QuantityKind, UnitDef> {
  return sys === 'SI' ? SI_UNITS : US_UNITS;
}

/** convert an internal SI value to a display value */
export function toDisplay(value: number, kind: QuantityKind, sys: UnitSystem): number {
  return value * unitTable(sys)[kind].factor;
}

/** convert a display value back to internal SI units */
export function toInternal(value: number, kind: QuantityKind, sys: UnitSystem): number {
  return value / unitTable(sys)[kind].factor;
}

export function unitLabel(kind: QuantityKind, sys: UnitSystem): string {
  return unitTable(sys)[kind].label;
}

/** Format an internal value for display with sensible precision. */
export function fmt(
  value: number,
  kind: QuantityKind,
  sys: UnitSystem,
  digits = 2,
): string {
  const v = toDisplay(value, kind, sys);
  if (!isFinite(v)) return '–';
  return v.toFixed(digits);
}

export function fmtWithUnit(
  value: number,
  kind: QuantityKind,
  sys: UnitSystem,
  digits = 2,
): string {
  const label = unitLabel(kind, sys);
  return `${fmt(value, kind, sys, digits)}${label ? ' ' + label : ''}`;
}
