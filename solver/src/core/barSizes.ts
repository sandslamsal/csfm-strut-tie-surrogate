/**
 * Standard reinforcing-bar sizes for the SI (metric) and US (imperial/ASTM)
 * systems. Diameters are stored internally in mm; the picker shows the
 * designation appropriate to the active unit system.
 */

import type { UnitSystem } from './units';

export interface BarSize {
  label: string;   // designation shown to the user
  dia: number;     // nominal diameter (mm)
}

/** Metric reinforcing bars (ISO 6935 / common European-Canadian sizes). */
export const SI_BARS: BarSize[] = [
  { label: 'Ø8', dia: 8 },
  { label: 'Ø10', dia: 10 },
  { label: 'Ø12', dia: 12 },
  { label: 'Ø14', dia: 14 },
  { label: 'Ø16', dia: 16 },
  { label: 'Ø20', dia: 20 },
  { label: 'Ø25', dia: 25 },
  { label: 'Ø28', dia: 28 },
  { label: 'Ø32', dia: 32 },
  { label: 'Ø40', dia: 40 },
];

/** US imperial reinforcing bars (ASTM A615 — # designations). */
export const US_BARS: BarSize[] = [
  { label: '#3', dia: 9.525 },
  { label: '#4', dia: 12.7 },
  { label: '#5', dia: 15.875 },
  { label: '#6', dia: 19.05 },
  { label: '#7', dia: 22.225 },
  { label: '#8', dia: 25.4 },
  { label: '#9', dia: 28.651 },
  { label: '#10', dia: 32.258 },
  { label: '#11', dia: 35.814 },
  { label: '#14', dia: 43.0 },
  { label: '#18', dia: 57.33 },
];

export function barTable(sys: UnitSystem): BarSize[] {
  return sys === 'SI' ? SI_BARS : US_BARS;
}

/** Designation of the standard bar nearest a given diameter (mm). */
export function nearestBar(dia: number, sys: UnitSystem): BarSize {
  const table = barTable(sys);
  return table.reduce((a, b) =>
    Math.abs(b.dia - dia) < Math.abs(a.dia - dia) ? b : a,
  );
}
