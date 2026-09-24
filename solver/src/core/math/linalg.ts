/**
 * Lightweight linear-algebra utilities for the structural solvers.
 * Dense Gaussian elimination for small systems (truss / STM) and a sparse
 * preconditioned conjugate-gradient solver for the finite-element model.
 */

/** Dense linear solve A x = b by Gaussian elimination with partial pivoting. */
export function solveDense(A: number[][], b: number[]): number[] {
  const n = b.length;
  // augmented copy
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // pivot
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-12) {
      throw new Error('Singular system — model is unstable (check supports).');
    }
    [M[col], M[piv]] = [M[piv], M[col]];
    // eliminate
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  // back-substitution
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return x;
}

/* ----------------------------------------------------------------------- */
/* Sparse matrix in coordinate / CSR-ish form for the FE solver              */
/* ----------------------------------------------------------------------- */

export class SparseSym {
  /** map of "i,j" -> value, upper triangle only (i<=j) */
  private data = new Map<number, number>();
  constructor(public readonly n: number) {}

  add(i: number, j: number, v: number): void {
    if (v === 0) return;
    const a = Math.min(i, j);
    const b = Math.max(i, j);
    const key = a * this.n + b;
    this.data.set(key, (this.data.get(key) ?? 0) + v);
  }

  get(i: number, j: number): number {
    const a = Math.min(i, j);
    const b = Math.max(i, j);
    return this.data.get(a * this.n + b) ?? 0;
  }

  /** matrix-vector product y = A x (A symmetric) */
  multiply(x: number[]): number[] {
    const y = new Array(this.n).fill(0);
    for (const [key, v] of this.data) {
      const i = Math.floor(key / this.n);
      const j = key % this.n;
      y[i] += v * x[j];
      if (i !== j) y[j] += v * x[i];
    }
    return y;
  }

  /** diagonal (used for Jacobi preconditioning) */
  diagonal(): number[] {
    const d = new Array(this.n).fill(0);
    for (let i = 0; i < this.n; i++) d[i] = this.get(i, i);
    return d;
  }
}

/**
 * Jacobi-preconditioned conjugate gradient for a symmetric positive-definite
 * sparse system K u = f. Returns the displacement vector u.
 */
export function conjugateGradient(
  K: SparseSym,
  f: number[],
  opts: { tol?: number; maxIter?: number; u0?: number[] } = {},
): { u: number[]; iterations: number; converged: boolean } {
  const n = f.length;
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? Math.max(2000, 4 * n);
  const diag = K.diagonal().map((d) => (Math.abs(d) > 1e-30 ? d : 1));

  // optional warm-start (e.g. previous solution in an iterative optimiser)
  const warm = !!(opts.u0 && opts.u0.length === n);
  const u = warm ? [...opts.u0!] : new Array(n).fill(0);
  let r: number[];
  if (warm) {
    const Ku = K.multiply(u);
    r = f.map((fi, i) => fi - Ku[i]);
  } else {
    r = [...f];
  }
  let z = r.map((ri, i) => ri / diag[i]);
  let p = [...z];
  let rz = dot(r, z);
  const bnorm = Math.sqrt(dot(f, f)) || 1;

  let iter = 0;
  for (; iter < maxIter; iter++) {
    const Kp = K.multiply(p);
    const alpha = rz / (dot(p, Kp) || 1e-30);
    for (let i = 0; i < n; i++) {
      u[i] += alpha * p[i];
      r[i] -= alpha * Kp[i];
    }
    if (Math.sqrt(dot(r, r)) / bnorm < tol) {
      iter++;
      break;
    }
    z = r.map((ri, i) => ri / diag[i]);
    const rzNew = dot(r, z);
    const beta = rzNew / (rz || 1e-30);
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    rz = rzNew;
  }
  return { u, iterations: iter, converged: iter < maxIter };
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
