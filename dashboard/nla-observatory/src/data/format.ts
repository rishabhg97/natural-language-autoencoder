/** Number/text formatting helpers shared by all stations. */

export function fmt(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return "—";
  if (value !== 0 && Math.abs(value) < 10 ** -digits) {
    return value.toExponential(1);
  }
  return value.toFixed(digits).replace(/\.?0+$/, (m) => (m.startsWith(".") ? "" : m));
}

export function fmtFixed(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function fmtPct(value: number, digits = 1): string {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

export function fmtSigned(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return "—";
  const s = value.toFixed(digits);
  return value > 0 ? `+${s}` : s;
}

export function shortHash(hash: string, length = 10): string {
  return hash ? `${hash.slice(0, length)}…` : "—";
}

export function shortRowId(rowId: string): string {
  return rowId.replace(/^validation-/, "v-");
}

/** Deterministic clamp of long text for previews; full text stays reachable. */
export function preview(text: string, max = 140): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  return collapsed.length <= max ? collapsed : `${collapsed.slice(0, max - 1)}…`;
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Simple linear scale factory (avoids a d3 dependency for basic charts). */
export function linearScale(
  domain: [number, number],
  range: [number, number],
): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
}

export function extent(values: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [0, 1];
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

/** Nice ticks for a linear axis (5-ish). */
export function ticks(domain: [number, number], count = 5): number[] {
  const [lo, hi] = domain;
  const span = hi - lo;
  if (span <= 0) return [lo];
  const step = 10 ** Math.floor(Math.log10(span / count));
  const err = (span / count) / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = step * mult;
  const start = Math.ceil(lo / s) * s;
  const out: number[] = [];
  for (let v = start; v <= hi + s * 1e-9; v += s) out.push(Number(v.toFixed(12)));
  return out;
}
