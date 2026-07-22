/**
 * AUDIT station — defensive helpers for the loosely-typed corners of the
 * audit shard (drift card, magnitude fit, privacy card are Record<string,
 * unknown> by contract; the builder fails closed but we still read carefully
 * and never invent values that are not present).
 */

import { Fragment } from "react";
import { fmt } from "../../data/format";

export function asRecord(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

export function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function asString(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

export function humanize(key: string): string {
  return key.replace(/_/g, " ");
}

function renderScalar(v: string | number | boolean): string {
  if (typeof v === "number") return fmt(v, 4);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return v;
}

/**
 * Render the scalar entries of an unknown record verbatim as a kv-list.
 * Nested objects/arrays are skipped; callers render those explicitly so
 * nothing is silently flattened away.
 */
export function KvList(props: { record: Record<string, unknown>; skip?: readonly string[] }) {
  const entries = Object.entries(props.record).filter(
    ([k, v]) =>
      !(props.skip ?? []).includes(k) &&
      (typeof v === "number" || typeof v === "string" || typeof v === "boolean"),
  );
  if (entries.length === 0) return null;
  return (
    <dl className="kv-list">
      {entries.map(([k, v]) => (
        <Fragment key={k}>
          <dt>{humanize(k)}</dt>
          <dd>{renderScalar(v as string | number | boolean)}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
