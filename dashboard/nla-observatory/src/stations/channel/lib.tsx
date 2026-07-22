/**
 * CHANNEL station shared bits. Local to src/stations/channel — nothing here
 * carries evidence numbers; everything displayed comes from loaded shards.
 */

import type { ReactNode } from "react";
import type { AppState } from "../../app/urlState";

/** Deep link into the AUDIT station at a claim anchor. */
export function AuditLink(props: {
  claim: string;
  update: (patch: Partial<AppState>) => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      className="chan-audit-link"
      title={`Open AUDIT at claim anchor '${props.claim}'`}
      onClick={() => props.update({ station: "audit", claim: props.claim })}
    >
      audit ▸ {props.label ?? props.claim}
    </button>
  );
}

export function StatTile(props: {
  label: string;
  value: string;
  sub?: ReactNode;
  negative?: boolean;
}) {
  return (
    <div className={`chan-tile${props.negative ? " chan-tile-negative" : ""}`}>
      <div className="chan-tile-value">{props.value}</div>
      <div className="chan-tile-label">{props.label}</div>
      {props.sub ? <div className="chan-tile-sub">{props.sub}</div> : null}
    </div>
  );
}

export function LegendItem(props: {
  swatch: string;
  children: ReactNode;
  line?: boolean;
  dotted?: boolean;
}) {
  const cls = `chan-swatch${props.line ? " chan-swatch-line" : ""}${
    props.dotted ? " chan-swatch-dotted" : ""
  }`;
  return (
    <span className="chan-legend-item">
      <span
        className={cls}
        style={props.dotted ? { borderColor: props.swatch } : { background: props.swatch }}
        aria-hidden
      />
      {props.children}
    </span>
  );
}

/** Control ablation variants of the information waterfall (furniture: always shown). */
export const CONTROL_VARIANTS: ReadonlySet<string> = new Set([
  "av_zero",
  "av_mean",
  "av_shuffled",
  "av_none",
]);

export function waterfallColor(variant: string): string {
  if (variant === "av_real") return "var(--series-1)";
  if (variant === "teacher") return "var(--series-3)";
  if (CONTROL_VARIANTS.has(variant)) return "var(--ink-3)";
  return "var(--series-4)";
}

/**
 * Plain-language meaning per report-e0 waterfall variant. Names come from the
 * report; these one-liners only translate the conditioning, they add no
 * numbers. `mean` is the no-text mean-activation predictor baseline.
 */
export const VARIANT_MEANING: Record<string, string> = {
  av_real: "AV description of the true stored activation",
  teacher: "teacher reference explanation, re-encoded",
  mean: "no text — dataset-mean activation used as the prediction",
  av_mean: "AV description generated from the dataset-mean activation",
  av_zero: "AV description generated from a zeroed activation",
  av_shuffled: "AV description generated from a mismatched (shuffled) activation",
  av_none: "AV description generated with no activation input",
};

/** Shorten content-family ids for compact tables (full id stays in title). */
export function shortFamily(id: string): string {
  const stripped = id.replace(/^cf_/, "");
  return stripped.length <= 10 ? stripped : `${stripped.slice(0, 10)}…`;
}

/** Pad a numeric domain by a fraction on both ends. */
export function padDomain(domain: [number, number], frac = 0.06): [number, number] {
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  return [lo - span * frac, hi + span * frac];
}
