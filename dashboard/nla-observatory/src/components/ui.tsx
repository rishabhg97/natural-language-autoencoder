/** Shared UI primitives. Stations compose these; keep them dependency-free. */

import { type ReactNode, useState } from "react";
import { AlertTriangle, Check, Copy } from "lucide-react";
import type { EvidenceStatus } from "../data/types";
import {
  panelGuideFor,
  TERM_DEFINITIONS,
  type PanelGuide,
} from "./panelGuides";

export function Panel(props: {
  title: string;
  sub?: ReactNode;
  span?: 3 | 4 | 5 | 6 | 7 | 8 | 12;
  badges?: ReactNode;
  children: ReactNode;
  id?: string;
  guide?: PanelGuide | false;
}) {
  const span = props.span ?? 6;
  const guide = props.guide === false ? null : (props.guide ?? panelGuideFor(props.title));
  const terms = guide?.terms?.filter((term) => TERM_DEFINITIONS[term]) ?? [];
  return (
    <section className={`panel span-${span}`} id={props.id} aria-label={props.title}>
      <h3>
        {props.title}
        {props.badges}
      </h3>
      {guide || props.sub ? (
        <div className="panel-reading-guide">
          <div>
            <div className="panel-guide-label">What this shows</div>
            <p>{guide?.what ?? props.sub}</p>
            {guide && props.sub ? (
              <p className="panel-evidence-note">
                <strong>Evidence note:</strong> {props.sub}
              </p>
            ) : null}
          </div>
          {guide ? (
            <div>
              <div className="panel-guide-label">How to read it</div>
              <p>{guide.read}</p>
            </div>
          ) : null}
        </div>
      ) : null}
      {terms.length > 0 ? (
        <details className="panel-terms">
          <summary>Terms in this panel ({terms.length})</summary>
          <dl>
            {terms.map((term) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{TERM_DEFINITIONS[term]}</dd>
              </div>
            ))}
          </dl>
        </details>
      ) : null}
      <div className="panel-content">{props.children}</div>
    </section>
  );
}

export interface BriefMetric {
  label: string;
  value: string;
  detail: string;
}

/** Summary-first station header: scientific question, bounded answer, then evidence. */
export function StationBrief(props: {
  station: string;
  question: string;
  answer: ReactNode;
  status: EvidenceStatus | "caveat";
  statusLabel: string;
  metrics: BriefMetric[];
  note?: ReactNode;
}) {
  return (
    <section className="station-brief" aria-labelledby={`${props.station}-brief-title`}>
      <div className="station-brief-copy">
        <div className="station-brief-kicker">{props.station} · guiding question</div>
        <h2 id={`${props.station}-brief-title`}>{props.question}</h2>
        <p className="station-brief-answer">{props.answer}</p>
        {props.note ? <p className="station-brief-note">{props.note}</p> : null}
      </div>
      <div className="station-brief-evidence">
        <Badge status={props.status} label={props.statusLabel} />
        <div className="station-metric-strip">
          {props.metrics.map((metric) => (
            <div className="station-metric" key={metric.label}>
              <div className="station-metric-value">{metric.value}</div>
              <div className="station-metric-label">{metric.label}</div>
              <div className="station-metric-detail">{metric.detail}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const BADGE_LABEL: Record<EvidenceStatus | "caveat" | "selected", string> = {
  qualified: "qualified",
  exploratory: "exploratory",
  negative: "negative",
  unavailable: "unavailable",
  caveat: "caveat",
  selected: "selected",
};

export function Badge(props: {
  status: EvidenceStatus | "caveat" | "selected";
  label?: string;
  title?: string;
}) {
  const label = props.label ?? BADGE_LABEL[props.status];
  return (
    <span
      className={`badge badge-${props.status}`}
      title={props.title}
      role="status"
      aria-label={`evidence status: ${label}`}
    >
      {props.status === "negative" || props.status === "caveat" ? (
        <AlertTriangle size={10} aria-hidden />
      ) : null}
      {label}
    </span>
  );
}

export function HashChip(props: { hash: string; label?: string; title?: string }) {
  const [copied, setCopied] = useState(false);
  const short = `${props.hash.slice(0, 10)}…`;
  return (
    <button
      type="button"
      className="hash-chip"
      title={props.title ?? `${props.hash} (click to copy)`}
      onClick={() => {
        void navigator.clipboard?.writeText(props.hash).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        });
      }}
    >
      {copied ? <Check size={10} aria-hidden /> : <Copy size={10} aria-hidden />}
      {props.label ? `${props.label} ` : ""}
      <span className="mono">{short}</span>
    </button>
  );
}

export function Segmented<T extends string>(props: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  format?: (v: T) => string;
}) {
  return (
    <div className="segmented" role="group" aria-label={props.label}>
      {props.options.map((opt) => (
        <button
          key={opt}
          type="button"
          aria-pressed={opt === props.value}
          onClick={() => props.onChange(opt)}
        >
          {props.format ? props.format(opt) : opt}
        </button>
      ))}
      <span className="visually-hidden" aria-live="polite">
        {props.label}: {props.format ? props.format(props.value) : props.value}
      </span>
    </div>
  );
}

export function SelectControl(props: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="control">
      {props.label}
      <select value={props.value} onChange={(e) => props.onChange(e.target.value)}>
        {props.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function LoadingBox(props: { what: string }) {
  return (
    <div className="state-box" role="status" aria-live="polite">
      Loading {props.what}…
    </div>
  );
}

export function ErrorBox(props: { message: string }) {
  return (
    <div className="state-box error" role="alert">
      Evidence failed to load (fail-closed; nothing is substituted):{"\n"}
      {props.message}
    </div>
  );
}

export function UnavailableBox(props: { children: ReactNode }) {
  return (
    <div className="state-box unavailable">
      <Badge status="unavailable" /> {props.children}
    </div>
  );
}
