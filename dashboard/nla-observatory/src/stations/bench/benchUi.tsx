/** Small BENCH-local UI helpers: lane legend, swatches, deep-link buttons. */

import type { ReactNode } from "react";
import { LANE_META, type LaneKey } from "./laneModel";

export function LaneSwatch(props: { lane: LaneKey }) {
  return (
    <span
      className="bench-swatch"
      style={{ background: LANE_META[props.lane].color }}
      aria-hidden
    />
  );
}

/** Legend pairs color with text — never color alone. */
export function LaneLegend(props: { lanes: readonly LaneKey[]; extra?: ReactNode }) {
  return (
    <div className="bench-legend" role="list" aria-label="lane legend">
      {props.lanes.map((lane) => (
        <span role="listitem" key={lane} className="bench-legend-item">
          <LaneSwatch lane={lane} />
          {LANE_META[lane].label}
        </span>
      ))}
      {props.extra ? <span role="listitem">{props.extra}</span> : null}
    </div>
  );
}

export function LinkButton(props: {
  onClick: () => void;
  children: ReactNode;
  title?: string;
}) {
  return (
    <button type="button" className="bench-link" onClick={props.onClick} title={props.title}>
      {props.children}
    </button>
  );
}

/** Render token text with visible whitespace so BPE pieces stay legible. */
export function visibleToken(text: string): string {
  return text.replace(/ /g, "␣").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
}
