/**
 * Panel 3c — divergence gauges. All four lanes overlaid per metric as grouped
 * horizontal bullet bars. The identity lane is the harness floor; this bundle
 * ships no external reference rails, so the control lanes ARE the rails.
 */

import { useMemo } from "react";
import type { BenchCell } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { fmt } from "../../data/format";
import { HBar } from "../../components/charts";
import { LANE_META, type LaneCell } from "./laneModel";
import { LaneLegend } from "./benchUi";

type BehaviorKey = keyof Pick<
  BenchCell["behavior"],
  "js_divergence" | "kl_original_to_patched" | "top_10_overlap" | "top_50_overlap" | "logit_pearson"
>;

const METRICS: { key: BehaviorKey; label: string; hint: string }[] = [
  { key: "js_divergence", label: "JS divergence", hint: "higher = bigger shift" },
  { key: "kl_original_to_patched", label: "KL(orig → patched)", hint: "higher = bigger shift" },
  { key: "top_10_overlap", label: "top-10 overlap", hint: "lower = bigger shift" },
  { key: "top_50_overlap", label: "top-50 overlap", hint: "lower = bigger shift" },
  { key: "logit_pearson", label: "logit Pearson r", hint: "lower = bigger shift" },
];

const W = 660;
const LABEL_W = 196;
const VALUE_W = 84;
const BAR_H = 9;
const BAR_GAP = 3;
const GROUP_PAD = 16;

/** Relative tolerance for "does the edit separate from a control?". */
const SEP_TOL = 0.05;

export default function GaugesPanel(props: { rack: LaneCell[] }) {
  const { rack } = props;
  const lanes = useMemo(() => rack.filter((l) => l.cell !== null), [rack]);
  const groupH = lanes.length * (BAR_H + BAR_GAP) + GROUP_PAD;
  const height = METRICS.length * groupH + 8;
  const plotW = W - LABEL_W - VALUE_W;

  // Computed readouts from the displayed numbers only: (a) lanes whose stored
  // values coincide exactly on every gauge, (b) whether the edit separates
  // from every control lane anywhere.
  const readout = useMemo(() => {
    const byLane = new Map(lanes.map((l) => [l.lane, l.cell!.behavior]));
    const laneKeys = lanes.map((l) => l.lane);
    const coincident: string[] = [];
    for (let i = 0; i < laneKeys.length; i++) {
      for (let j = i + 1; j < laneKeys.length; j++) {
        const a = byLane.get(laneKeys[i])!;
        const b = byLane.get(laneKeys[j])!;
        if (METRICS.every((m) => a[m.key] === b[m.key])) {
          coincident.push(
            `${LANE_META[laneKeys[i]].label} and ${LANE_META[laneKeys[j]].label}`,
          );
        }
      }
    }
    const edit = byLane.get("edit");
    const controls = laneKeys.filter((k) => k !== "edit");
    let separated: string[] = [];
    if (edit && controls.length > 0) {
      separated = METRICS.filter((m) =>
        controls.every((k) => {
          const c = byLane.get(k)!;
          const denom = Math.max(Math.abs(c[m.key]), 1e-9);
          return Math.abs(edit[m.key] - c[m.key]) / denom > SEP_TOL;
        }),
      ).map((m) => m.label);
    }
    return { coincident, separated, hasEdit: Boolean(edit && controls.length) };
  }, [lanes]);

  if (lanes.length === 0) {
    return (
      <Panel title="Divergence gauges" span={6}>
        <UnavailableBox>no behavior cells resolved for this selection.</UnavailableBox>
      </Panel>
    );
  }

  return (
    <Panel
      title="Divergence gauges"
      span={6}
      badges={<Badge status="exploratory" label="functional: validation-only" />}
      sub="The identity lane is the harness floor — the unedited teacher re-encoding already moves the distribution this much. No external reference rails exist in this bundle beyond the lanes themselves: controls are the rails."
    >
      <LaneLegend lanes={lanes.map((l) => l.lane)} />
      <svg
        viewBox={`0 0 ${W} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="chart-svg bench-chart"
        role="img"
        aria-label={`Divergence gauges: ${METRICS.length} behavior metrics with ${lanes.length} lanes each; exact values are printed at each bar end`}
      >
        {METRICS.map((metric, mi) => {
          const y0 = 8 + mi * groupH;
          const values = lanes.map((l) => l.cell!.behavior[metric.key]);
          const lo = Math.min(0, ...values);
          const hi = Math.max(1e-9, ...values);
          const scale = (v: number) => LABEL_W + ((v - lo) / (hi - lo || 1)) * plotW;
          return (
            <g key={metric.key}>
              <text x={0} y={y0 + 10} fontWeight={650}>
                {metric.label}
              </text>
              <text x={0} y={y0 + 22} className="tick-label">
                {metric.hint}
              </text>
              <line
                x1={scale(Math.max(lo, 0))}
                x2={scale(Math.max(lo, 0))}
                y1={y0}
                y2={y0 + lanes.length * (BAR_H + BAR_GAP)}
                className="axis-line"
              />
              {lanes.map((l, li) => {
                const v = l.cell!.behavior[metric.key];
                const y = y0 + li * (BAR_H + BAR_GAP);
                const x1 = scale(v);
                return (
                  <g key={l.lane}>
                    <text
                      x={LABEL_W - 6}
                      y={y + BAR_H - 1}
                      textAnchor="end"
                      className="tick-label"
                    >
                      {LANE_META[l.lane].short}
                    </text>
                    <HBar
                      x0={scale(Math.max(lo, 0))}
                      x1={x1}
                      y={y}
                      height={BAR_H}
                      fill={LANE_META[l.lane].color}
                    />
                    <text x={x1 + 4} y={y + BAR_H - 1}>
                      <title>{`${LANE_META[l.lane].label}: ${metric.label} = ${fmt(v, 4)}`}</title>
                      {fmt(v, 3)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      {readout.hasEdit ? (
        readout.separated.length > 0 ? (
          <p className="bench-note">
            Computed from the values above: the edit differs from every control lane by more than{" "}
            {Math.round(SEP_TOL * 100)}% on {readout.separated.join(", ")}. Check the dose and wake
            panes before treating that as a semantic effect.
          </p>
        ) : (
          <p className="bench-note">
            Computed from the values above: the edit&apos;s movement stays within ~
            {Math.round(SEP_TOL * 100)}% of at least one control lane on every gauge — no
            lane-specific effect on this cell.
          </p>
        )
      ) : null}
      {readout.coincident.length > 0 ? (
        <p className="bench-note">
          Note: {readout.coincident.join("; ")} carry byte-identical stored values on this cell, so
          their bars overlap exactly.
        </p>
      ) : null}
    </Panel>
  );
}
