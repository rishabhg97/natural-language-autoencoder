/**
 * Panel 3d — causal wake. Teacher-forced per-position divergence after the
 * patch, one line per lane. Teacher-forcing masks compounding divergence, so
 * the wake understates free-running effects.
 */

import { useMemo, useState } from "react";
import type { WakePoint } from "../../data/types";
import type { AppState } from "../../app/urlState";
import { Badge, Panel, Segmented, UnavailableBox } from "../../components/ui";
import { extent, fmt } from "../../data/format";
import { Axes, Dot, linePath, makeFrame, useTooltip } from "../../components/charts";
import { LANE_META, type LaneCell } from "./laneModel";
import { LaneLegend, LaneSwatch } from "./benchUi";

export type WakeMetric = "kl" | "js" | "top_10_overlap";

const WAKE_METRICS: readonly WakeMetric[] = ["kl", "js", "top_10_overlap"];
const METRIC_LABEL: Record<WakeMetric, string> = {
  kl: "KL",
  js: "JS",
  top_10_overlap: "top-10 overlap",
};

export function isWakeMetric(v: string | null): v is WakeMetric {
  return v === "kl" || v === "js" || v === "top_10_overlap";
}

const W = 660;
const H = 260;

export default function WakePanel(props: {
  rack: LaneCell[];
  metric: WakeMetric;
  update: (patch: Partial<AppState>) => void;
}) {
  const { rack, metric, update } = props;
  const tooltip = useTooltip();
  const [hoverOffset, setHoverOffset] = useState<number | null>(null);

  const lanes = useMemo(
    () => rack.filter((l) => l.cell !== null && l.cell.wake.length > 0),
    [rack],
  );

  const frame = useMemo(() => {
    const offsets = lanes.flatMap((l) => l.cell!.wake.map((w) => w.offset));
    const values = lanes.flatMap((l) => l.cell!.wake.map((w) => w[metric]));
    const [lo, hi] = extent(values);
    return makeFrame({
      width: W,
      height: H,
      xDomain: offsets.length ? extent(offsets) : [1, 16],
      yDomain: [Math.min(0, lo), hi * 1.06 || 1],
      margin: { left: 52, right: 112, bottom: 34 },
    });
  }, [lanes, metric]);

  // End-of-line labels with simple collision nudging (direct labels, no hover needed).
  const endLabels = useMemo(() => {
    const labels = lanes
      .map((l) => {
        const last = l.cell!.wake[l.cell!.wake.length - 1];
        return { lane: l.lane, x: frame.x(last.offset), y: frame.y(last[metric]) };
      })
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      if (labels[i].y - labels[i - 1].y < 12) labels[i].y = labels[i - 1].y + 12;
    }
    return labels;
  }, [lanes, frame, metric]);

  const offsets = lanes[0]?.cell!.wake.map((w) => w.offset) ?? [];

  return (
    <Panel
      title="Causal wake"
      span={6}
      badges={<Badge status="exploratory" label="functional: validation-only" />}
      sub="teacher-forced per-position divergence after the patch; teacher-forcing masks compounding divergence."
    >
      <div className="bench-controls">
        <Segmented
          options={WAKE_METRICS}
          value={metric}
          onChange={(v) => update({ view: v })}
          label="wake metric"
          format={(m) => METRIC_LABEL[m]}
        />
      </div>
      {lanes.length === 0 ? (
        <UnavailableBox>no precomputed wake trajectories for this selection.</UnavailableBox>
      ) : (
        <>
          <LaneLegend lanes={lanes.map((l) => l.lane)} />
          <svg
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            className="chart-svg bench-chart"
            role="img"
            aria-label={`Causal wake: ${METRIC_LABEL[metric]} by offset after the patch for ${lanes.length} lanes; full values in the data table below`}
          >
            <Axes
              frame={frame}
              xLabel="offset after patch (teacher-forced)"
              yLabel={METRIC_LABEL[metric]}
              xFormat={(v) => String(Math.round(v))}
              yFormat={(v) => fmt(v, 2)}
            />
            {lanes.map((l) => (
              <path
                key={l.lane}
                d={linePath(
                  frame,
                  l.cell!.wake.map((w) => ({ x: w.offset, y: w[metric] })),
                )}
                fill="none"
                stroke={LANE_META[l.lane].color}
                strokeWidth={2}
              />
            ))}
            {hoverOffset !== null
              ? lanes.map((l) => {
                  const pt = l.cell!.wake.find((w) => w.offset === hoverOffset);
                  return pt ? (
                    <Dot
                      key={`h-${l.lane}`}
                      cx={frame.x(pt.offset)}
                      cy={frame.y(pt[metric])}
                      r={4}
                      fill={LANE_META[l.lane].color}
                    />
                  ) : null;
                })
              : null}
            {endLabels.map((lab) => (
              <text key={lab.lane} x={lab.x + 8} y={lab.y + 3}>
                {LANE_META[lab.lane].short}
              </text>
            ))}
            {offsets.map((offset) => {
              const step = frame.innerWidth / Math.max(1, offsets.length - 1);
              return (
                <rect
                  key={offset}
                  x={frame.x(offset) - step / 2}
                  y={frame.margin.top}
                  width={step}
                  height={frame.innerHeight}
                  fill="transparent"
                  onPointerMove={(e) => {
                    setHoverOffset(offset);
                    tooltip.show({
                      x: e.clientX,
                      y: e.clientY,
                      content: (
                        <div>
                          <div className="tt-label">offset {offset}</div>
                          {lanes.map((l) => {
                            const pt = l.cell!.wake.find((w) => w.offset === offset);
                            return (
                              <div key={l.lane}>
                                <LaneSwatch lane={l.lane} /> {LANE_META[l.lane].short}:{" "}
                                <span className="tt-value">
                                  {pt ? fmt(pt[metric], 4) : "—"}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      ),
                    });
                  }}
                  onPointerLeave={() => {
                    setHoverOffset(null);
                    tooltip.hide();
                  }}
                />
              );
            })}
          </svg>
          <details className="bench-details">
            <summary>data table — {METRIC_LABEL[metric]} per offset per lane</summary>
            <div className="bench-scroll">
              <table className="data-table" tabIndex={0} aria-label={`wake ${METRIC_LABEL[metric]} values`}>
                <thead>
                  <tr>
                    <th scope="col" className="num">
                      offset
                    </th>
                    {lanes.map((l) => (
                      <th key={l.lane} scope="col" className="num">
                        {LANE_META[l.lane].short}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {offsets.map((offset) => (
                    <tr key={offset}>
                      <td className="num">{offset}</td>
                      {lanes.map((l) => {
                        const pt: WakePoint | undefined = l.cell!.wake.find(
                          (w) => w.offset === offset,
                        );
                        return (
                          <td key={l.lane} className="num">
                            {pt ? fmt(pt[metric], 4) : "—"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </Panel>
  );
}
