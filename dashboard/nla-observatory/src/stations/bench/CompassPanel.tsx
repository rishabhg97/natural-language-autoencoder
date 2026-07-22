/**
 * Panel 3a — reconstruction compass. Precomputed PCA geometry (validation-
 * fitted, display only) with the stored target as a crosshair and each lane's
 * cell as a dot, plus an arrow from the identity cell to each lane cell.
 * The metrics table below carries the stored-snapshot numbers per critic.
 */

import { useMemo } from "react";
import type { Critic } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { extent, fmt } from "../../data/format";
import { Axes, Dot, makeFrame, useTooltip } from "../../components/charts";
import { criticMetrics, LANE_META, type LaneCell } from "./laneModel";
import { LaneLegend } from "./benchUi";

const W = 640;
const H = 320;

function pad(domain: [number, number]): [number, number] {
  const span = domain[1] - domain[0] || 1;
  return [domain[0] - span * 0.14, domain[1] + span * 0.14];
}

function MetricsTable(props: { rack: LaneCell[] }) {
  const critics: readonly Critic[] = ["primary", "independent"];
  return (
    <div className="bench-scroll">
      <table className="data-table" tabIndex={0} aria-label="stored-snapshot metrics per lane per critic">
        <thead>
          <tr>
            <th scope="col">lane</th>
            {critics.map((c) => (
              <th key={c} scope="col" colSpan={3}>
                {c}
              </th>
            ))}
          </tr>
          <tr>
            <th scope="col" aria-hidden />
            {critics.map((c) => (
              <MetricHead key={c} />
            ))}
          </tr>
        </thead>
        <tbody>
          {props.rack.map((laneCell) => (
            <tr key={laneCell.lane}>
              <td>{LANE_META[laneCell.lane].label}</td>
              {critics.map((critic) => {
                const m = laneCell.cell ? criticMetrics(laneCell.cell, critic) : null;
                return m ? (
                  <MetricCells key={critic} dmse={m.dmse} cosine={m.cosine} norm={m.norm_ratio} />
                ) : (
                  <MetricMissing key={critic} />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricHead() {
  return (
    <>
      <th scope="col" className="num">
        dmse
      </th>
      <th scope="col" className="num">
        cosine
      </th>
      <th scope="col" className="num">
        norm ratio
      </th>
    </>
  );
}

function MetricCells(props: { dmse: number; cosine: number; norm: number }) {
  return (
    <>
      <td className="num">{fmt(props.dmse)}</td>
      <td className="num">{fmt(props.cosine)}</td>
      <td className="num">{fmt(props.norm)}</td>
    </>
  );
}

function MetricMissing() {
  const title = "not scored by independent critic";
  return (
    <>
      <td className="num" title={title}>
        —
      </td>
      <td className="num" title={title}>
        —
      </td>
      <td className="num" title={title}>
        —
      </td>
    </>
  );
}

export default function CompassPanel(props: {
  rack: LaneCell[];
  target: { x: number; y: number; z: number } | null;
  rowId: string;
}) {
  const { rack, target } = props;
  const tooltip = useTooltip();

  const points = useMemo(
    () =>
      rack
        .filter((l) => l.cell && l.cell.geometry)
        .map((l) => ({
          lane: l.lane,
          x: l.cell!.geometry!.x,
          y: l.cell!.geometry!.y,
          cellId: l.cell!.cell_id,
        })),
    [rack],
  );

  const identity = points.find((p) => p.lane === "teacher") ?? null;

  const frame = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    if (target) {
      xs.push(target.x);
      ys.push(target.y);
    }
    return makeFrame({
      width: W,
      height: H,
      xDomain: pad(extent(xs)),
      yDomain: pad(extent(ys)),
      margin: { left: 52, right: 16, bottom: 42 },
    });
  }, [points, target]);

  const lanesShown = rack.map((l) => l.lane);

  return (
    <Panel
      title="Reconstruction compass"
      span={6}
      badges={
        <>
          <Badge status="qualified" label="stored-snapshot" title="metrics table" />
          <Badge status="exploratory" label="validation-fitted PCA" title="projection basis" />
        </>
      }
      sub="validation-fitted PCA of unnormalized vectors — display only; metrics are computed in native 2688-d. Arrows run from the identity cell to each lane cell."
    >
      {points.length === 0 && !target ? (
        <UnavailableBox>
          no precomputed geometry for this selection — nothing is plotted in its place.
        </UnavailableBox>
      ) : (
        <>
          <LaneLegend
            lanes={lanesShown}
            extra={
              <span className="bench-legend-item">
                <span aria-hidden>✛</span> stored target activation
              </span>
            }
          />
          <svg
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            className="chart-svg bench-chart"
            role="img"
            aria-label={`Reconstruction compass: PCA positions of the stored target and ${points.length} lane cells for ${props.rowId}`}
          >
            <Axes frame={frame} xLabel="PC1 (unnormalized, validation-fitted — display only)" yLabel="PC2" />
            {identity
              ? points
                  .filter((p) => p.lane !== "teacher")
                  .map((p) => (
                    <line
                      key={`arrow-${p.lane}`}
                      x1={frame.x(identity.x)}
                      y1={frame.y(identity.y)}
                      x2={frame.x(p.x)}
                      y2={frame.y(p.y)}
                      stroke={LANE_META[p.lane].color}
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      opacity={0.8}
                    />
                  ))
              : null}
            {target ? (
              <g aria-label="stored target activation">
                <line
                  x1={frame.x(target.x) - 9}
                  x2={frame.x(target.x) + 9}
                  y1={frame.y(target.y)}
                  y2={frame.y(target.y)}
                  stroke="var(--ink)"
                  strokeWidth={1.5}
                />
                <line
                  x1={frame.x(target.x)}
                  x2={frame.x(target.x)}
                  y1={frame.y(target.y) - 9}
                  y2={frame.y(target.y) + 9}
                  stroke="var(--ink)"
                  strokeWidth={1.5}
                />
                <text x={frame.x(target.x) + 12} y={frame.y(target.y) - 6}>
                  target
                </text>
              </g>
            ) : null}
            {(() => {
              // Nudge coincident/near labels apart vertically so overlapping
              // lane points (e.g. placebo == identity) stay readable.
              const placed: { x: number; y: number }[] = [];
              const labelY = (px: number, py: number): number => {
                let y = py + 3;
                while (placed.some((q) => Math.abs(q.x - px) < 46 && Math.abs(q.y - y) < 11)) {
                  y += 11;
                }
                placed.push({ x: px, y });
                return y;
              };
              return points.map((p) => (
              <g
                key={p.lane}
                role="img"
                aria-label={`${LANE_META[p.lane].label}: PC1 ${fmt(p.x, 2)}, PC2 ${fmt(p.y, 2)}`}
              >
                <title>{`${LANE_META[p.lane].label}: PC1 ${fmt(p.x, 2)}, PC2 ${fmt(p.y, 2)}`}</title>
                <Dot
                  cx={frame.x(p.x)}
                  cy={frame.y(p.y)}
                  r={5}
                  fill={LANE_META[p.lane].color}
                  onHover={(e) =>
                    tooltip.show({
                      x: e.clientX,
                      y: e.clientY,
                      content: (
                        <span>
                          <span className="tt-label">{LANE_META[p.lane].label}</span>{" "}
                          <span className="tt-value">
                            PC1 {fmt(p.x, 2)} · PC2 {fmt(p.y, 2)}
                          </span>
                        </span>
                      ),
                    })
                  }
                  onLeave={tooltip.hide}
                />
                <text x={frame.x(p.x) + 8} y={labelY(frame.x(p.x) + 8, frame.y(p.y))}>
                  {LANE_META[p.lane].short}
                </text>
              </g>
              ));
            })()}
          </svg>
          {!target ? (
            <p className="bench-note">no stored target geometry for this row.</p>
          ) : null}
          {target ? (
            <p className="bench-note bench-note-strong" role="note">
              Read the distance to the target with caution: this projection keeps raw magnitude, so
              lane norm ratios well above 1 push every reconstruction away from the target even when
              the directional cosine below is high. Distances here must not be read against the
              directional claim — use the table.
            </p>
          ) : null}
        </>
      )}
      <MetricsTable rack={rack} />
    </Panel>
  );
}
