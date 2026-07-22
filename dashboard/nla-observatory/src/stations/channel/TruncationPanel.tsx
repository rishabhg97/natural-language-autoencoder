/**
 * WORDS-BUY-DIRECTION / RATE-DISTORTION — dMSE vs prefix length in words for
 * the selected row (blue) over all-row context lines, with the row's identity
 * dMSE as a labelled reference line.
 */

import { useMemo } from "react";
import type { ChannelShard } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { Axes, Dot, linePath, makeFrame, useTooltip } from "../../components/charts";
import { fmt, fmtPct, shortRowId } from "../../data/format";
import { LegendItem } from "./lib";

const W = 680;
const H = 300;

export default function TruncationPanel(props: {
  channel: ChannelShard;
  selectedRowId: string;
}) {
  const tooltip = useTooltip();
  const truncation = props.channel.truncation;
  const allRows = useMemo(() => Object.entries(truncation), [truncation]);
  const selected = truncation[props.selectedRowId];
  const identity = useMemo(
    () =>
      props.channel.identity.find(
        (i) => i.row_id === props.selectedRowId && i.critic === "primary",
      ),
    [props.channel.identity, props.selectedRowId],
  );

  const frame = useMemo(() => {
    const words = allRows.flatMap(([, pts]) => pts.map((p) => p.words));
    const dmses = allRows.flatMap(([, pts]) => pts.map((p) => p.dmse));
    if (identity) dmses.push(identity.dmse);
    return makeFrame({
      width: W,
      height: H,
      xDomain: [0, Math.max(...words, 1) * 1.03],
      yDomain: [0, Math.max(...dmses, 0.001) * 1.06],
      margin: { left: 52, bottom: 34 },
    });
  }, [allRows, identity]);

  return (
    <Panel
      id="chan-truncation"
      title="Words buy direction (rate–distortion)"
      span={8}
      badges={<Badge status="qualified" label="stored-snapshot" />}
      sub={
        <>
          AR-usable information under prefix truncation of the teacher explanation — truncated text
          is out-of-distribution for the AR critic. Scored under the primary critic.
        </>
      }
    >
      {allRows.length === 0 ? (
        <UnavailableBox>channel.json carries no truncation sweeps.</UnavailableBox>
      ) : !selected ? (
        <UnavailableBox>
          No truncation sweep for row {shortRowId(props.selectedRowId)} in the shard.
        </UnavailableBox>
      ) : (
        <>
          <div className="chan-legend" role="group" aria-label="Truncation legend">
            <LegendItem swatch="var(--series-1)" line>
              selected row {shortRowId(props.selectedRowId)}
            </LegendItem>
            <LegendItem swatch="var(--ink-3)" line>
              all {allRows.length} panel rows (context)
            </LegendItem>
            <LegendItem swatch="var(--ink-2)" line>
              identity dMSE (untruncated, this row)
            </LegendItem>
          </div>
          <svg
            className="chan-chart"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Directional MSE vs prefix length in words for row ${shortRowId(props.selectedRowId)}, from ${fmt(selected[0]?.dmse ?? NaN)} at ${selected[0]?.words ?? 0} words down to ${fmt(selected[selected.length - 1]?.dmse ?? NaN)} at ${selected[selected.length - 1]?.words ?? 0} words. Full values are in the table below.`}
          >
            <Axes
              frame={frame}
              xLabel="teacher-explanation prefix length (words)"
              yLabel="dMSE"
              xFormat={(v) => fmt(v, 0)}
              yFormat={(v) => fmt(v, 2)}
            />
            {allRows.map(([rowId, pts]) =>
              rowId === props.selectedRowId ? null : (
                <path
                  key={rowId}
                  d={linePath(frame, pts.map((p) => ({ x: p.words, y: p.dmse })))}
                  fill="none"
                  stroke="var(--ink-3)"
                  strokeWidth={1}
                  opacity={0.25}
                  aria-hidden
                />
              ),
            )}
            {identity ? (
              <g aria-hidden>
                <line
                  x1={frame.margin.left}
                  x2={frame.margin.left + frame.innerWidth}
                  y1={frame.y(identity.dmse)}
                  y2={frame.y(identity.dmse)}
                  stroke="var(--ink-2)"
                  strokeWidth={1}
                />
                <text
                  x={frame.margin.left + frame.innerWidth - 4}
                  y={frame.y(identity.dmse) - 4}
                  textAnchor="end"
                  fill="var(--ink-2)"
                  fontSize={10}
                >
                  identity {fmt(identity.dmse)}
                </text>
              </g>
            ) : null}
            <path
              d={linePath(frame, selected.map((p) => ({ x: p.words, y: p.dmse })))}
              fill="none"
              stroke="var(--series-1)"
              strokeWidth={2}
              aria-hidden
            />
            {selected.map((p) => (
              <Dot
                key={p.fraction}
                cx={frame.x(p.words)}
                cy={frame.y(p.dmse)}
                fill="var(--series-1)"
                onHover={(e) =>
                  tooltip.show({
                    x: e.clientX,
                    y: e.clientY,
                    content: (
                      <div>
                        <div className="tt-value">
                          {p.words} words ({fmtPct(p.fraction, 0)} prefix)
                        </div>
                        <div className="tt-label">dMSE {fmt(p.dmse)} · cosine {fmt(p.cosine)}</div>
                      </div>
                    ),
                  })
                }
                onLeave={tooltip.hide}
              />
            ))}
          </svg>
          {!identity ? (
            <UnavailableBox>
              No primary-critic identity metric for this row in the shard — reference line omitted.
            </UnavailableBox>
          ) : null}
          <details className="chan-details">
            <summary>Truncation table for {shortRowId(props.selectedRowId)}</summary>
            <div className="chan-table-wrap">
              <table className="data-table" tabIndex={0}>
                <thead>
                  <tr>
                    <th scope="col" className="num">prefix fraction</th>
                    <th scope="col" className="num">words</th>
                    <th scope="col" className="num">dMSE</th>
                    <th scope="col" className="num">cosine</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.map((p) => (
                    <tr key={p.fraction}>
                      <td className="num">{fmtPct(p.fraction, 0)}</td>
                      <td className="num">{p.words}</td>
                      <td className="num">{fmt(p.dmse)}</td>
                      <td className="num">{fmt(p.cosine)}</td>
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
