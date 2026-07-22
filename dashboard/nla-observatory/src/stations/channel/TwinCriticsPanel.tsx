/**
 * TWIN CRITICS — per-row identity dMSE under the primary vs independent AR
 * critic, with the shared-teacher confound stated verbatim from the shard.
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import type { ChannelShard, Critic } from "../../data/types";
import { Badge, Panel } from "../../components/ui";
import { Axes, makeFrame, useTooltip } from "../../components/charts";
import { extent, fmt, shortRowId } from "../../data/format";
import { AuditLink, padDomain } from "./lib";

const W = 640;
const H = 320;

const SUMMARY_KEYS = [
  ["cells", "cells"],
  ["identity_directional_mse", "identity dMSE"],
  ["mean_directional_mse", "mean dMSE"],
  ["mean_cosine", "mean cosine"],
] as const;

export default function TwinCriticsPanel(props: {
  channel: ChannelShard;
  critic: Critic;
  selectedRowId: string;
  update: (patch: Partial<AppState>) => void;
}) {
  const { twin_critics } = props.channel;
  const tooltip = useTooltip();
  const rows = twin_critics.per_row;

  const domain = useMemo(() => {
    const all = rows.flatMap((r) => [r.primary_dmse, r.independent_dmse]);
    return padDomain(extent(all));
  }, [rows]);

  const frame = useMemo(
    () =>
      makeFrame({
        width: W,
        height: H,
        xDomain: domain,
        yDomain: domain,
        margin: { left: 52, bottom: 34 },
      }),
    [domain],
  );

  const select = (rowId: string) => props.update({ row: rowId });

  return (
    <Panel
      id="chan-twin-critics"
      title="Twin critics"
      span={5}
      badges={
        <>
          <Badge status="qualified" label="stored-snapshot" />
          <AuditLink claim="stored_snapshot_channel" update={props.update} />
        </>
      }
      sub={twin_critics.confound}
    >
      {rows.length === 0 ? (
        <div className="state-box unavailable">
          <Badge status="unavailable" /> no per-row twin-critic metrics in channel.json.
        </div>
      ) : (
        <>
          <svg
            className="chan-chart"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            role="group"
            aria-label={`Scatter of ${rows.length} rows: primary critic identity dMSE (x) vs independent critic identity dMSE (y), with a y equals x hairline. Each point is a button that selects its row.`}
          >
            <Axes
              frame={frame}
              xLabel="primary dMSE"
              yLabel="independent dMSE"
              xFormat={(v) => fmt(v, 2)}
              yFormat={(v) => fmt(v, 2)}
            />
            <line
              x1={frame.x(domain[0])}
              y1={frame.y(domain[0])}
              x2={frame.x(domain[1])}
              y2={frame.y(domain[1])}
              stroke="var(--axis)"
              strokeWidth={1}
              aria-hidden
            />
            <text
              x={frame.x(domain[1]) - 4}
              y={frame.y(domain[1]) + 12}
              textAnchor="end"
              fill="var(--ink-3)"
              fontSize={10}
              aria-hidden
            >
              y = x
            </text>
            {rows.map((r) => {
              const selected = r.row_id === props.selectedRowId;
              const cx = frame.x(r.primary_dmse);
              const cy = frame.y(r.independent_dmse);
              const showTip = (x: number, y: number) =>
                tooltip.show({
                  x,
                  y,
                  content: (
                    <div>
                      <div className="tt-value">{shortRowId(r.row_id)}</div>
                      <div className="tt-label">primary dMSE {fmt(r.primary_dmse)} · cos {fmt(r.primary_cosine)}</div>
                      <div className="tt-label">independent dMSE {fmt(r.independent_dmse)} · cos {fmt(r.independent_cosine)}</div>
                      <div className="tt-label">click / Enter to select this row</div>
                    </div>
                  ),
                });
              return (
                <g key={r.row_id}>
                  {selected ? (
                    <circle cx={cx} cy={cy} r={9} fill="none" stroke="var(--selected)" strokeWidth={2} aria-hidden />
                  ) : null}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={selected ? 6 : 4}
                    fill="var(--series-1)"
                    stroke="var(--surface-1)"
                    strokeWidth={2}
                    opacity={selected ? 1 : 0.8}
                    tabIndex={0}
                    role="button"
                    aria-label={`Select row ${shortRowId(r.row_id)}: primary dMSE ${fmt(r.primary_dmse)}, independent dMSE ${fmt(r.independent_dmse)}${selected ? " (selected)" : ""}`}
                    style={{ cursor: "pointer" }}
                    onClick={() => select(r.row_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        select(r.row_id);
                      }
                    }}
                    onPointerMove={(e) => showTip(e.clientX, e.clientY)}
                    onPointerLeave={tooltip.hide}
                    onFocus={(e) => {
                      const b = (e.currentTarget as SVGCircleElement).getBoundingClientRect();
                      showTip(b.right, b.top);
                    }}
                    onBlur={tooltip.hide}
                  />
                </g>
              );
            })}
          </svg>
          <div className="chan-table-wrap">
            <table className="data-table" tabIndex={0} aria-label="Primary and independent critic reconstruction metrics">
              <caption className="visually-hidden">
                Twin-critic summary metrics from reports e3 and p2
              </caption>
              <thead>
                <tr>
                  <th scope="col">report</th>
                  <th scope="col">critic</th>
                  {SUMMARY_KEYS.map(([, label]) => (
                    <th key={label} scope="col" className="num">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["e3", twin_critics.e3_summaries],
                    ["p2", twin_critics.p2_summaries],
                  ] as const
                ).flatMap(([report, summaries]) =>
                  (["primary", "independent"] as const).map((critic) => {
                    const rec = summaries[critic];
                    return (
                      <tr key={`${report}-${critic}`} aria-selected={critic === props.critic}>
                        <td className="mono">{report}</td>
                        <td>{critic}</td>
                        {SUMMARY_KEYS.map(([key, label]) => (
                          <td key={label} className="num">
                            {rec && key in rec
                              ? key === "cells"
                                ? String(rec[key])
                                : fmt(rec[key])
                              : "—"}
                          </td>
                        ))}
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
          <details className="chan-details">
            <summary>Per-row table ({rows.length} rows; select via keyboard)</summary>
            <div className="chan-table-wrap chan-scroll-y">
              <table className="data-table" tabIndex={0} aria-label="Twin critic interpretation summary">
                <thead>
                  <tr>
                    <th scope="col">row</th>
                    <th scope="col" className="num">primary dMSE</th>
                    <th scope="col" className="num">independent dMSE</th>
                    <th scope="col" className="num">primary cos</th>
                    <th scope="col" className="num">independent cos</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.row_id}
                      className="selectable"
                      aria-selected={r.row_id === props.selectedRowId}
                    >
                      <td>
                        <button type="button" className="chan-audit-link" onClick={() => select(r.row_id)}>
                          {shortRowId(r.row_id)}
                        </button>
                      </td>
                      <td className="num">{fmt(r.primary_dmse)}</td>
                      <td className="num">{fmt(r.independent_dmse)}</td>
                      <td className="num">{fmt(r.primary_cosine)}</td>
                      <td className="num">{fmt(r.independent_cosine)}</td>
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
