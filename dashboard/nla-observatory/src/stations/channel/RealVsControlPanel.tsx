/**
 * REAL-VS-CONTROL AV LOSS — AV-conditioned AR loss for the learned description
 * (real) vs control texts; e1_av 8-row canary and e2 512-row paired evidence.
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import type { ChannelShard } from "../../data/types";
import { Badge, Panel } from "../../components/ui";
import { useTooltip } from "../../components/charts";
import { fmt, fmtPct, fmtSigned, linearScale, ticks } from "../../data/format";
import { AuditLink, LegendItem } from "./lib";

const W = 620;
const LEFT = 84;
const RIGHT = 54;
const GROUP_H = 36;
const BAR_H = 10;
const TOP = 6;
const BOTTOM = 44;

function orderVariants(keys: string[]): string[] {
  const rest = keys.filter((k) => k !== "real").sort();
  return keys.includes("real") ? ["real", ...rest] : rest;
}

export default function RealVsControlPanel(props: {
  channel: ChannelShard;
  update: (patch: Partial<AppState>) => void;
}) {
  const { e1_av, e2 } = props.channel.real_vs_control;
  const tooltip = useTooltip();

  const variants = useMemo(
    () => orderVariants(Array.from(new Set([...Object.keys(e2.mean_loss), ...Object.keys(e1_av.losses)]))),
    [e1_av, e2],
  );
  const xMax = useMemo(
    () =>
      Math.max(
        ...variants.map((v) => Math.max(e2.mean_loss[v] ?? 0, e1_av.losses[v] ?? 0)),
        0,
      ),
    [variants, e1_av, e2],
  );
  const x = linearScale([0, xMax], [LEFT, W - RIGHT]);
  const xTicks = ticks([0, xMax], 4);
  const height = TOP + variants.length * GROUP_H + BOTTOM;

  const pairedEntries = Object.entries(e2.paired);
  const minWin = pairedEntries.length
    ? Math.min(...pairedEntries.map(([, p]) => p.real_win_fraction))
    : NaN;

  const bar = (x1: number, y: number) => {
    const w = Math.max(1, x1 - LEFT);
    const r = Math.min(4, BAR_H / 2, w / 2);
    return `M${LEFT},${y} h${Math.max(0, w - r)} a${r},${r} 0 0 1 ${r},${r} v${
      BAR_H - 2 * r
    } a${r},${r} 0 0 1 ${-r},${r} h${-Math.max(0, w - r)} z`;
  };

  return (
    <Panel
      id="chan-real-vs-control"
      title="Real-vs-control AV loss"
      span={5}
      badges={
        <>
          <Badge
            status="qualified"
            label="stored-snapshot"
            title={`Qualified validation-split evidence: ${e2.source_report} is the ${e2.rows}-row paired comparison`}
          />
          <AuditLink claim="null_text" update={props.update} />
        </>
      }
      sub={
        <>
          AV token loss — the AV model&apos;s autoregressive negative log-likelihood of the
          explanation text — under real vs control activation conditioning. Canary{" "}
          <span className="mono">{e1_av.source_report}</span> ({e1_av.rows} rows; a small
          smoke-test subset) and <span className="mono">{e2.source_report}</span> ({e2.rows} rows,{" "}
          {e2.records} records).
        </>
      }
    >
      {variants.length === 0 ? (
        <div className="state-box unavailable">
          <Badge status="unavailable" /> no real-vs-control losses in channel.json.
        </div>
      ) : (
        <>
          <div className="chan-legend" role="group" aria-label="Loss series legend">
            <LegendItem swatch="var(--series-1)">e2 mean loss ({e2.rows} rows)</LegendItem>
            <LegendItem swatch="var(--series-2)">e1_av loss ({e1_av.rows}-row canary)</LegendItem>
          </div>
          <svg
            className="chan-chart"
            viewBox={`0 0 ${W} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Grouped bars of AV token loss per conditioning variant. ${variants
              .map((v) => `${v}: e2 ${fmt(e2.mean_loss[v] ?? NaN)}, e1 ${fmt(e1_av.losses[v] ?? NaN)}`)
              .join("; ")}.`}
          >
            <g aria-hidden>
              {xTicks.map((t) => (
                <g key={t}>
                  <line
                    x1={x(t)}
                    x2={x(t)}
                    y1={TOP}
                    y2={TOP + variants.length * GROUP_H}
                    stroke="var(--grid)"
                    strokeWidth={1}
                  />
                  <text
                    x={x(t)}
                    y={TOP + variants.length * GROUP_H + 14}
                    textAnchor="middle"
                    fill="var(--ink-3)"
                    fontSize={10.5}
                  >
                    {fmt(t, 2)}
                  </text>
                </g>
              ))}
              <text
                x={(LEFT + W - RIGHT) / 2}
                y={height - 4}
                textAnchor="middle"
                fill="var(--ink-3)"
                fontSize={10.5}
              >
                mean AV token loss (lower is better)
              </text>
            </g>
            {variants.map((v, i) => {
              const yTop = TOP + i * GROUP_H;
              const e2v = e2.mean_loss[v];
              const e1v = e1_av.losses[v];
              return (
                <g key={v}>
                  {v === "real" ? (
                    <rect
                      x={0}
                      y={yTop}
                      width={W}
                      height={GROUP_H - 4}
                      fill="var(--selected-wash)"
                      rx={4}
                    />
                  ) : null}
                  <text
                    x={LEFT - 8}
                    y={yTop + GROUP_H / 2 + 2}
                    textAnchor="end"
                    fill="var(--ink)"
                    fontSize={11.5}
                    fontWeight={v === "real" ? 650 : 450}
                  >
                    {v}
                  </text>
                  {e2v !== undefined ? (
                    <>
                      <path d={bar(x(e2v), yTop + 5)} fill="var(--series-1)" />
                      <text
                        x={x(e2v) + 4}
                        y={yTop + 5 + BAR_H - 1}
                        fill="var(--ink-2)"
                        fontSize={9.5}
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        {fmt(e2v)}
                      </text>
                    </>
                  ) : null}
                  {e1v !== undefined ? (
                    <>
                      <path d={bar(x(e1v), yTop + 5 + BAR_H + 3)} fill="var(--series-2)" />
                      <text
                        x={x(e1v) + 4}
                        y={yTop + 5 + 2 * BAR_H + 2}
                        fill="var(--ink-2)"
                        fontSize={9.5}
                        style={{ fontVariantNumeric: "tabular-nums" }}
                      >
                        {fmt(e1v)}
                      </text>
                    </>
                  ) : null}
                  <rect
                    x={0}
                    y={yTop}
                    width={W}
                    height={GROUP_H}
                    fill="transparent"
                    onPointerMove={(e) =>
                      tooltip.show({
                        x: e.clientX,
                        y: e.clientY,
                        content: (
                          <div>
                            <div className="tt-value">{v}</div>
                            <div className="tt-label">e2 mean loss {fmt(e2v ?? NaN)} · e1_av loss {fmt(e1v ?? NaN)}</div>
                            {v !== "real" && e2.paired[v] ? (
                              <div className="tt-label">
                                paired Δ(real−{v}) {fmtSigned(e2.paired[v].mean_real_minus_control)} · real wins{" "}
                                {fmtPct(e2.paired[v].real_win_fraction)}
                              </div>
                            ) : null}
                          </div>
                        ),
                      })
                    }
                    onPointerLeave={tooltip.hide}
                  />
                </g>
              );
            })}
          </svg>
          {Number.isFinite(minWin) ? (
            <p className="chan-note">
              Paired per-row check (e2): the real description beats each control on{" "}
              <strong>{fmtPct(minWin, 1)}</strong> of {e2.rows} rows (
              <span className="mono">real_win_fraction</span>, minimum across controls).
            </p>
          ) : (
            <div className="state-box unavailable">
              <Badge status="unavailable" /> no paired e2 comparison in the shard.
            </div>
          )}
          <div className="chan-table-wrap">
            <table className="data-table" tabIndex={0} aria-label="AV real and control loss comparison">
              <thead>
                <tr>
                  <th scope="col">variant</th>
                  <th scope="col" className="num">e1_av loss</th>
                  <th scope="col" className="num">e2 loss</th>
                  <th scope="col" className="num">Δ real−control</th>
                  <th scope="col" className="num">real wins</th>
                </tr>
              </thead>
              <tbody>
                {variants.map((v) => {
                  const p = e2.paired[v];
                  return (
                    <tr key={v}>
                      <td className="mono">{v}</td>
                      <td className="num">{fmt(e1_av.losses[v] ?? NaN)}</td>
                      <td className="num">{fmt(e2.mean_loss[v] ?? NaN)}</td>
                      <td className="num">{p ? fmtSigned(p.mean_real_minus_control) : "—"}</td>
                      <td className="num">{p ? fmtPct(p.real_win_fraction) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}
