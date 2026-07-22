/**
 * ALTERNATE-TELLINGS FAN — k sampled re-tellings of the same stored activation:
 * strip plot of per-sample dMSE against the row's identity dMSE and the family
 * aggregate mean, plus the ranked texts.
 */

import { useMemo } from "react";
import type { ChannelShard, Critic } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { Dot, useTooltip } from "../../components/charts";
import { extent, fmt, linearScale, preview, shortRowId, ticks } from "../../data/format";
import { LegendItem, padDomain } from "./lib";

const W = 640;
const H = 132;
const LEFT = 16;
const RIGHT = 16;
const BASE = H - 38;
const CENTER_Y = 52;

export default function TellingsPanel(props: {
  channel: ChannelShard;
  critic: Critic;
  selectedRowId: string;
}) {
  const tooltip = useTooltip();
  const samples = props.channel.tellings[props.selectedRowId];
  const identity = useMemo(
    () =>
      props.channel.identity.find(
        (i) => i.row_id === props.selectedRowId && i.critic === props.critic,
      ),
    [props.channel.identity, props.selectedRowId, props.critic],
  );
  const aggKey = `${props.critic}.alternate_telling.directional_mse`;
  const aggregate = props.channel.aggregates[aggKey];

  const ranked = useMemo(
    () => (samples ? [...samples].sort((a, b) => a.dmse - b.dmse) : []),
    [samples],
  );
  const rowMean = useMemo(
    () =>
      samples && samples.length > 0
        ? samples.reduce((sum, s) => sum + s.dmse, 0) / samples.length
        : null,
    [samples],
  );

  const domain = useMemo(() => {
    if (!samples || samples.length === 0) return null;
    const values = samples.map((s) => s.dmse);
    if (identity) values.push(identity.dmse);
    if (aggregate) values.push(aggregate.mean);
    if (rowMean !== null) values.push(rowMean);
    return padDomain(extent(values), 0.1);
  }, [samples, identity, aggregate, rowMean]);

  const x = domain ? linearScale(domain, [LEFT, W - RIGHT]) : null;

  return (
    <Panel
      id="chan-tellings"
      title={`Alternate-tellings fan — ${shortRowId(props.selectedRowId)}`}
      span={6}
      badges={<Badge status="qualified" label="stored-snapshot" />}
      sub={
        samples
          ? `sampling-noise vs channel ceiling: k=${samples.length} alternate tellings of the same stored activation.`
          : "sampling-noise vs channel ceiling: alternate tellings of the same stored activation."
      }
    >
      {!samples || samples.length === 0 || !domain || !x ? (
        <UnavailableBox>
          No alternate tellings for row {shortRowId(props.selectedRowId)} in the shard.
        </UnavailableBox>
      ) : (
        <>
          <div className="chan-legend" role="group" aria-label="Tellings strip legend">
            <LegendItem swatch="var(--series-1)">sampled telling (dMSE)</LegendItem>
            <LegendItem swatch="var(--ink-2)" line>
              identity dMSE ({props.critic} critic)
            </LegendItem>
            <LegendItem swatch="var(--series-1)" line>
              this row&apos;s mean of {samples.length} tellings
            </LegendItem>
            <LegendItem swatch="var(--series-3)" line>
              all-rows mean (alternate tellings, whole panel)
            </LegendItem>
          </div>
          <svg
            className="chan-chart"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Strip plot of dMSE for ${samples.length} alternate tellings, from ${fmt(ranked[0].dmse)} to ${fmt(ranked[ranked.length - 1].dmse)}. Ranked values are listed below.`}
          >
            <g aria-hidden>
              {ticks(domain, 5).map((t) => (
                <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={10} y2={BASE} stroke="var(--grid)" strokeWidth={1} />
                  <text x={x(t)} y={BASE + 14} textAnchor="middle" fill="var(--ink-3)" fontSize={10.5}>
                    {fmt(t, 2)}
                  </text>
                </g>
              ))}
              <line x1={LEFT} x2={W - RIGHT} y1={BASE} y2={BASE} stroke="var(--axis)" strokeWidth={1} />
              <text x={(LEFT + W - RIGHT) / 2} y={H - 2} textAnchor="middle" fill="var(--ink-3)" fontSize={10.5}>
                dMSE
              </text>
            </g>
            {identity ? (
              <g aria-hidden>
                <line
                  x1={x(identity.dmse)}
                  x2={x(identity.dmse)}
                  y1={12}
                  y2={BASE}
                  stroke="var(--ink-2)"
                  strokeWidth={1.5}
                />
                <text x={x(identity.dmse)} y={10} textAnchor="middle" fill="var(--ink-2)" fontSize={10}>
                  identity {fmt(identity.dmse)}
                </text>
              </g>
            ) : null}
            {rowMean !== null ? (
              <g aria-hidden>
                <line
                  x1={x(rowMean)}
                  x2={x(rowMean)}
                  y1={30}
                  y2={BASE}
                  stroke="var(--series-1)"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
                <text x={x(rowMean)} y={28} textAnchor="middle" fill="var(--ink-2)" fontSize={10}>
                  row mean {fmt(rowMean)}
                </text>
              </g>
            ) : null}
            {aggregate ? (
              <g aria-hidden>
                <line
                  x1={x(aggregate.mean)}
                  x2={x(aggregate.mean)}
                  y1={22}
                  y2={BASE}
                  stroke="var(--series-3)"
                  strokeWidth={1.5}
                />
                <text x={x(aggregate.mean)} y={20} textAnchor="middle" fill="var(--ink-2)" fontSize={10}>
                  all-rows mean {fmt(aggregate.mean)}
                </text>
              </g>
            ) : null}
            {samples.map((s, i) => (
              <Dot
                key={s.cell_id}
                cx={x(s.dmse)}
                cy={CENTER_Y + ((i % 3) - 1) * 10}
                fill="var(--series-1)"
                onHover={(e) =>
                  tooltip.show({
                    x: e.clientX,
                    y: e.clientY,
                    content: (
                      <div>
                        <div className="tt-value">sample {s.sample_index}</div>
                        <div className="tt-label">dMSE {fmt(s.dmse)} · cosine {fmt(s.cosine)}</div>
                        <div className="tt-label">{preview(s.text, 140)}</div>
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
              No identity metric for the {props.critic} critic on this row — identity tick omitted.
            </UnavailableBox>
          ) : null}
          {!aggregate ? (
            <UnavailableBox>
              Aggregate <span className="mono">{aggKey}</span> is not in the shard — the all-rows
              mean tick is omitted for the {props.critic} critic.
            </UnavailableBox>
          ) : null}
          <ol className="chan-tellings-list" aria-label="Alternate tellings ranked by dMSE (best first)">
            {ranked.map((s) => (
              <li key={s.cell_id}>
                <details>
                  <summary>
                    <span className="chan-tellings-metric">
                      dMSE {fmt(s.dmse)} · cos {fmt(s.cosine)}
                    </span>
                    {preview(s.text, 96)}
                  </summary>
                  <div className="text-block">{s.text}</div>
                </details>
              </li>
            ))}
          </ol>
        </>
      )}
    </Panel>
  );
}
