/**
 * CAPACITY LADDER — retrieval accuracy vs gallery size, with the mandatory
 * assumptions card rendered verbatim and the top confusion pairs.
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import type { ChannelShard } from "../../data/types";
import { Badge, Panel } from "../../components/ui";
import { Axes, Dot, linePath, makeFrame, useTooltip } from "../../components/charts";
import { extent, fmt, fmtPct } from "../../data/format";
import { AuditLink, LegendItem, padDomain, shortFamily } from "./lib";

const W = 640;
const H = 260;

export default function CapacityPanel(props: {
  channel: ChannelShard;
  update: (patch: Partial<AppState>) => void;
}) {
  const { capacity_ladder } = props.channel;
  const tooltip = useTooltip();
  const ladder = capacity_ladder.ladder;

  const frame = useMemo(() => {
    const xs = ladder.map((r) => r.gallery_bits);
    const ys = ladder.flatMap((r) => [r.top1_accuracy, r.top5_accuracy]);
    const [yLo] = extent(ys);
    return makeFrame({
      width: W,
      height: H,
      xDomain: padDomain(extent(xs), 0.04),
      yDomain: [Math.max(0, yLo - (1 - yLo) * 0.35 - 0.005), 1.0005],
      margin: { left: 68, bottom: 34 },
    });
  }, [ladder]);

  return (
    <Panel
      id="chan-capacity"
      title="Capacity ladder"
      span={7}
      badges={
        <>
          <Badge status="qualified" label="stored-snapshot" />
          <AuditLink claim="stored_snapshot_channel" update={props.update} />
        </>
      }
      sub={
        <>
          Fano-style lower bound, not channel capacity (assumptions verbatim on the right).
          Distance <span className="mono">{capacity_ladder.distance}</span>, variant{" "}
          <span className="mono">{capacity_ladder.variant}</span>, report{" "}
          <span className="mono">{capacity_ladder.source_report}</span>.
        </>
      }
    >
      {ladder.length === 0 ? (
        <div className="state-box unavailable">
          <Badge status="unavailable" /> no capacity-ladder rungs in channel.json.
        </div>
      ) : (
        <div className="chan-cap-grid">
          <div style={{ minWidth: 0 }}>
            <div className="chan-legend" role="group" aria-label="Capacity ladder legend">
              <LegendItem swatch="var(--series-1)" line>
                top-1 accuracy
              </LegendItem>
              <LegendItem swatch="var(--series-2)" line>
                top-5 accuracy
              </LegendItem>
            </div>
            <svg
              className="chan-chart"
              viewBox={`0 0 ${W} ${H}`}
              preserveAspectRatio="xMidYMid meet"
              role="img"
              aria-label={`Retrieval accuracy vs gallery bits. ${ladder
                .map(
                  (r) =>
                    `${r.gallery_bits} bits (gallery ${r.gallery_size}): top-1 ${fmtPct(r.top1_accuracy)}, top-5 ${fmtPct(r.top5_accuracy)}`,
                )
                .join("; ")}.`}
            >
              <Axes
                frame={frame}
                xLabel="gallery size (rows, log₂-spaced)"
                yLabel="accuracy"
                xTicks={ladder.map((r) => r.gallery_bits)}
                xFormat={(v) => String(2 ** Math.round(v))}
                yFormat={(v) => fmtPct(v, 1)}
              />
              <path
                d={linePath(frame, ladder.map((r) => ({ x: r.gallery_bits, y: r.top1_accuracy })))}
                fill="none"
                stroke="var(--series-1)"
                strokeWidth={2}
                aria-hidden
              />
              <path
                d={linePath(frame, ladder.map((r) => ({ x: r.gallery_bits, y: r.top5_accuracy })))}
                fill="none"
                stroke="var(--series-2)"
                strokeWidth={2}
                aria-hidden
              />
              {ladder.map((r) => {
                const tip = (e: React.PointerEvent) =>
                  tooltip.show({
                    x: e.clientX,
                    y: e.clientY,
                    content: (
                      <div>
                        <div className="tt-value">
                          gallery {r.gallery_size} ({fmt(r.gallery_bits, 0)} bits)
                        </div>
                        <div className="tt-label">top-1 {fmtPct(r.top1_accuracy)} · top-5 {fmtPct(r.top5_accuracy)}</div>
                        <div className="tt-label">median rank {fmt(r.median_rank, 1)} · MRR {fmt(r.mean_reciprocal_rank)}</div>
                        <div className="tt-label">Fano lower bound {fmt(r.fano_information_lower_bound_bits)} bits</div>
                      </div>
                    ),
                  });
                return (
                  <g key={r.gallery_bits}>
                    <Dot
                      cx={frame.x(r.gallery_bits)}
                      cy={frame.y(r.top1_accuracy)}
                      fill="var(--series-1)"
                      onHover={tip}
                      onLeave={tooltip.hide}
                    />
                    <Dot
                      cx={frame.x(r.gallery_bits)}
                      cy={frame.y(r.top5_accuracy)}
                      fill="var(--series-2)"
                      onHover={tip}
                      onLeave={tooltip.hide}
                    />
                  </g>
                );
              })}
              {/* Selective labels: first and last top-1 points only. */}
              {[ladder[0], ladder[ladder.length - 1]].map((r, i) => (
                <text
                  key={`lbl-${i}`}
                  x={frame.x(r.gallery_bits) + (i === 0 ? 8 : -8)}
                  y={frame.y(r.top1_accuracy) + 14}
                  textAnchor={i === 0 ? "start" : "end"}
                  fill="var(--ink-2)"
                  fontSize={10}
                  aria-hidden
                >
                  {fmtPct(r.top1_accuracy, 1)}
                </text>
              ))}
            </svg>
            <p className="chan-note">
              The y-axis is truncated to resolve the drop: top-1 falls from{" "}
              {fmtPct(ladder[0].top1_accuracy, 1)} at gallery {ladder[0].gallery_size} to{" "}
              {fmtPct(ladder[ladder.length - 1].top1_accuracy, 1)} at gallery{" "}
              {ladder[ladder.length - 1].gallery_size} — a{" "}
              {fmt((ladder[0].top1_accuracy - ladder[ladder.length - 1].top1_accuracy) * 100, 1)}-point
              absolute change.
            </p>
            <details className="chan-details">
              <summary>Ladder table (all rungs)</summary>
              <div className="chan-table-wrap">
                <table className="data-table" tabIndex={0}>
                  <thead>
                    <tr>
                      <th scope="col" className="num">gallery</th>
                      <th scope="col" className="num">bits</th>
                      <th scope="col" className="num">top-1</th>
                      <th scope="col" className="num">top-5</th>
                      <th scope="col" className="num">median rank</th>
                      <th scope="col" className="num">MRR</th>
                      <th scope="col" className="num">Fano LB (bits)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ladder.map((r) => (
                      <tr key={r.gallery_bits}>
                        <td className="num">{r.gallery_size}</td>
                        <td className="num">{fmt(r.gallery_bits, 0)}</td>
                        <td className="num">{fmtPct(r.top1_accuracy)}</td>
                        <td className="num">{fmtPct(r.top5_accuracy)}</td>
                        <td className="num">{fmt(r.median_rank, 1)}</td>
                        <td className="num">{fmt(r.mean_reciprocal_rank, 4)}</td>
                        <td className="num">{fmt(r.fano_information_lower_bound_bits)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
          <div>
            <div className="chan-aside-card">
              <h4>Assumptions (verbatim from shard)</h4>
              <dl>
                {Object.entries(capacity_ladder.assumptions).map(([k, v]) => (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd>{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <div className="chan-aside-card" style={{ marginTop: 8 }}>
              <h4>Top confusions</h4>
              <p className="chan-note" style={{ marginTop: 2 }}>
                Content-family ids (provenance keys, not semantic labels); hover for the full id.
              </p>
              {capacity_ladder.top_confusions.length === 0 ? (
                <div className="state-box unavailable">
                  <Badge status="unavailable" /> none recorded in shard.
                </div>
              ) : (
                <div className="chan-table-wrap">
                  <table className="data-table" tabIndex={0}>
                    <thead>
                      <tr>
                        <th scope="col">source fam</th>
                        <th scope="col">retrieved fam</th>
                        <th scope="col" className="num">n</th>
                      </tr>
                    </thead>
                    <tbody>
                      {capacity_ladder.top_confusions.slice(0, 5).map((c, i) => (
                        <tr key={`${c.source_family}-${c.retrieved_family}-${i}`}>
                          <td className="mono" title={c.source_family}>
                            {shortFamily(c.source_family)}
                          </td>
                          <td className="mono" title={c.retrieved_family}>
                            {shortFamily(c.retrieved_family)}
                          </td>
                          <td className="num">{c.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {capacity_ladder.top_confusions.length > 5 ? (
                    <details className="chan-details chan-confusion-details">
                      <summary>
                        Browse {capacity_ladder.top_confusions.length - 5} more confusion pairs
                      </summary>
                      <table className="data-table" tabIndex={0}>
                        <tbody>
                          {capacity_ladder.top_confusions.slice(5).map((c, i) => (
                            <tr key={`${c.source_family}-${c.retrieved_family}-more-${i}`}>
                              <td className="mono" title={c.source_family}>
                                {shortFamily(c.source_family)}
                              </td>
                              <td className="mono" title={c.retrieved_family}>
                                {shortFamily(c.retrieved_family)}
                              </td>
                              <td className="num">{c.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </details>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
