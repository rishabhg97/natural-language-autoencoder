/**
 * INFORMATION WATERFALL — directional MSE ladder over ablation variants of
 * the learned description channel (report e0, 512-row qualified evidence).
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import type { ChannelShard } from "../../data/types";
import { Badge, Panel } from "../../components/ui";
import { useTooltip } from "../../components/charts";
import { fmt, linearScale, ticks } from "../../data/format";
import { AuditLink, LegendItem, VARIANT_MEANING, waterfallColor } from "./lib";

const W = 680;
const LEFT = 150;
const RIGHT = 64;
const ROW_H = 32;
const BAR_H = 16;
const TOP = 6;
const BOTTOM = 44;

export default function WaterfallPanel(props: {
  channel: ChannelShard;
  update: (patch: Partial<AppState>) => void;
}) {
  const { waterfall } = props.channel;
  const tooltip = useTooltip();

  const entries = useMemo(
    () => Object.entries(waterfall.variants).sort((a, b) => b[1].dmse - a[1].dmse),
    [waterfall],
  );
  const xMax = useMemo(
    () => Math.max(...entries.map(([, v]) => Math.max(v.dmse, v.ci_high)), 0),
    [entries],
  );
  const x = linearScale([0, xMax], [LEFT, W - RIGHT]);
  const xTicks = ticks([0, xMax], 5);
  const height = TOP + entries.length * ROW_H + BOTTOM;
  const realRows = waterfall.variants["av_real"]?.rows;
  // All variants usually share the same evidence size; hoist it to one caption.
  const uniformSize = useMemo(() => {
    const sizes = new Set(entries.map(([, v]) => `${v.rows}·${v.families}`));
    return sizes.size === 1 ? entries[0]?.[1] : null;
  }, [entries]);

  return (
    <Panel
      id="chan-waterfall"
      title="Information waterfall"
      span={7}
      badges={
        <>
          <Badge
            status="qualified"
            label="stored-snapshot"
            title="Qualified channel claim: directional recovery of the stored activation snapshot"
          />
          <AuditLink claim="stored_snapshot_channel" update={props.update} />
        </>
      }
      sub={
        <>
          dMSE = 2·(1−cosine); directional, not raw-magnitude.{" "}
          {realRows !== undefined ? `${realRows}-row qualified evidence` : "Qualified evidence"} from
          report <span className="mono">{waterfall.source_report}</span> ({waterfall.split} split).
          Whiskers span <span className="mono">ci_low…ci_high</span> from the shard.
        </>
      }
    >
      {entries.length === 0 ? (
        <div className="state-box unavailable">
          <Badge status="unavailable" /> channel.json carries no waterfall variants.
        </div>
      ) : (
        <>
          <div className="chan-legend" role="group" aria-label="Waterfall series legend">
            <LegendItem swatch="var(--series-1)">
              av_real — learned description of the stored activation
            </LegendItem>
            <LegendItem swatch="var(--series-3)">teacher — reference explanation (the key comparison)</LegendItem>
            <LegendItem swatch="var(--ink-3)">
              controls: av_zero · av_mean · av_shuffled · av_none
            </LegendItem>
            <LegendItem swatch="var(--series-4)">no-text baseline (mean)</LegendItem>
          </div>
          {uniformSize ? (
            <p className="chan-note">
              All variants: {uniformSize.rows} rows · {uniformSize.families} families
              (family-clustered CIs).
            </p>
          ) : null}
          <svg
            className="chan-chart"
            viewBox={`0 0 ${W} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Information waterfall: directional MSE per ablation variant, sorted worst to best. ${entries
              .map(([name, v]) => `${name} ${fmt(v.dmse)}`)
              .join(", ")}.`}
          >
            <g aria-hidden>
              {xTicks.map((t) => (
                <g key={t}>
                  <line
                    x1={x(t)}
                    x2={x(t)}
                    y1={TOP}
                    y2={TOP + entries.length * ROW_H}
                    stroke="var(--grid)"
                    strokeWidth={1}
                  />
                  <text
                    x={x(t)}
                    y={TOP + entries.length * ROW_H + 14}
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
                directional MSE (dMSE)
              </text>
            </g>
            {entries.map(([name, v], i) => {
              const yTop = TOP + i * ROW_H;
              const barY = yTop + (ROW_H - BAR_H) / 2;
              const cy = yTop + ROW_H / 2;
              const w = Math.max(1, x(v.dmse) - LEFT);
              const r = Math.min(4, BAR_H / 2, w / 2);
              return (
                <g key={name}>
                  <text
                    x={LEFT - 8}
                    y={uniformSize ? cy + 4 : cy - 3}
                    textAnchor="end"
                    fill="var(--ink)"
                    fontSize={11.5}
                    fontWeight={name === "av_real" || name === "teacher" ? 650 : 450}
                  >
                    {name}
                  </text>
                  {uniformSize ? null : (
                    <text x={LEFT - 8} y={cy + 10} textAnchor="end" fill="var(--ink-3)" fontSize={9.5}>
                      {v.rows} rows · {v.families} fam
                    </text>
                  )}
                  <path
                    d={`M${LEFT},${barY} h${Math.max(0, w - r)} a${r},${r} 0 0 1 ${r},${r} v${
                      BAR_H - 2 * r
                    } a${r},${r} 0 0 1 ${-r},${r} h${-Math.max(0, w - r)} z`}
                    fill={waterfallColor(name)}
                    opacity={name === "av_real" ? 1 : 0.75}
                  />
                  {/* CI whisker */}
                  <line
                    x1={x(v.ci_low)}
                    x2={x(v.ci_high)}
                    y1={cy}
                    y2={cy}
                    stroke="var(--ink)"
                    strokeWidth={1.25}
                  />
                  <line x1={x(v.ci_low)} x2={x(v.ci_low)} y1={cy - 4} y2={cy + 4} stroke="var(--ink)" strokeWidth={1.25} />
                  <line x1={x(v.ci_high)} x2={x(v.ci_high)} y1={cy - 4} y2={cy + 4} stroke="var(--ink)" strokeWidth={1.25} />
                  <text
                    x={W - 6}
                    y={cy + 4}
                    textAnchor="end"
                    fill="var(--ink)"
                    fontSize={11}
                    fontWeight={650}
                    style={{ fontVariantNumeric: "tabular-nums" }}
                  >
                    {fmt(v.dmse)}
                  </text>
                  {/* hover surface */}
                  <rect
                    x={0}
                    y={yTop}
                    width={W}
                    height={ROW_H}
                    fill="transparent"
                    onPointerMove={(e) =>
                      tooltip.show({
                        x: e.clientX,
                        y: e.clientY,
                        content: (
                          <div>
                            <div className="tt-value">{name}</div>
                            {VARIANT_MEANING[name] ? (
                              <div className="tt-label">{VARIANT_MEANING[name]}</div>
                            ) : null}
                            <div className="tt-label">dMSE {fmt(v.dmse)} · CI [{fmt(v.ci_low)}, {fmt(v.ci_high)}]</div>
                            <div className="tt-label">cosine_mean {fmt(v.cosine_mean)}</div>
                            <div className="tt-label">norm_ratio_mean {fmt(v.norm_ratio_mean)}</div>
                            <div className="tt-label">{v.rows} rows · {v.families} families</div>
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
          <dl className="chan-variant-key" aria-label="What each variant means">
            {entries.map(([name]) =>
              VARIANT_MEANING[name] ? (
                <div key={name}>
                  <dt className="mono">{name}</dt>
                  <dd>{VARIANT_MEANING[name]}</dd>
                </div>
              ) : null,
            )}
          </dl>
          <p className="chan-note">
            This bundle ships no source-text re-encoding floor (<span className="mono">source_raw</span>);
            the best text rung shown here is the teacher reference.
          </p>
          <details className="chan-details">
            <summary>Full variant table (dMSE, CI, cosine, norm ratio)</summary>
            <div className="chan-table-wrap">
              <table className="data-table" tabIndex={0}>
                <thead>
                  <tr>
                    <th scope="col">variant</th>
                    <th scope="col" className="num">dMSE</th>
                    <th scope="col" className="num">ci_low</th>
                    <th scope="col" className="num">ci_high</th>
                    <th scope="col" className="num">cosine_mean</th>
                    <th scope="col" className="num">norm_ratio_mean</th>
                    <th scope="col" className="num">rows</th>
                    <th scope="col" className="num">families</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(([name, v]) => (
                    <tr key={name}>
                      <td className="mono">{name}</td>
                      <td className="num">{fmt(v.dmse)}</td>
                      <td className="num">{fmt(v.ci_low)}</td>
                      <td className="num">{fmt(v.ci_high)}</td>
                      <td className="num">{fmt(v.cosine_mean)}</td>
                      <td className="num">{fmt(v.norm_ratio_mean)}</td>
                      <td className="num">{v.rows}</td>
                      <td className="num">{v.families}</td>
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
