/**
 * Real-vs-shuffled planning-onset curve for one poetry case. The shuffled
 * lane is a control and always renders; the onset gate value comes from the
 * shard, never a literal.
 */
import { useMemo, type PointerEvent as ReactPointerEvent } from "react";
import type { PoetryCase, PoetryPositionScore } from "../../data/types";
import { Axes, Dot, linePath, makeFrame, useTooltip } from "../../components/charts";
import { fmt, fmtPct } from "../../data/format";
import { UnavailableBox } from "../../components/ui";

const W = 660;
const H = 250;

export default function OnsetCurve(props: {
  kase: PoetryCase;
  gate: number;
  selectedOffset: number;
  onSelectOffset: (offset: number) => void;
}) {
  const { kase, gate } = props;
  const tooltip = useTooltip();

  const { real, shuffled, offsets } = useMemo(() => {
    const byOffset = (a: PoetryPositionScore, b: PoetryPositionScore) =>
      a.relative_offset - b.relative_offset;
    const real = kase.position_scores.filter((s) => s.variant === "real").slice().sort(byOffset);
    const shuffled = kase.position_scores
      .filter((s) => s.variant === "shuffled")
      .slice()
      .sort(byOffset);
    const offsets = [...new Set(kase.position_scores.map((s) => s.relative_offset))].sort(
      (a, b) => a - b,
    );
    return { real, shuffled, offsets };
  }, [kase]);

  if (offsets.length === 0) {
    return (
      <UnavailableBox>
        no position scores for this case — the onset curve cannot be drawn.
      </UnavailableBox>
    );
  }

  const xDomain: [number, number] = [Math.min(...offsets), Math.max(...offsets)];
  const frame = makeFrame({
    width: W,
    height: H,
    xDomain,
    yDomain: [0, 1],
    margin: { bottom: 42 },
  });
  const onset = kase.planning_onset_relative_offset;
  const step = frame.innerWidth / Math.max(1, offsets.length - 1);
  const samplesPerPoint = Math.max(0, ...real.map((s) => s.samples));

  const showTip = (e: ReactPointerEvent, s: PoetryPositionScore) => {
    tooltip.show({
      x: e.clientX,
      y: e.clientY,
      content: (
        <div>
          <div className="tt-value">
            {s.variant} · offset {s.relative_offset}
          </div>
          <div className="tt-label">target-family rate {fmtPct(s.target_family_rate)}</div>
          <div className="tt-label">target-exact rate {fmtPct(s.target_exact_rate)}</div>
          <div className="tt-label">alternate-family rate {fmtPct(s.alternate_family_rate)}</div>
          <div className="tt-label">
            usable {fmtPct(s.usable_rate)} · {s.samples} samples
          </div>
        </div>
      ),
    });
  };

  return (
    <>
      <div className="trc-legend">
        <span>
          <span className="trc-swatch" style={{ background: "var(--series-1)" }} /> real prefix
        </span>
        <span>
          <span className="trc-swatch trc-swatch-hollow" aria-hidden /> shuffled-prefix control
          (hollow, dashed — visible even where it ties the real line)
        </span>
        <span>
          <span className="trc-swatch trc-swatch-line" /> onset gate {fmt(gate, 2)}
        </span>
      </div>
      {samplesPerPoint > 0 ? (
        <p className="trc-note">
          Each point is the share of {samplesPerPoint} AV samples at that offset — rates are
          quantized to steps of {fmt(1 / samplesPerPoint, 2)}; the {fmt(gate, 2)} gate means at
          least {Math.max(1, Math.ceil(gate * samplesPerPoint))} sample(s).
        </p>
      ) : null}
      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Planning-onset curve for case ${kase.case_id}: target-family rate by relative offset, real versus shuffled control, onset gate at ${fmt(gate, 2)}. Exact values are in the position score table below.`}
      >
        <Axes
          frame={frame}
          xLabel="relative offset (tokens before the anchor)"
          yLabel="target-family rate"
          xTicks={offsets.filter((o) => o % 2 === 0)}
          yTicks={[0, 0.25, 0.5, 0.75, 1]}
        />
        {/* Selected-position crosshair, synced with the inline source token. */}
        <line
          x1={frame.x(props.selectedOffset)}
          x2={frame.x(props.selectedOffset)}
          y1={frame.y(1)}
          y2={frame.y(0)}
          stroke="var(--selected)"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.8}
        />
        {/* onset gate (value from the shard) */}
        <line
          x1={frame.x(xDomain[0])}
          x2={frame.x(xDomain[1])}
          y1={frame.y(gate)}
          y2={frame.y(gate)}
          stroke="var(--ink-3)"
          strokeWidth={1}
          strokeDasharray="5 3"
        />
        <text className="tick-label" x={frame.x(xDomain[1])} y={frame.y(gate) - 4} textAnchor="end">
          onset gate {fmt(gate, 2)}
        </text>
        {onset !== null ? (
          <>
            <line
              x1={frame.x(onset)}
              x2={frame.x(onset)}
              y1={frame.y(1)}
              y2={frame.y(0)}
              stroke="var(--axis)"
              strokeWidth={1}
            />
            <text className="tick-label" x={frame.x(onset) + 4} y={frame.y(1) + 10}>
              onset {onset}
            </text>
          </>
        ) : null}
        {/* click columns select the analysis position (rendered under the marks) */}
        {offsets.map((o) => {
          const x0 = Math.max(frame.margin.left, frame.x(o) - step / 2);
          const x1 = Math.min(frame.margin.left + frame.innerWidth, frame.x(o) + step / 2);
          return (
            <rect
              key={o}
              x={x0}
              y={frame.margin.top}
              width={Math.max(1, x1 - x0)}
              height={frame.innerHeight}
              fill="transparent"
              style={{ cursor: "pointer" }}
              onClick={() => props.onSelectOffset(o)}
            >
              <title>{`select offset ${o}`}</title>
            </rect>
          );
        })}
        <path
          d={linePath(
            frame,
            shuffled.map((s) => ({ x: s.relative_offset, y: s.target_family_rate })),
          )}
          fill="none"
          stroke="var(--series-4)"
          strokeWidth={2}
          strokeDasharray="5 4"
        />
        <path
          d={linePath(
            frame,
            real.map((s) => ({ x: s.relative_offset, y: s.target_family_rate })),
          )}
          fill="none"
          stroke="var(--series-1)"
          strokeWidth={2}
        />
        {shuffled.map((s) => (
          <g
            key={`s${s.relative_offset}`}
            onClick={() => props.onSelectOffset(s.relative_offset)}
            style={{ cursor: "pointer" }}
          >
            {/* Hollow control marker: stays visible where it coincides with
                the real series (equal values would otherwise be occluded). */}
            <circle
              cx={frame.x(s.relative_offset)}
              cy={frame.y(s.target_family_rate)}
              r={5.5}
              fill="var(--surface-1)"
              stroke="var(--series-4)"
              strokeWidth={2}
              onPointerMove={(e) => showTip(e, s)}
              onPointerLeave={tooltip.hide}
            />
          </g>
        ))}
        {real.map((s) => (
          <g
            key={`r${s.relative_offset}`}
            onClick={() => props.onSelectOffset(s.relative_offset)}
            style={{ cursor: "pointer" }}
          >
            <Dot
              cx={frame.x(s.relative_offset)}
              cy={frame.y(s.target_family_rate)}
              fill="var(--series-1)"
              onHover={(e) => showTip(e, s)}
              onLeave={tooltip.hide}
            />
          </g>
        ))}
      </svg>
      {shuffled.length === 0 ? (
        <UnavailableBox>the shuffled control has no scored positions for this case.</UnavailableBox>
      ) : null}
      <details className="trc-details">
        <summary>position score table</summary>
        <div className="trc-table-scroll">
          <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th className="num">offset</th>
                <th className="num">real target-family</th>
                <th className="num">shuffled target-family</th>
                <th className="num">real usable</th>
                <th className="num">real samples</th>
              </tr>
            </thead>
            <tbody>
              {offsets.map((o) => {
                const r = real.find((s) => s.relative_offset === o);
                const sh = shuffled.find((s) => s.relative_offset === o);
                return (
                  <tr key={o}>
                    <td className="num">{o}</td>
                    <td className="num">{r ? fmtPct(r.target_family_rate) : "—"}</td>
                    <td className="num">{sh ? fmtPct(sh.target_family_rate) : "—"}</td>
                    <td className="num">{r ? fmtPct(r.usable_rate) : "—"}</td>
                    <td className="num">{r ? r.samples : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}
