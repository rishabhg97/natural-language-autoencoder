/**
 * Horizontal bar readout of adjacent-description similarity — one row per
 * transition between consecutive sampled positions. Long runs of high
 * similarity read as persistence of the encoding, not of a belief.
 */
import { fmt, linearScale } from "../../data/format";
import { HBar } from "../../components/charts";
import type { TracePosition } from "../../data/types";

export interface Transition {
  from: TracePosition;
  to: TracePosition;
  similarity: number;
}

const W = 640;
const ROW_H = 10;
const TOP = 24;
const LEFT = 84;
const RIGHT = 44;
const GRID_TICKS = [0, 0.25, 0.5, 0.75, 1];

export default function PersistenceChart(props: {
  transitions: Transition[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}) {
  const { transitions, selectedIndex } = props;
  const height = TOP + transitions.length * ROW_H + 8;
  const x = linearScale([0, 1], [LEFT, W - RIGHT]);
  return (
    <svg
      className="chart-svg trc-persistence"
      viewBox={`0 0 ${W} ${height}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`Word-level Jaccard similarity of adjacent learned descriptions across ${transitions.length} transitions; long runs of high similarity read as persistence of the encoding. Exact values are in the transition table below.`}
    >
      {GRID_TICKS.map((t) => (
        <g key={t} aria-hidden>
          <line className="grid-line" x1={x(t)} x2={x(t)} y1={TOP - 6} y2={height - 4} />
          <text className="tick-label" x={x(t)} y={TOP - 10} textAnchor="middle">
            {fmt(t, 2)}
          </text>
        </g>
      ))}
      <text className="tick-label" x={W - RIGHT} y={TOP - 10} textAnchor="start" aria-hidden>
        {" "}Jaccard
      </text>
      {transitions.map((t, i) => {
        const y = TOP + i * ROW_H;
        const isSel = i === selectedIndex;
        return (
          <g key={t.to.position}>
            {isSel ? (
              <rect x={0} y={y - 0.5} width={W} height={ROW_H} fill="var(--selected-wash)" />
            ) : null}
            {i % 5 === 0 || isSel ? (
              <text className="tick-label" x={LEFT - 6} y={y + ROW_H - 2.5} textAnchor="end">
                {t.from.position}→{t.to.position}
              </text>
            ) : null}
            <HBar
              x0={x(0)}
              x1={x(t.similarity)}
              y={y + 1.5}
              height={7}
              fill={isSel ? "var(--selected)" : "var(--series-1)"}
              opacity={isSel ? 1 : 0.55}
            />
            {isSel ? (
              <text
                className="tick-label"
                x={Math.min(x(t.similarity) + 4, W - 6)}
                y={y + ROW_H - 2.5}
              >
                {fmt(t.similarity, 2)}
              </text>
            ) : null}
            <rect
              x={0}
              y={y - 0.5}
              width={W}
              height={ROW_H}
              fill="transparent"
              style={{ cursor: "pointer" }}
              onClick={() => props.onSelect(i)}
            >
              <title>{`p${t.from.position} → p${t.to.position} · similarity ${fmt(t.similarity, 3)} — click for word diff`}</title>
            </rect>
          </g>
        );
      })}
    </svg>
  );
}
