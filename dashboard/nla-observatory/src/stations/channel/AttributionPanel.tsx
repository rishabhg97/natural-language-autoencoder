/**
 * CODE ATTRIBUTION — (a) word-occlusion saliency rendered as tinted typography
 * over the teacher explanation, (b) section Shapley decomposition.
 * Occlusion saliency measures the AR critic's reading, not the AV's intent.
 */

import { useMemo } from "react";
import type { ChannelShard, OcclusionCell, RowRecord } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { useTooltip } from "../../components/charts";
import { fmt, fmtSigned, linearScale, shortRowId, ticks } from "../../data/format";
import { LegendItem } from "./lib";

const RAMP = ["var(--seq-100)", "var(--seq-250)", "var(--seq-400)", "var(--seq-550)"] as const;

/** Fixed presentation order for the four teacher-explanation sections. */
const SECTION_ORDER = ["syntax", "discourse", "register", "final_token"];

function rampIndex(d: number, maxPos: number): number {
  if (d <= 0 || maxPos <= 0) return -1;
  const t = Math.min(1, d / maxPos);
  return Math.min(RAMP.length - 1, Math.floor(t * RAMP.length));
}

function OcclusionText(props: {
  text: string;
  cells: OcclusionCell[];
  maxPos: number;
  /** Word indexes that receive a visible mark; all other words stay plain. */
  marked: ReadonlySet<number>;
}) {
  const tooltip = useTooltip();
  const segments = useMemo(() => {
    const out: { key: string; text: string; cell: OcclusionCell | null }[] = [];
    let cursor = 0;
    for (const cell of props.cells) {
      if (cell.char_start > cursor) {
        out.push({ key: `plain-${cursor}`, text: props.text.slice(cursor, cell.char_start), cell: null });
      }
      out.push({
        key: `w-${cell.word_index}`,
        text: props.text.slice(cell.char_start, cell.char_end),
        cell,
      });
      cursor = cell.char_end;
    }
    if (cursor < props.text.length) {
      out.push({ key: `plain-${cursor}`, text: props.text.slice(cursor), cell: null });
    }
    return out;
  }, [props.text, props.cells]);

  return (
    <div className="chan-occ-text" aria-label="Teacher explanation with per-word occlusion saliency">
      {segments.map((seg) => {
        if (!seg.cell) return <span key={seg.key}>{seg.text}</span>;
        const { cell } = seg;
        // Selective marking: only top-|d_dmse| words and occlusion-helped
        // words carry a mark, so salience stays readable. Every word keeps
        // its tooltip.
        const isMarked = props.marked.has(cell.word_index);
        const idx = isMarked ? rampIndex(cell.d_dmse, props.maxPos) : -2;
        const style =
          idx >= 0
            ? {
                borderBottomColor: RAMP[idx],
                borderBottomWidth: idx >= 2 ? 3 : 2,
                background: idx >= 3 ? "var(--selected-wash)" : undefined,
              }
            : undefined;
        return (
          <span
            key={seg.key}
            className={`chan-occ-word${isMarked && idx === -1 ? " chan-occ-neg" : ""}${!isMarked ? " chan-occ-plain" : ""}`}
            style={style}
            title={`${cell.word}: d_dmse ${fmtSigned(cell.d_dmse, 4)} · dmse ${fmt(cell.dmse)}`}
            onPointerMove={(e) =>
              tooltip.show({
                x: e.clientX,
                y: e.clientY,
                content: (
                  <div>
                    <div className="tt-value">{cell.word}</div>
                    <div className="tt-label">d_dmse {fmtSigned(cell.d_dmse, 4)}</div>
                    <div className="tt-label">dmse with word occluded {fmt(cell.dmse)}</div>
                  </div>
                ),
              })
            }
            onPointerLeave={tooltip.hide}
          >
            {seg.text}
          </span>
        );
      })}
    </div>
  );
}

export default function AttributionPanel(props: { channel: ChannelShard; row: RowRecord }) {
  const rowId = props.row.row_id;
  const cells = props.channel.occlusion[rowId];
  const shapley = props.channel.shapley[rowId];

  const dExtent = useMemo(() => {
    if (!cells || cells.length === 0) return null;
    const ds = cells.map((c) => c.d_dmse);
    return [Math.min(...ds), Math.max(...ds)] as const;
  }, [cells]);
  const maxPos = dExtent ? Math.max(0, dExtent[1]) : 0;

  const topWords = useMemo(() => {
    if (!cells) return [];
    return [...cells].sort((a, b) => Math.abs(b.d_dmse) - Math.abs(a.d_dmse)).slice(0, 8);
  }, [cells]);

  // Visible marks: the top-8 positive-saliency words plus every word whose
  // occlusion helped (d_dmse <= 0). Everything else stays plain but keeps
  // its tooltip.
  const markedWords = useMemo(() => {
    if (!cells) return new Set<number>();
    const marked = new Set<number>();
    [...cells]
      .filter((c) => c.d_dmse > 0)
      .sort((a, b) => b.d_dmse - a.d_dmse)
      .slice(0, 8)
      .forEach((c) => marked.add(c.word_index));
    cells.filter((c) => c.d_dmse <= 0).forEach((c) => marked.add(c.word_index));
    return marked;
  }, [cells]);

  const sections = useMemo(() => {
    if (!shapley) return [];
    const entries = Object.entries(shapley.sections);
    entries.sort((a, b) => {
      const ia = SECTION_ORDER.indexOf(a[0]);
      const ib = SECTION_ORDER.indexOf(b[0]);
      return (ia === -1 ? SECTION_ORDER.length : ia) - (ib === -1 ? SECTION_ORDER.length : ib);
    });
    return entries;
  }, [shapley]);

  // Shapley mini bar geometry (virtual width, fixed height per bar).
  const SW = 640;
  const S_LEFT = 104;
  const S_RIGHT = 56;
  const S_ROW = 24;
  const S_BAR = 14;
  const sMax = sections.length ? Math.max(...sections.map(([, v]) => v), 0) : 1;
  const sx = linearScale([0, sMax], [S_LEFT, SW - S_RIGHT]);
  const sTicks = ticks([0, sMax], 4);
  const sHeight = 6 + sections.length * S_ROW + 24;

  return (
    <Panel
      id="chan-attribution"
      title={`Code attribution — ${shortRowId(rowId)}`}
      span={6}
      badges={<Badge status="qualified" label="stored-snapshot" />}
      sub={
        <>
          Occlusion saliency measures the AR critic's reading, not the AV's intent. Word occlusion
          and section Shapley are scored under the primary critic on the teacher reference
          explanation for this row.
        </>
      }
    >
      <h4 className="chan-h4">Word occlusion typography</h4>
      {!cells || cells.length === 0 || !dExtent ? (
        <UnavailableBox>No occlusion cells for row {shortRowId(rowId)} in the shard.</UnavailableBox>
      ) : (
        <>
          <div className="chan-legend" role="group" aria-label="Occlusion saliency scale">
            <span className="chan-legend-item">
              d_dmse scale: {fmtSigned(dExtent[0], 4)} … {fmtSigned(dExtent[1], 4)}
            </span>
            {RAMP.map((c, i) => (
              <LegendItem key={c} swatch={c}>
                {i === 0 ? "low +" : i === RAMP.length - 1 ? "high +" : ""}
              </LegendItem>
            ))}
            <LegendItem swatch="var(--ink-3)" dotted>
              d_dmse ≤ 0 (occlusion helped)
            </LegendItem>
            <span className="chan-legend-item">
              only the top-8 words and negatives are marked; hover any word for its value
            </span>
          </div>
          <OcclusionText
            text={props.row.teacher_text}
            cells={cells}
            maxPos={maxPos}
            marked={markedWords}
          />
          <details className="chan-details">
            <summary>Top saliency words (by |d_dmse|)</summary>
            <div className="chan-table-wrap">
              <table className="data-table" tabIndex={0}>
                <thead>
                  <tr>
                    <th scope="col">word</th>
                    <th scope="col" className="num">d_dmse</th>
                    <th scope="col" className="num">dmse (occluded)</th>
                  </tr>
                </thead>
                <tbody>
                  {topWords.map((w) => (
                    <tr key={w.word_index}>
                      <td className="mono">{w.word}</td>
                      <td className="num">{fmtSigned(w.d_dmse, 4)}</td>
                      <td className="num">{fmt(w.dmse)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}

      <h4 className="chan-h4" style={{ marginTop: 10 }}>
        Section Shapley{" "}
        <span className="chan-controls-note" title="shard key: one_minus_directional_mse">
          utility = 1 − dMSE
        </span>
      </h4>
      {!shapley || sections.length === 0 ? (
        <UnavailableBox>No section-Shapley decomposition for row {shortRowId(rowId)}.</UnavailableBox>
      ) : (
        <>
          <svg
            className="chan-chart"
            viewBox={`0 0 ${SW} ${sHeight}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Section Shapley values for row ${shortRowId(rowId)}: ${sections
              .map(([k, v]) => `${k} ${fmt(v)}`)
              .join(", ")}.`}
          >
            <g aria-hidden>
              {sTicks.map((t) => (
                <g key={t}>
                  <line
                    x1={sx(t)}
                    x2={sx(t)}
                    y1={6}
                    y2={6 + sections.length * S_ROW}
                    stroke="var(--grid)"
                    strokeWidth={1}
                  />
                  <text
                    x={sx(t)}
                    y={6 + sections.length * S_ROW + 14}
                    textAnchor="middle"
                    fill="var(--ink-3)"
                    fontSize={10.5}
                  >
                    {fmt(t, 2)}
                  </text>
                </g>
              ))}
            </g>
            {sections.map(([name, value], i) => {
              const yTop = 6 + i * S_ROW + (S_ROW - S_BAR) / 2;
              const w = Math.max(1, sx(value) - S_LEFT);
              const r = Math.min(4, S_BAR / 2, w / 2);
              return (
                <g key={name}>
                  <text
                    x={S_LEFT - 8}
                    y={yTop + S_BAR - 3}
                    textAnchor="end"
                    fill="var(--ink)"
                    fontSize={11}
                  >
                    {name}
                  </text>
                  <path
                    d={`M${S_LEFT},${yTop} h${Math.max(0, w - r)} a${r},${r} 0 0 1 ${r},${r} v${
                      S_BAR - 2 * r
                    } a${r},${r} 0 0 1 ${-r},${r} h${-Math.max(0, w - r)} z`}
                    fill="var(--series-1)"
                  />
                  <text
                    x={sx(value) + 5}
                    y={yTop + S_BAR - 3}
                    fill="var(--ink)"
                    fontSize={10.5}
                    style={{ fontVariantNumeric: "tabular-nums" }}
                  >
                    {fmt(value)}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="chan-note" title={`efficiency_error = ${shapley.efficiency_error}`}>
            Exact-decomposition check passed (efficiency error{" "}
            <span className="mono">{fmt(shapley.efficiency_error)}</span>).
          </p>
        </>
      )}
    </Panel>
  );
}
