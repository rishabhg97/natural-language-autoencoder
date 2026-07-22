/**
 * Panel 3b — next-token top-k movement for the focused lane cell: original
 * (stored activation forward) vs patched top-10 with rank-movement markers,
 * plus a cross-lane strip of top-1 patched tokens.
 */

import { useMemo } from "react";
import type { TopKToken } from "../../data/types";
import type { AppState } from "../../app/urlState";
import { Badge, Panel, Segmented, UnavailableBox } from "../../components/ui";
import { fmt } from "../../data/format";
import { HBar } from "../../components/charts";
import { LANE_META, TEACHER_CHIP, type LaneCell, type LaneKey } from "./laneModel";
import { LaneSwatch, LinkButton, visibleToken } from "./benchUi";

const W = 680;
const HEADER_Y = 14;
const ROW_H = 22;
const BAR_H = 10;
const LEFT = { rankX: 2, tokenX: 20, barX: 116, barMax: 138, colEnd: 302 };
const RIGHT_DX = 380;
const RIGHT = { rankX: 0, tokenX: 18, barX: 112, barMax: 100, indicatorX: W - 2 };

function clipToken(text: string, max = 13): string {
  const v = visibleToken(text);
  return v.length <= max ? v : `${v.slice(0, max - 1)}…`;
}

export default function TopKPanel(props: {
  rack: LaneCell[];
  focusLane: LaneKey;
  chip: string;
  update: (patch: Partial<AppState>) => void;
}) {
  const { rack, focusLane, chip, update } = props;
  const availableLanes = rack.filter((l) => l.cell !== null).map((l) => l.lane);
  const focus = rack.find((l) => l.lane === focusLane && l.cell) ?? rack.find((l) => l.cell);
  const cell = focus?.cell ?? null;

  const rows = useMemo(() => {
    if (!cell) return null;
    const original = cell.topk.original;
    const patched = cell.topk.patched;
    const maxP = Math.max(...original.map((t) => t.p), ...patched.map((t) => t.p), 1e-12);
    const origRank = new Map<number, number>();
    original.forEach((t, i) => origRank.set(t.id, i));
    const patchedIds = new Set(patched.map((t) => t.id));
    return { original, patched, maxP, origRank, patchedIds };
  }, [cell]);

  const height = HEADER_Y + 10 + (rows ? Math.max(rows.original.length, rows.patched.length) : 10) * ROW_H + 6;

  return (
    <Panel
      title="Next-token top-k movement"
      span={6}
      badges={<Badge status="exploratory" label="functional: validation-only" />}
      sub="Original = next-token distribution over the stored activation; patched = after substituting the lane's re-encoded description. Tokens in both lists carry a rank-movement marker; patched-only tokens are marked new."
      id="bench-topk"
    >
      <div className="bench-controls">
        <Segmented
          options={availableLanes}
          value={availableLanes.includes(focusLane) ? focusLane : availableLanes[0] ?? "edit"}
          onChange={(v) => update({ cell: v })}
          label="focused lane"
          format={(l) => LANE_META[l].short}
        />
        <LinkButton
          onClick={() => update({ station: "audit", claim: "functional_interventions" })}
          title="Functional interventions claim scope on AUDIT"
        >
          AUDIT: functional claim
        </LinkButton>
      </div>
      {!cell || !rows ? (
        <UnavailableBox>no precomputed top-k readout for this selection.</UnavailableBox>
      ) : (
        <>
          <svg
            viewBox={`0 0 ${W} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            className="chart-svg bench-chart"
            role="img"
            aria-label={`Top-10 next-token movement for the ${LANE_META[focus!.lane].label} lane: original vs patched probabilities`}
          >
            <text x={LEFT.rankX} y={HEADER_Y} fontWeight={650}>
              original top-10
            </text>
            <text x={RIGHT_DX + RIGHT.rankX} y={HEADER_Y} fontWeight={650}>
              patched top-10
            </text>
            <text x={RIGHT.indicatorX} y={HEADER_Y} textAnchor="end">
              move
            </text>
            {rows.original.map((t: TopKToken, i: number) => {
              const y = HEADER_Y + 10 + i * ROW_H;
              const barW = (t.p / rows.maxP) * LEFT.barMax;
              const dropped = !rows.patchedIds.has(t.id);
              return (
                <g key={`o-${t.id}-${i}`}>
                  <text x={LEFT.rankX} y={y + 15} className="tick-label">
                    {i + 1}
                  </text>
                  <text x={LEFT.tokenX} y={y + 15} className="mono">
                    <title>{`token ${JSON.stringify(t.text)} · p=${fmt(t.p, 4)}${dropped ? " · leaves the patched top-10" : ""}`}</title>
                    {clipToken(t.text)}
                  </text>
                  <HBar
                    x0={LEFT.barX}
                    x1={LEFT.barX + Math.max(1, barW)}
                    y={y + 6}
                    height={BAR_H}
                    fill="var(--axis)"
                  />
                  <text x={LEFT.barX + Math.max(1, barW) + 4} y={y + 15}>
                    {fmt(t.p, 3)}
                  </text>
                </g>
              );
            })}
            {rows.patched.map((t: TopKToken, i: number) => {
              const y = HEADER_Y + 10 + i * ROW_H;
              const barW = (t.p / rows.maxP) * RIGHT.barMax;
              const from = rows.origRank.get(t.id);
              const marker =
                from === undefined
                  ? "new"
                  : from === i
                    ? "="
                    : from > i
                      ? `↑${from - i}`
                      : `↓${i - from}`;
              return (
                <g key={`p-${t.id}-${i}`}>
                  {from !== undefined ? (
                    <line
                      x1={LEFT.colEnd}
                      y1={HEADER_Y + 10 + from * ROW_H + 11}
                      x2={RIGHT_DX - 4}
                      y2={y + 11}
                      stroke="var(--grid)"
                      strokeWidth={1}
                    >
                      <title>{`${JSON.stringify(t.text)}: rank ${from + 1} → ${i + 1}`}</title>
                    </line>
                  ) : null}
                  <text x={RIGHT_DX + RIGHT.rankX} y={y + 15} className="tick-label">
                    {i + 1}
                  </text>
                  <text x={RIGHT_DX + RIGHT.tokenX} y={y + 15} className="mono">
                    <title>{`token ${JSON.stringify(t.text)} · p=${fmt(t.p, 4)}`}</title>
                    {clipToken(t.text)}
                  </text>
                  <HBar
                    x0={RIGHT_DX + RIGHT.barX}
                    x1={RIGHT_DX + RIGHT.barX + Math.max(1, barW)}
                    y={y + 6}
                    height={BAR_H}
                    fill={LANE_META[focus!.lane].color}
                  />
                  <text x={RIGHT_DX + RIGHT.barX + Math.max(1, barW) + 4} y={y + 15}>
                    {fmt(t.p, 3)}
                  </text>
                  <text
                    x={RIGHT.indicatorX}
                    y={y + 15}
                    textAnchor="end"
                    fontWeight={marker === "new" ? 700 : 450}
                  >
                    <title>
                      {marker === "new"
                        ? "token only in the patched top-10"
                        : `rank moved ${marker === "=" ? "0" : marker}`}
                    </title>
                    {marker}
                  </text>
                </g>
              );
            })}
          </svg>
          {(() => {
            const withCells = rack.filter((l) => l.cell);
            const top1s = new Set(
              withCells.map((l) => l.cell!.topk.patched[0]?.id ?? -1),
            );
            const overlaps = new Set(
              withCells.map((l) => l.cell!.behavior.top_10_overlap),
            );
            return withCells.length > 1 && top1s.size === 1 ? (
              <p className="bench-note">
                Computed from the strip below: every lane — including the controls — produces the
                same patched top-1 token
                {overlaps.size === 1 ? " with identical top-10 overlap" : ""}. This readout shows
                no lane-specific movement on this cell.
              </p>
            ) : null;
          })()}
          <p className="bench-note">
            Tokens are model tokens — fragments like <span className="mono">osi</span> are
            sub-word pieces, not whole words.
          </p>
          <div className="bench-scroll">
            <table className="data-table bench-topk-strip" tabIndex={0} aria-label="cross-lane top-1 patched token comparison">
              <thead>
                <tr>
                  <th scope="col">lane</th>
                  <th scope="col">top-1 patched token</th>
                  <th scope="col" className="num">
                    p
                  </th>
                  <th scope="col" className="num">
                    top-10 overlap
                  </th>
                </tr>
              </thead>
              <tbody>
                {rack.map((l) => {
                  const top1 = l.cell?.topk.patched[0];
                  return (
                    <tr key={l.lane} aria-selected={l.lane === focus!.lane}>
                      <td>
                        <LaneSwatch lane={l.lane} /> {LANE_META[l.lane].label}
                      </td>
                      {l.cell && top1 ? (
                        <>
                          <td>
                            <span className="bench-token" title={JSON.stringify(top1.text)}>
                              {visibleToken(top1.text)}
                            </span>
                          </td>
                          <td className="num">{fmt(top1.p, 3)}</td>
                          <td className="num">{fmt(l.cell.behavior.top_10_overlap, 2)}</td>
                        </>
                      ) : (
                        <td colSpan={3}>— no cell</td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {chip === TEACHER_CHIP ? (
            <p className="bench-note">
              identity-only view: even the unedited teacher re-encoding moves the top-10 — that
              movement is the harness floor.
            </p>
          ) : null}
        </>
      )}
    </Panel>
  );
}
