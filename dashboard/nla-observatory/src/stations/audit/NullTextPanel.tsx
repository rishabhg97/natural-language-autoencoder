/**
 * Null-text almanac — words enriched in learned descriptions of real stored
 * activations vs the zero-vector control, plus the E1-AV / E2 loss lanes.
 * The scope and backfill caveat render verbatim; the zero-control list is
 * furniture and always shows.
 */

import { useMemo } from "react";
import type { AuditShard } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { HBar } from "../../components/charts";
import { fmt, fmtPct, fmtSigned, preview } from "../../data/format";

const TOP_N = 20;
const ROW_H = 15;
const VIRTUAL_W = 320;
const LABEL_W = 118;
const VALUE_W = 44;

function WordBars(props: {
  title: string;
  color: string;
  words: { token: string; log_odds_real_vs_zero: number }[];
  max: number;
}) {
  const height = props.words.length * ROW_H + 8;
  const barMax = VIRTUAL_W - LABEL_W - VALUE_W - 8;
  return (
    <figure className="audit-wordlist">
      <figcaption>{props.title}</figcaption>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${VIRTUAL_W} ${height}`}
        width="100%"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`${props.title}: top ${props.words.length} words by absolute log-odds, values labelled per bar`}
      >
        {props.words.map((w, i) => {
          const mag = Math.abs(w.log_odds_real_vs_zero);
          const bw = props.max > 0 ? (mag / props.max) * barMax : 0;
          const y = 4 + i * ROW_H;
          return (
            <g key={`${w.token}-${i}`}>
              <text x={LABEL_W - 4} y={y + 10} textAnchor="end">
                <title>{w.token}</title>
                {preview(w.token, 16)}
              </text>
              <HBar x0={LABEL_W} x1={LABEL_W + bw} y={y + 2.5} height={10} fill={props.color} />
              <text x={LABEL_W + bw + 4} y={y + 10}>
                {fmtSigned(w.log_odds_real_vs_zero, 2)}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

function lossOrder(keys: string[]): string[] {
  return keys.slice().sort((a, b) => {
    if (a === "real") return -1;
    if (b === "real") return 1;
    return a.localeCompare(b);
  });
}

export default function NullTextPanel(props: { nt: AuditShard["null_text"] }) {
  const { nt } = props;
  const real = useMemo(() => nt.real_enriched_words.slice(0, TOP_N), [nt]);
  const zero = useMemo(() => nt.zero_enriched_words.slice(0, TOP_N), [nt]);
  const max = useMemo(
    () =>
      Math.max(
        0,
        ...real.map((w) => Math.abs(w.log_odds_real_vs_zero)),
        ...zero.map((w) => Math.abs(w.log_odds_real_vs_zero)),
      ),
    [real, zero],
  );
  const lossKeys = useMemo(
    () =>
      lossOrder(
        Array.from(new Set([...Object.keys(nt.e1_av_losses), ...Object.keys(nt.e2_mean_loss)])),
      ),
    [nt],
  );

  return (
    <Panel
      id="null_text"
      span={6}
      title="Null-text almanac"
      sub={
        <>
          Word enrichment in learned descriptions of real stored activations vs the zero-vector
          control ({nt.row_count.toLocaleString()} rows). Scope, verbatim:{" "}
          <span className="mono">{nt.scope}</span>
        </>
      }
      badges={<Badge status="exploratory" label="preliminary" />}
    >
      <p className="audit-note">
        <Badge status="caveat" label="backfill pending" /> {nt.backfill_note}
      </p>

      {real.length === 0 && zero.length === 0 ? (
        <UnavailableBox>No enriched-word lists in this bundle.</UnavailableBox>
      ) : (
        <div className="audit-wordlists">
          {real.length > 0 ? (
            <WordBars
              title="Enriched with real activation (log-odds real vs zero)"
              color="var(--series-1)"
              words={real}
              max={max}
            />
          ) : (
            <UnavailableBox>No real-enriched words.</UnavailableBox>
          )}
          {zero.length > 0 ? (
            <WordBars
              title="Enriched under zero-vector control"
              color="var(--ink-3)"
              words={zero}
              max={max}
            />
          ) : (
            <UnavailableBox>No zero-enriched words.</UnavailableBox>
          )}
        </div>
      )}

      <h4 className="audit-subhead">Description losses, real vs controls</h4>
      <div className="audit-table-scroll">
        <table className="data-table" tabIndex={0} aria-label="Null text control results">
          <thead>
            <tr>
              <th scope="col">condition</th>
              <th scope="col" className="num">
                E1-AV loss
              </th>
              <th scope="col" className="num">
                E2 mean loss
              </th>
              <th scope="col" className="num">
                Δ real − control
              </th>
              <th scope="col" className="num">
                real wins
              </th>
            </tr>
          </thead>
          <tbody>
            {lossKeys.map((k) => {
              const paired = nt.e2_paired[k];
              return (
                <tr key={k}>
                  <td className="mono">{k}</td>
                  <td className="num">
                    {nt.e1_av_losses[k] !== undefined ? fmt(nt.e1_av_losses[k], 4) : "—"}
                  </td>
                  <td className="num">
                    {nt.e2_mean_loss[k] !== undefined ? fmt(nt.e2_mean_loss[k], 4) : "—"}
                  </td>
                  <td className="num">
                    {paired ? fmtSigned(paired.mean_real_minus_control, 4) : "—"}
                  </td>
                  <td className="num">{paired ? fmtPct(paired.real_win_fraction, 1) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
