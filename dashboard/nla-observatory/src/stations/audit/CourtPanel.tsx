/**
 * Cipher-court docket — per-critic thresholds (validation-fitted) and the
 * full 50-row x 2-critic docket. Paraphrase cells calibrate the positive
 * class; heavy corruption calibrates the negative class; light corruption is
 * "context" (scored against the threshold, not used to fit it). Worst
 * verdicts sort first so suspect rows are never buried.
 */

import { useMemo } from "react";
import type { StationProps } from "../../app/stationProps";
import type { AuditShard, CourtDocketCell, CourtDocketRow, Critic } from "../../data/types";
import { Badge, Panel, Segmented, UnavailableBox } from "../../components/ui";
import { useTooltip } from "../../components/charts";
import { fmtFixed, shortRowId } from "../../data/format";
import { humanize } from "./util";

const CRITICS = ["primary", "independent"] as const;
const VERDICT_RANK: Record<CourtDocketRow["row_verdict"], number> = {
  suspect: 0,
  mixed: 1,
  honest: 2,
};

function verdictBadge(verdict: CourtDocketRow["row_verdict"]) {
  if (verdict === "honest") return <Badge status="qualified" label="honest" />;
  if (verdict === "mixed") return <Badge status="caveat" label="mixed" />;
  return <Badge status="negative" label="suspect" />;
}

/**
 * One corruption mini-cell: color = calibration label, glyph = outcome
 * (✓ code preserved / ✕ code broken). Expectation is stated per class:
 * negative-calibration (heavy) corruption is EXPECTED to break the code, so
 * a preserved heavy corruption gets the "unexpected" ring, not a green tick.
 */
function CourtCell(props: { cell: CourtDocketCell }) {
  const tip = useTooltip();
  const c = props.cell;
  const outcome = c.verdict ? "code preserved" : "code broken";
  const expectation =
    c.label === "negative"
      ? c.verdict
        ? "unexpected — heavy corruption should break the code"
        : "as expected for heavy corruption"
      : c.label === "positive"
        ? c.verdict
          ? "as expected for a paraphrase"
          : "unexpected — paraphrases should preserve the code"
        : "no calibration expectation (context cell)";
  const unexpected =
    (c.label === "negative" && c.verdict) || (c.label === "positive" && !c.verdict);
  const text = `${c.variant} — identity cosine ${fmtFixed(c.identity_cosine, 4)} — ${outcome} (${expectation})`;
  const content = (
    <div>
      <span className="tt-value">{c.variant}</span>
      <br />
      <span className="tt-label">identity cosine </span>
      <span className="tt-value">{fmtFixed(c.identity_cosine, 4)}</span>
      <br />
      <span className="tt-value">{outcome}</span>
      <br />
      <span className="tt-label">{expectation}</span>
    </div>
  );
  return (
    <button
      type="button"
      className={`audit-court-cell audit-court-cell-${c.label}${unexpected ? " audit-court-cell-unexpected" : ""}`}
      aria-label={text}
      title={text}
      onPointerMove={(e) => tip.show({ x: e.clientX, y: e.clientY, content })}
      onPointerLeave={tip.hide}
      onFocus={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        tip.show({ x: r.right, y: r.bottom, content });
      }}
      onBlur={tip.hide}
    >
      {c.verdict ? "✓" : "✕"}
    </button>
  );
}

export default function CourtPanel(props: { audit: AuditShard } & StationProps) {
  const { court } = props.audit;
  const critic: Critic = props.state.critic;

  const docket = useMemo(() => {
    return court.docket
      .filter((r) => r.critic === critic)
      .slice()
      .sort(
        (a, b) =>
          (VERDICT_RANK[a.row_verdict] ?? 3) - (VERDICT_RANK[b.row_verdict] ?? 3) ||
          a.row_id.localeCompare(b.row_id),
      );
  }, [court.docket, critic]);

  const verdictCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const r of docket) counts[r.row_verdict] = (counts[r.row_verdict] ?? 0) + 1;
    return counts;
  }, [docket]);

  const labelVariants = useMemo(() => {
    const byLabel = new Map<string, Set<string>>();
    for (const r of docket) {
      for (const c of r.corruption_cells) {
        if (!byLabel.has(c.label)) byLabel.set(c.label, new Set());
        byLabel.get(c.label)!.add(c.variant);
      }
    }
    return [...byLabel.entries()]
      .map(([label, vs]) => ({ label, variants: [...vs].sort() }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [docket]);

  const criticOrder = (Object.keys(court.thresholds) as Critic[]).sort((a, b) =>
    a === "primary" ? -1 : b === "primary" ? 1 : a.localeCompare(b),
  );

  return (
    <Panel
      id="court"
      span={12}
      title="Cipher court docket"
      sub="Row-level semanticity verdicts. Paraphrases calibrate the positive class; heavy corruption calibrates the negative class; light corruption is scored as 'context' and does not fit the threshold."
      badges={<Badge status="exploratory" label="validation-fitted" />}
    >
      <div className="audit-court-thresholds">
        {criticOrder.map((c) => {
          const t = court.thresholds[c];
          return (
            <div className="audit-threshold-card" key={c}>
              <h4 className="audit-subhead">
                {c} critic <Badge status="exploratory" label={`fit on ${court.fit_split}`} />
              </h4>
              <dl className="kv-list">
                <dt>threshold (identity cosine)</dt>
                <dd>{fmtFixed(t.threshold, 4)}</dd>
                <dt>balanced accuracy</dt>
                <dd>{fmtFixed(t.balanced_accuracy, 4)}</dd>
                <dt>positive recall</dt>
                <dd>{fmtFixed(t.positive_recall, 4)}</dd>
                <dt>negative recall</dt>
                <dd>{fmtFixed(t.negative_recall, 4)}</dd>
              </dl>
            </div>
          );
        })}
      </div>
      <p className="audit-note">
        <Badge status="caveat" label="confound" /> {court.confound}
      </p>

      <div className="controls-row">
        <Segmented
          options={CRITICS}
          value={critic}
          onChange={(v) => props.update({ critic: v })}
          label="Docket critic"
        />
        <span className="audit-inline-count" aria-live="polite">
          {docket.length} rows ·{" "}
          {["suspect", "mixed", "honest"]
            .filter((v) => verdictCounts[v])
            .map((v) => `${verdictCounts[v]} ${v}`)
            .join(" · ")}
        </span>
      </div>

      {docket.length === 0 ? (
        <UnavailableBox>No docket rows for the {critic} critic in this bundle.</UnavailableBox>
      ) : (
        <details className="audit-details audit-docket-details">
          <summary>Browse all {docket.length} row-level verdicts</summary>
          <div className="audit-table-scroll">
            <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th scope="col">row</th>
                <th scope="col">verdict</th>
                <th scope="col" className="num">
                  para min cos
                </th>
                <th scope="col" className="num">
                  para mean cos
                </th>
                <th scope="col" className="num">
                  para verdicts
                </th>
                <th scope="col">corruption cells</th>
              </tr>
            </thead>
            <tbody>
              {docket.map((r) => (
                <tr key={`${r.critic}-${r.row_id}`}>
                  <td>
                    <button
                      type="button"
                      className="audit-rowlink"
                      onClick={() => props.update({ station: "channel", row: r.row_id })}
                      title={`Open ${r.row_id} in the CHANNEL station`}
                    >
                      {shortRowId(r.row_id)}
                    </button>
                  </td>
                  <td>{verdictBadge(r.row_verdict)}</td>
                  <td className="num">{fmtFixed(r.paraphrase_min_identity_cosine, 4)}</td>
                  <td className="num">{fmtFixed(r.paraphrase_mean_identity_cosine, 4)}</td>
                  <td className="num">
                    {r.paraphrase_verdicts_true}/{r.paraphrase_cells.length}
                  </td>
                  <td>
                    <span className="audit-court-cells">
                      {r.corruption_cells
                        .slice()
                        .sort((a, b) => a.variant.localeCompare(b.variant))
                        .map((c) => (
                          <CourtCell cell={c} key={c.variant} />
                        ))}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            </table>
          </div>
        </details>
      )}

      <div className="audit-legend-row">
        {labelVariants.map(({ label, variants }) => (
          <span key={label}>
            <span className={`audit-court-cell audit-court-cell-${label} audit-court-cell-swatch`} aria-hidden>
              {"✓"}
            </span>{" "}
            {humanize(label)}
            {label === "negative" ? " calibration" : ""}: <span className="mono">{variants.join(", ")}</span>
          </span>
        ))}
        <span>
          glyph = per-cell outcome ({"✓"} code preserved / {"✕"} code broken)
        </span>
        <span>
          Expected: paraphrases preserve ({"✓"}); heavy corruption breaks ({"✕"}) — a preserved
          heavy corruption is the suspicious outcome and gets a ring.
        </span>
      </div>
    </Panel>
  );
}
