/**
 * REWRITE EXPLORER — side-by-side original vs transformed learned description
 * with per-critic metric deltas and the semanticity court verdicts.
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import type {
  ChannelShard,
  Critic,
  RewriteCell,
  RewritesShard,
  RowRecord,
  SlimMetric,
} from "../../data/types";
import { useShard } from "../../data/loader";
import {
  Badge,
  ErrorBox,
  LoadingBox,
  Panel,
  Segmented,
  UnavailableBox,
} from "../../components/ui";
import { fmt, fmtSigned, shortRowId } from "../../data/format";
import { AuditLink } from "./lib";

const CRITICS = ["primary", "independent"] as const;
const FAMILIES = ["paraphrase", "corruption"] as const;
type RewriteFamily = (typeof FAMILIES)[number];

type MetricGoodness = "lower" | "higher" | "near-one";

const METRIC_ROWS: { key: keyof SlimMetric; label: string; better: MetricGoodness }[] = [
  { key: "dmse", label: "directional MSE", better: "lower" },
  { key: "raw_mse", label: "raw MSE", better: "lower" },
  { key: "cosine", label: "cosine", better: "higher" },
  { key: "norm_ratio", label: "norm ratio", better: "near-one" },
];

const CAVEAT =
  "Format-preserving transformations are not guaranteed to be semantically identical — 'format paraphrase' names the intent of the transform, not a verified property.";

function familyLabel(family: RewriteFamily): string {
  return family === "paraphrase" ? "format paraphrase" : "corruption";
}

function chipLabel(cell: RewriteCell): string {
  if (cell.family === "corruption" && cell.spec.rate !== undefined) {
    return `${cell.spec.kind} ${Math.round(cell.spec.rate * 100)}%`;
  }
  return cell.variant.replace(/_/g, " ");
}

function variantSortKey(cell: RewriteCell): string {
  const rate = cell.spec.rate !== undefined ? cell.spec.rate.toFixed(3) : "";
  return `${cell.spec.kind}|${rate}|${cell.variant}`;
}

/**
 * Δ vs identity with a direction-of-goodness glyph: ▲ always means "better for
 * this metric" (lower dMSE/raw MSE, higher cosine, norm ratio closer to 1).
 * Near-zero deltas render as ＝ with no arrow.
 */
function DeltaCell(props: { value: number; identity: number; better: MetricGoodness }) {
  const delta = props.value - props.identity;
  let arrow = "＝";
  let sense = "unchanged";
  if (Math.abs(delta) >= 5e-4) {
    let improved: boolean;
    if (props.better === "lower") improved = delta < 0;
    else if (props.better === "higher") improved = delta > 0;
    else improved = Math.abs(props.value - 1) < Math.abs(props.identity - 1);
    arrow = improved ? "▲" : "▼";
    sense = improved ? "better than identity" : "worse than identity";
  }
  return (
    <td
      className="num"
      title={`transformed − identity = ${fmtSigned(delta)} (${sense}; ▲ = better for this metric)`}
    >
      <span aria-hidden>{arrow}</span> {fmtSigned(delta)}
    </td>
  );
}

/**
 * Court outcome in outcome language with the calibration expectation stated:
 * paraphrases are expected to preserve the code; heavy (negative-calibration)
 * corruption is expected to break it; light corruption ("context") carries no
 * expectation.
 */
function CourtOutcome(props: { verdict: boolean; label: string }) {
  const outcome = props.verdict ? "code preserved" : "code broken";
  const expected =
    props.label === "positive" ? true : props.label === "negative" ? false : null;
  const asExpected = expected === null ? null : expected === props.verdict;
  return (
    <span className="chan-court-outcome">
      <span className="badge badge-neutral">{outcome}</span>
      {asExpected === null ? (
        <span className="chan-controls-note">no calibration expectation (context cell)</span>
      ) : asExpected ? (
        <span className="chan-controls-note">as expected for this cell class</span>
      ) : (
        <span className="chan-court-unexpected">unexpected for this cell class</span>
      )}
    </span>
  );
}

export default function RewritePanel(props: {
  channel: ChannelShard;
  critic: Critic;
  row: RowRecord;
  state: AppState;
  update: (patch: Partial<AppState>) => void;
}) {
  const shard = useShard<RewritesShard>("rewrites.json");
  const rowId = props.row.row_id;

  const derived = useMemo(() => {
    if (shard.status !== "ready") return null;
    const byId = new Map<string, RewriteCell>();
    for (const cell of shard.data.cells) byId.set(cell.cell_id, cell);
    const rowCells = shard.data.cells.filter((c) => c.row_id === rowId);
    const byFamily = new Map<RewriteFamily, RewriteCell[]>();
    for (const family of FAMILIES) {
      byFamily.set(
        family,
        rowCells
          .filter((c) => c.family === family)
          .sort((a, b) => variantSortKey(a).localeCompare(variantSortKey(b))),
      );
    }
    const identity = shard.data.identity.find((i) => i.row_id === rowId) ?? null;
    return { byId, byFamily, identity };
  }, [shard, rowId]);

  // Resolve the selected transform cell: exact cell id from URL state when it
  // matches the selected row; otherwise carry family+variant to the new row;
  // otherwise a display-only default (not written to the URL until user acts).
  const selectedCell = useMemo<RewriteCell | null>(() => {
    if (!derived) return null;
    const fromState = props.state.cell ? derived.byId.get(props.state.cell) : undefined;
    if (fromState && fromState.row_id === rowId) return fromState;
    if (fromState) {
      const carried = derived.byFamily
        .get(fromState.family)
        ?.find((c) => c.variant === fromState.variant);
      if (carried) return carried;
    }
    return derived.byFamily.get("paraphrase")?.[0] ?? derived.byFamily.get("corruption")?.[0] ?? null;
  }, [derived, props.state.cell, rowId]);

  const family: RewriteFamily = selectedCell?.family ?? "paraphrase";

  const pickFamily = (next: RewriteFamily) => {
    if (!derived) return;
    const cells = derived.byFamily.get(next) ?? [];
    const sameVariant = selectedCell
      ? cells.find((c) => c.variant === selectedCell.variant)
      : undefined;
    const target = sameVariant ?? cells[0];
    if (target) props.update({ cell: target.cell_id });
  };

  return (
    <Panel
      id="chan-rewrite"
      title={`Rewrite explorer — ${shortRowId(rowId)}`}
      span={12}
      badges={
        <>
          <Badge status="qualified" label="stored-snapshot" />
          <AuditLink claim="court" update={props.update} />
        </>
      }
      sub="Side-by-side: the original learned description of the stored activation vs a format transform, scored by both critics against the same stored target."
    >
      {shard.status === "loading" ? (
        <LoadingBox what="rewrite cells (rewrites.json)" />
      ) : shard.status === "error" ? (
        <ErrorBox message={shard.message} />
      ) : !derived || (!derived.identity && !selectedCell) ? (
        <UnavailableBox>No rewrite evidence for row {shortRowId(rowId)} in rewrites.json.</UnavailableBox>
      ) : (
        <>
          <div className="chan-rw-grid">
            {/* ------------------------------ original ----------------------------- */}
            <div className="chan-rw-pane">
              <h4 className="chan-h4">Original — learned description (identity)</h4>
              {!derived.identity ? (
                <UnavailableBox>No identity entry for this row in rewrites.json.</UnavailableBox>
              ) : (
                <>
                  <div className="text-block">{derived.identity.text}</div>
                  <div className="chan-table-wrap">
                    <table className="data-table" tabIndex={0}>
                      <caption className="visually-hidden">Identity metrics for both critics</caption>
                      <thead>
                        <tr>
                          <th scope="col">metric</th>
                          {CRITICS.map((c) => (
                            <th key={c} scope="col" className="num">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {METRIC_ROWS.map((m) => (
                          <tr key={m.key}>
                            <td>{m.label}</td>
                            {CRITICS.map((c) => (
                              <td key={c} className="num">
                                {fmt(derived.identity!.metrics[c][m.key])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>

            {/* ----------------------------- transformed --------------------------- */}
            <div className="chan-rw-pane">
              <h4 className="chan-h4">Transformed</h4>
              <div className="chan-rw-picker">
                <Segmented
                  options={FAMILIES}
                  value={family}
                  onChange={pickFamily}
                  label="Transform family"
                  format={familyLabel}
                />
                <p className="chan-caveat chan-rw-caveat">{CAVEAT}</p>
                <div role="group" aria-label={`${family} variants`} style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {(derived.byFamily.get(family) ?? []).map((c) => (
                    <button
                      key={c.cell_id}
                      type="button"
                      className="token-chip"
                      aria-pressed={selectedCell?.cell_id === c.cell_id}
                      onClick={() => props.update({ cell: c.cell_id })}
                      title={`variant ${c.variant}`}
                    >
                      {chipLabel(c)}
                    </button>
                  ))}
                </div>
              </div>
              {!selectedCell ? (
                <UnavailableBox>
                  No {family} cells for row {shortRowId(rowId)} in rewrites.json.
                </UnavailableBox>
              ) : (
                <>
                  <div className="text-block">{selectedCell.text}</div>
                  <div className="chan-table-wrap">
                    <table className="data-table" tabIndex={0}>
                      <caption className="visually-hidden">
                        Transformed metrics with deltas vs identity for both critics
                      </caption>
                      <thead>
                        <tr>
                          <th scope="col">metric</th>
                          <th scope="col" className="num">primary</th>
                          <th scope="col" className="num">Δ vs identity</th>
                          <th scope="col" className="num">independent</th>
                          <th scope="col" className="num">Δ vs identity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {METRIC_ROWS.map((m) => (
                          <tr key={m.key}>
                            <td>{m.label}</td>
                            {CRITICS.map((c) => {
                              const value = selectedCell.metrics[c][m.key];
                              const ident = derived.identity?.metrics[c][m.key];
                              return (
                                <FragmentCells
                                  key={c}
                                  value={value}
                                  identity={ident}
                                  better={m.better}
                                />
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="chan-note">
                    Δ columns: ▲ = better than identity for that metric, ▼ = worse.
                  </p>
                  <details className="chan-details">
                    <summary>cell spec &amp; id (provenance)</summary>
                    <p className="chan-note">
                      spec: <span className="mono">{JSON.stringify(selectedCell.spec)}</span> · cell{" "}
                      <span className="mono">{selectedCell.cell_id}</span>
                    </p>
                  </details>
                </>
              )}
            </div>
          </div>

          {/* -------------------------------- court -------------------------------- */}
          {selectedCell ? (
            <div className="chan-rw-court">
              <h4 className="chan-h4">
                Semanticity court
                <Badge
                  status="exploratory"
                  label="validation-fitted"
                  title={`Court thresholds fit on the ${props.channel.fit_split} split`}
                />
                <span className="chan-controls-note">
                  identity cosine ≥ threshold ⇒ code preserved · paraphrases should preserve,
                  heavy corruption should break
                </span>
              </h4>
              <div className="chan-table-wrap">
                <table className="data-table" tabIndex={0}>
                  <caption className="visually-hidden">
                    Court verdicts per critic for the selected transform
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">critic</th>
                      <th scope="col" className="num">identity_cosine</th>
                      <th scope="col" className="num">threshold</th>
                      <th scope="col">calibration_label</th>
                      <th scope="col">semanticity_verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {CRITICS.map((c) => {
                      const court = selectedCell.court[c];
                      const threshold = props.channel.court_thresholds[c];
                      return (
                        <tr key={c} aria-selected={c === props.critic}>
                          <td>{c}</td>
                          <td className="num">{fmt(court.identity_cosine, 4)}</td>
                          <td className="num">{threshold ? fmt(threshold.threshold, 4) : "—"}</td>
                          <td className="mono">{court.calibration_label}</td>
                          <td>
                            <CourtOutcome
                              verdict={court.semanticity_verdict}
                              label={court.calibration_label}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </>
      )}
    </Panel>
  );
}

/** Two-cell fragment: metric value + goodness-encoded delta vs identity. */
function FragmentCells(props: {
  value: number;
  identity: number | undefined;
  better: MetricGoodness;
}) {
  return (
    <>
      <td className="num">{fmt(props.value)}</td>
      {props.identity === undefined ? (
        <td className="num">—</td>
      ) : (
        <DeltaCell value={props.value} identity={props.identity} better={props.better} />
      )}
    </>
  );
}
