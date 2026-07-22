/**
 * All-cases summary strip + aggregate footer. Weak and negative results stay
 * visible: zero lifts render as zero, missing onsets carry a negative badge,
 * and the interpretation notes (including the "rough planning signal, not
 * proof" line) render in full.
 */
import type { PoetryShard } from "../../data/types";
import { Badge, HashChip } from "../../components/ui";
import { fmt, fmtPct, fmtSigned } from "../../data/format";

export default function CaseSummary(props: {
  poetry: PoetryShard;
  currentCaseId: string;
  onSelect: (caseId: string) => void;
}) {
  const { poetry } = props;
  const agg = poetry.aggregates;
  const phases = Object.entries(poetry.reports);
  const phasesPassed = phases.filter(([, r]) => r.passed).length;

  return (
    <>
      <div className="trc-cases" role="group" aria-label="Poetry cases">
        {poetry.cases.map((c) => (
          <button
            key={c.case_id}
            type="button"
            className="trc-case-card"
            aria-pressed={c.case_id === props.currentCaseId}
            onClick={() => props.onSelect(c.case_id)}
          >
            <span className="trc-case-head">
              <strong className="trc-case-id mono">{c.case_id}</strong>
              <span>{c.case_id === props.currentCaseId ? "currently open" : "open case"}</span>
            </span>
            <span className="trc-case-copy">
              <small>Original first line shown to Nano30B</small>
              <span>{c.first_line}</span>
            </span>
            <span className="trc-case-copy trc-case-copy-reference">
              <small>Held-out reference second line</small>
              <span>{c.second_line}</span>
            </span>
            <span className="trc-case-copy trc-case-copy-output">
              <small>Observed unpatched baseline output</small>
              <span>{c.baseline_continuation}</span>
            </span>
            <span className="trc-case-metrics">
              <span title="real minus shuffled target-family rate, measured at the anchor position only — an early-window spike can coexist with zero anchor lift">
                anchor lift <strong>{fmtSigned(c.anchor_lift, 3)}</strong>
              </span>
              <span>
                {c.planning_onset_relative_offset !== null ? (
                  `planning onset ${c.planning_onset_relative_offset}`
                ) : (
                  <Badge status="negative" label="no onset" />
                )}
              </span>
              <span>{c.baseline_hits_target_family ? "✓" : "✗"} baseline target rhyme</span>
              <span>{c.reconstruction !== null ? "✓" : "✗"} editable</span>
            </span>
          </button>
        ))}
      </div>
      <div className="trc-agg">
        <dl className="kv-list">
          <dt title="real minus shuffled target-family rate at the anchor position only">
            mean anchor lift (real − shuffled, at the anchor)
          </dt>
          <dd>{fmtSigned(agg.mean_anchor_lift, 3)}</dd>
          <dt>cases with planning onset</dt>
          <dd>
            {agg.cases_with_planning_onset}/{agg.cases}
          </dd>
          <dt>baseline target rhyme</dt>
          <dd>
            {agg.cases_with_baseline_target_rhyme}/{agg.cases}
          </dd>
          <dt>editable cases</dt>
          <dd>
            {agg.editable_cases}/{agg.cases}
          </dd>
          <dt>edited alternate-family hit rate</dt>
          <dd>{fmtPct(agg.edited_alternate_hit_rate)}</dd>
          <dt>random-control alternate-family hit rate</dt>
          <dd>{fmtPct(agg.random_alternate_hit_rate)}</dd>
          <dt>usable sample fraction</dt>
          <dd>{fmtPct(agg.usable_fraction)}</dd>
          <dt>samples / positions</dt>
          <dd>
            {agg.samples} / {agg.positions}
          </dd>
          <dt>steering doses</dt>
          <dd>{agg.steering_doses.join(" / ")}</dd>
          <dt>mean original dMSE</dt>
          <dd>{fmt(agg.mean_original_dmse, 3)}</dd>
        </dl>
        <div>
          <p className="trc-note">
            <Badge
              status={poetry.interpretation.signal === "weak" ? "caveat" : "exploratory"}
              label={`signal: ${poetry.interpretation.signal}`}
            />{" "}
            pipeline phases completed: {phasesPassed}/{phases.length} (completeness of the run, not
            scientific success) <HashChip hash={poetry.config_sha256} label="poetry cfg" />
          </p>
          <ul className="trc-notes">
            {poetry.interpretation.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      </div>
    </>
  );
}
