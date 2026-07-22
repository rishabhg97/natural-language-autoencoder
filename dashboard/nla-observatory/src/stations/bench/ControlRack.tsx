/**
 * Panel 2 — control rack. The four lanes travel together in one component:
 * edit + paraphrase placebo + random edit + identity (teacher). There is no
 * way to dismiss a control lane.
 */

import type { Critic } from "../../data/types";
import type { AppState } from "../../app/urlState";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { fmt, fmtSigned } from "../../data/format";
import { criticMetrics, LANE_META, TEACHER_CHIP, type LaneCell } from "./laneModel";
import { LinkButton } from "./benchUi";

const CRITICS: readonly Critic[] = ["primary", "independent"];

function LaneMetricsTable(props: { laneCell: LaneCell }) {
  const cell = props.laneCell.cell;
  if (!cell) return null;
  return (
    <div className="bench-scroll">
      <table className="data-table" tabIndex={0} aria-label={`${LANE_META[props.laneCell.lane].label} directional metrics`}>
        <thead>
          <tr>
            <th scope="col">critic</th>
            <th scope="col" className="num">
              dmse
            </th>
            <th scope="col" className="num">
              cosine
            </th>
            <th scope="col" className="num">
              norm ratio
            </th>
          </tr>
        </thead>
        <tbody>
          {CRITICS.map((critic) => {
            const m = criticMetrics(cell, critic);
            return (
              <tr key={critic}>
                <td>{critic}</td>
                {m ? (
                  <>
                    <td className="num">{fmt(m.dmse)}</td>
                    <td className="num">{fmt(m.cosine)}</td>
                    <td className="num">{fmt(m.norm_ratio)}</td>
                  </>
                ) : (
                  <>
                    <td className="num" title="not scored by independent critic">
                      —
                    </td>
                    <td className="num" title="not scored by independent critic">
                      —
                    </td>
                    <td className="num" title="not scored by independent critic">
                      —
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LaneCard(props: { laneCell: LaneCell; chip: string }) {
  const { laneCell, chip } = props;
  const meta = LANE_META[laneCell.lane];
  const isPrimary =
    laneCell.lane === "edit" || (chip === TEACHER_CHIP && laneCell.lane === "teacher");
  return (
    <article className="bench-lane-card" aria-label={`${meta.label} lane`}>
      <div className="bench-lane-key" style={{ background: meta.color }} aria-hidden />
      <div className="bench-lane-head">
        {meta.label}
        {isPrimary ? <Badge status="selected" label="primary selection" /> : null}
      </div>
      <p className="bench-lane-role">{meta.role}</p>
      {laneCell.cell ? (
        <>
          <span className="bench-cell-id" title={laneCell.cell.cell_id}>
            {laneCell.cell.cell_id}
          </span>
          <details className="bench-details">
            <summary>
              {laneCell.lane === "teacher"
                ? "teacher description text (identity encoding source)"
                : "lane description text (source of the patched encoding)"}
            </summary>
            <div className="text-block">{laneCell.cell.text}</div>
          </details>
          <LaneMetricsTable laneCell={laneCell} />
        </>
      ) : (
        <UnavailableBox>no precomputed cell for this lane at this dose.</UnavailableBox>
      )}
    </article>
  );
}

export default function ControlRack(props: {
  rack: LaneCell[];
  rowId: string;
  chip: string;
  dose: string;
  update: (patch: Partial<AppState>) => void;
}) {
  const { rack, rowId, chip, dose, update } = props;
  const focusLane = rack.find((lane) => lane.lane === "edit") ?? rack.find((lane) => lane.lane === "teacher");
  const placeboLane = rack.find((lane) => lane.lane === "paraphrase_placebo");
  const teacherLane = rack.find((lane) => lane.lane === "teacher");
  const focusMetrics = focusLane?.cell ? criticMetrics(focusLane.cell, "primary") : null;
  const placeboMetrics = placeboLane?.cell ? criticMetrics(placeboLane.cell, "primary") : null;
  const teacherMetrics = teacherLane?.cell ? criticMetrics(teacherLane.cell, "primary") : null;
  const dmseDelta =
    focusMetrics && placeboMetrics ? focusMetrics.dmse - placeboMetrics.dmse : null;
  const placeboEqualsTeacher = Boolean(
    placeboMetrics &&
      teacherMetrics &&
      placeboLane?.cell?.cell_id !== teacherLane?.cell?.cell_id &&
      placeboMetrics.dmse === teacherMetrics.dmse &&
      placeboMetrics.cosine === teacherMetrics.cosine &&
      placeboMetrics.norm_ratio === teacherMetrics.norm_ratio,
  );
  return (
    <Panel
      title="Control rack — lanes travel together"
      span={12}
      badges={<Badge status="qualified" label="stored-snapshot" />}
      sub={
        chip === TEACHER_CHIP
          ? "Identity-only view: the teacher lane is the whole selection. Pick a chip in the move picker to load its edit lane with both controls."
          : `Selection ${chip} @ dose ${dose}: the edit lane never appears without its paraphrase placebo, random edit, and identity partners. Directional metrics are stored-snapshot evidence; texts are learned descriptions of the stored activation.`
      }
    >
      {focusLane?.cell && focusMetrics ? (
        <div className="bench-quickread" aria-label="Selected intervention quick read">
          <div className="bench-quickread-copy">
            <strong>Read this selection</strong>
            <span>
              These values describe movement, not success. A semantic effect is credible only when
              the edit differs from its placebo and random controls in the consequence panels.
            </span>
          </div>
          <div className="bench-quickread-metrics">
            <span>
              <b>{fmt(focusMetrics.dmse, 3)}</b>
              <small>edit dMSE</small>
            </span>
            <span>
              <b>{dmseDelta === null ? "—" : fmtSigned(dmseDelta, 3)}</b>
              <small>Δ vs placebo</small>
            </span>
            <span>
              <b>{fmt(focusLane.cell.behavior.js_divergence, 3)}</b>
              <small>JS divergence</small>
            </span>
            <span>
              <b>{fmt(focusLane.cell.behavior.top_10_overlap, 2)}</b>
              <small>top-10 overlap</small>
            </span>
          </div>
        </div>
      ) : null}
      <div className="bench-rack">
        {rack.map((laneCell) => (
          <LaneCard key={laneCell.lane} laneCell={laneCell} chip={chip} />
        ))}
      </div>
      {rack.some(
        (l) => l.lane !== "teacher" && l.cell && criticMetrics(l.cell, "independent") === null,
      ) ? (
        <p className="bench-note">
          The independent critic scored identity cells only; edit, placebo, and random rows show —
          for it by design.
        </p>
      ) : null}
      {placeboEqualsTeacher ? (
        <p className="bench-note">
          Note: the paraphrase-placebo and identity lanes are distinct stored cells but carry
          identical primary-critic metrics on this selection — their re-encodings coincide.
        </p>
      ) : null}
      <div className="bench-lane-foot bench-rack-foot">
        <LinkButton
          onClick={() => update({ station: "channel", row: rowId })}
          title="Open this row on the CHANNEL station"
        >
          view row on CHANNEL
        </LinkButton>
        <LinkButton
          onClick={() => update({ station: "audit", claim: "court" })}
          title="Court/verdict context: how paraphrase vs corruption verdicts are calibrated (AUDIT)"
        >
          court context (AUDIT)
        </LinkButton>
      </div>
    </Panel>
  );
}
