/**
 * Fresh-vs-stored drift — how far fresh forward-pass activations sit from the
 * stored snapshots the qualified claim is made on. The card is rendered
 * generically from the shard (Record<string, unknown> by contract) and its
 * publication_ready=false flag is surfaced as a caveat, never hidden.
 */

import { useMemo } from "react";
import type { StationProps } from "../../app/stationProps";
import type { AuditShard } from "../../data/types";
import { Badge, HashChip, Panel, UnavailableBox } from "../../components/ui";
import { fmt, shortRowId } from "../../data/format";
import { KvList, asRecord, asString, humanize } from "./util";

export default function DriftPanel(props: { drift: AuditShard["drift"] } & StationProps) {
  const card = props.drift.card;
  const fidelity = asRecord(card.activation_fidelity);
  const primaryAssessment = asString(card.primary_fidelity_assessment);
  const sha = asString(card.sha256);
  const path = asString(card.path);

  const e5Sorted = useMemo(
    () => props.drift.e5_per_doc.slice().sort((a, b) => b.one_minus_cos - a.one_minus_cos),
    [props.drift.e5_per_doc],
  );

  return (
    <Panel
      id="drift"
      span={6}
      title="Fresh-vs-stored drift"
      sub="Agreement between fresh forward-pass activations and the stored snapshots. Bounds every fresh-forward view."
      badges={
        <>
          <Badge status="exploratory" label="fresh-forward" />
          {card.publication_ready === false ? (
            <Badge status="caveat" label="not publication-ready" />
          ) : null}
          {sha ? <HashChip hash={sha} label="card" /> : null}
        </>
      }
    >
      {fidelity === null ? (
        <UnavailableBox>The drift card carries no activation_fidelity block.</UnavailableBox>
      ) : (
        Object.entries(fidelity).map(([name, value]) => {
          const rec = asRecord(value);
          if (rec === null) return null;
          return (
            <div key={name} className="audit-spaced">
              <h4 className="audit-subhead">
                {humanize(name)}
                {primaryAssessment === name ? (
                  <Badge status="selected" label="primary assessment" />
                ) : null}
              </h4>
              <KvList record={rec} />
              {Object.entries(rec)
                .filter(([, v]) => asRecord(v) !== null)
                .map(([sub, v]) => (
                  <details key={sub} className="audit-details">
                    <summary>{humanize(sub)}</summary>
                    <KvList record={asRecord(v)!} />
                  </details>
                ))}
            </div>
          );
        })
      )}
      <KvList record={card} skip={["activation_fidelity", "path", "sha256"]} />
      {path ? <p className="audit-path">{path}</p> : null}

      <h4 className="audit-subhead">E5 per-document drift (worst first)</h4>
      {e5Sorted.length === 0 ? (
        <UnavailableBox>No per-document drift rows in this bundle.</UnavailableBox>
      ) : (
        <div className="audit-table-scroll">
          <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th scope="col">doc row</th>
                <th scope="col" className="num">
                  1 − cos
                </th>
                <th scope="col" className="num">
                  relative L2
                </th>
              </tr>
            </thead>
            <tbody>
              {e5Sorted.map((d) => (
                <tr key={d.row_id}>
                  <td>
                    <button
                      type="button"
                      className="audit-rowlink"
                      onClick={() => props.update({ station: "trace", row: d.row_id })}
                      title={`Open ${d.row_id} in the TRACE station`}
                    >
                      {shortRowId(d.row_id)}
                    </button>
                  </td>
                  <td className="num">{fmt(d.one_minus_cos, 5)}</td>
                  <td className="num">{fmt(d.relative_l2, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
