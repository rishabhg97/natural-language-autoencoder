/**
 * Magnitude card — why the qualified channel claim is directional. The
 * post-hoc magnitude calibration was fit on validation only; its claim
 * boundary renders verbatim and the fit block is read defensively
 * (Record<string, unknown> by contract — render what exists, invent nothing).
 */

import { useMemo } from "react";
import type { AuditShard } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { fmt } from "../../data/format";
import { KvList, asNumber, asRecord, asString, humanize } from "./util";

const METRIC_COLUMNS = ["cosine_mean", "directional_mse", "centered_r2"] as const;

export default function MagnitudePanel(props: { mag: AuditShard["magnitude"] }) {
  const { mag } = props;
  const candidates = useMemo(() => asRecord(mag.fit.candidate_metrics), [mag.fit]);
  const selectedMethod = asString(mag.fit.selected_method);
  const candidateKeys = useMemo(
    () => (candidates ? Object.keys(candidates).sort() : []),
    [candidates],
  );

  return (
    <Panel
      id="magnitude"
      span={6}
      title="Magnitude card"
      sub="Why the qualified claim is directional: reconstructing the activation's direction is qualified; recovering its magnitude is a separate, post-hoc calibration."
      badges={<Badge status="exploratory" label={humanize(mag.publication_status)} />}
    >
      <p className="audit-boundary">{mag.claim_boundary}</p>

      <KvList record={mag.fit} skip={["candidate_metrics", "candidate_parameters"]} />

      <h4 className="audit-subhead">Candidate magnitude calibrations</h4>
      {candidates === null || candidateKeys.length === 0 ? (
        <UnavailableBox>The fit block carries no candidate metrics.</UnavailableBox>
      ) : (
        <div className="audit-table-scroll">
          <table className="data-table" tabIndex={0} aria-label="Activation magnitude diagnostics">
            <thead>
              <tr>
                <th scope="col">candidate</th>
                {METRIC_COLUMNS.map((c) => (
                  <th scope="col" className="num" key={c}>
                    {humanize(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {candidateKeys.map((k) => {
                const rec = asRecord(candidates[k]);
                const isSelected = selectedMethod === k;
                return (
                  <tr key={k} aria-selected={isSelected}>
                    <td>
                      <span className="mono">{k}</span>{" "}
                      {isSelected ? <Badge status="selected" label="selected" /> : null}
                    </td>
                    {METRIC_COLUMNS.map((c) => {
                      const v = rec ? asNumber(rec[c]) : null;
                      return (
                        <td className="num" key={c}>
                          {v === null ? "—" : fmt(v, 4)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
