/**
 * Negative & weak results — first-class panel. Statements render verbatim;
 * nothing here is softened, filtered, or hidden behind a disclosure.
 */

import type { NegativeResult } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";

function statusBadge(status: NegativeResult["status"]) {
  if (status === "weak") return <Badge status="exploratory" label="weak" />;
  if (status === "negative") return <Badge status="negative" />;
  return <Badge status="caveat" />;
}

export default function NegativeResultsPanel(props: { results: NegativeResult[] }) {
  return (
    <Panel
      id="negative_results"
      span={6}
      title="Negative & weak results"
      sub="Kept next to the headlines on purpose: these statements bound every claim above."
      badges={<Badge status="negative" label={`${props.results.length} recorded`} />}
    >
      {props.results.length === 0 ? (
        <UnavailableBox>The bundle records no negative or weak results.</UnavailableBox>
      ) : (
        <div className="audit-table-scroll">
          <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th scope="col">status</th>
                <th scope="col">statement (verbatim)</th>
                <th scope="col">source</th>
              </tr>
            </thead>
            <tbody>
              {props.results.map((r) => (
                <tr key={r.id}>
                  <td>{statusBadge(r.status)}</td>
                  <td className="audit-statement">{r.statement}</td>
                  <td>
                    <span className="audit-chip">{r.source}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
