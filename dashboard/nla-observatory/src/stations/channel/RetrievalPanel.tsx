/**
 * RETRIEVAL — self-retrieval by expected cosine per critic. Misses (rank > 1)
 * are negative results and stay visible; clicking a miss selects the row.
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import type { ChannelShard, Critic } from "../../data/types";
import { Badge, Panel } from "../../components/ui";
import { fmt, fmtPct, shortRowId } from "../../data/format";
import { AuditLink, StatTile } from "./lib";

export default function RetrievalPanel(props: {
  channel: ChannelShard;
  critic: Critic;
  selectedRowId: string;
  update: (patch: Partial<AppState>) => void;
}) {
  const { critic } = props;
  const rows = useMemo(
    () => props.channel.retrieval.filter((r) => r.critic === critic),
    [props.channel.retrieval, critic],
  );
  const misses = useMemo(
    () => rows.filter((r) => r.rank !== 1).sort((a, b) => b.rank - a.rank),
    [rows],
  );
  const hits = rows.length - misses.length;

  return (
    <Panel
      id="chan-retrieval"
      title="Retrieval"
      span={4}
      badges={
        <>
          <Badge status="qualified" label="stored-snapshot" />
          {misses.length > 0 ? (
            <Badge
              status="negative"
              label={`${misses.length} miss${misses.length === 1 ? "" : "es"}`}
              title="Rows whose nearest stored target is not their own (negative result, kept visible)"
            />
          ) : null}
          <AuditLink claim="stored_snapshot_channel" update={props.update} />
        </>
      }
      sub={
        <>
          Self-retrieval under the <strong>{critic}</strong> critic: rank 1 means the reconstruction
          from the learned description is nearest (by expected cosine) to its own stored activation.
        </>
      }
    >
      {rows.length === 0 ? (
        <div className="state-box unavailable">
          <Badge status="unavailable" /> no retrieval records for the {critic} critic in channel.json.
        </div>
      ) : (
        <>
          <div className="chan-tiles">
            <StatTile label="top-1 hit rate" value={fmtPct(hits / rows.length)} sub={`${critic} critic`} />
            <StatTile label="hits (rank 1)" value={`${hits}/${rows.length}`} />
            <StatTile
              label="misses"
              value={String(misses.length)}
              negative={misses.length > 0}
              sub={misses.length > 0 ? "kept visible below" : "none"}
            />
          </div>
          {misses.length === 0 ? (
            <p className="chan-note">
              All {rows.length} rows retrieve their own stored activation at rank 1 under the {critic}{" "}
              critic.
            </p>
          ) : (
            <div className="chan-table-wrap">
              <table className="data-table" tabIndex={0}>
                <caption className="visually-hidden">
                  Retrieval misses under the {critic} critic; selecting a row updates every panel
                </caption>
                <thead>
                  <tr>
                    <th scope="col">row</th>
                    <th scope="col" className="num">rank</th>
                    <th scope="col">nearest row</th>
                    <th scope="col" className="num">expected cos</th>
                  </tr>
                </thead>
                <tbody>
                  {misses.map((m) => (
                    <tr
                      key={m.row_id}
                      className="selectable"
                      aria-selected={m.row_id === props.selectedRowId}
                    >
                      <td>
                        <button
                          type="button"
                          className="chan-audit-link"
                          onClick={() => props.update({ row: m.row_id })}
                          title={`Select ${m.row_id} across all panels`}
                        >
                          {shortRowId(m.row_id)}
                        </button>
                      </td>
                      <td className="num">{m.rank}</td>
                      <td className="mono" title={m.nearest_row_id}>
                        {shortRowId(m.nearest_row_id)}
                      </td>
                      <td className="num">{fmt(m.expected_cosine)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}
