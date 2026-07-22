/**
 * Panel 4 — session ledger. Append-only local record of every precomputed
 * experiment opened in this browser. Stores selection tuples + cell ids only
 * (no evidence values); the only way data leaves is the user copying it.
 */

import { useState } from "react";
import { Panel } from "../../components/ui";
import { shortRowId } from "../../data/format";
import { chipLabel } from "./laneModel";
import { LinkButton } from "./benchUi";
import type { LedgerEntry } from "./useLedger";

export default function LedgerPanel(props: { entries: LedgerEntry[]; onClear: () => void }) {
  const { entries, onClear } = props;
  const [copied, setCopied] = useState(false);
  const newestFirst = [...entries].reverse();

  return (
    <Panel
      title={`Session ledger — ${entries.length} experiment${entries.length === 1 ? "" : "s"} opened`}
      span={12}
      sub="Append-only record of every (row, chip, dose, cell-id set) opened here. Local to this browser; the only way data leaves is copying it yourself. Entries are never auto-deleted."
    >
      <div className="bench-ledger-actions">
        <LinkButton
          onClick={() => {
            void navigator.clipboard
              ?.writeText(JSON.stringify(entries, null, 2))
              .then(() => {
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1200);
              });
          }}
          title="Copy the full ledger to the clipboard as JSON"
        >
          {copied ? "copied ✓" : "copy as JSON"}
        </LinkButton>
        <LinkButton
          onClick={() => {
            if (
              window.confirm(
                "Clear the local bench ledger? This deletes the append-only record in this browser only.",
              )
            ) {
              onClear();
            }
          }}
          title="Explicitly clear the ledger (asks for confirmation)"
        >
          clear ledger
        </LinkButton>
      </div>
      {entries.length === 0 ? (
        <p className="bench-note">
          nothing recorded yet — opening a behavior selection appends an entry.
        </p>
      ) : (
        <div className="bench-scroll">
          <table className="data-table" tabIndex={0} aria-label="session ledger, newest first">
            <thead>
              <tr>
                <th scope="col">opened at</th>
                <th scope="col">row</th>
                <th scope="col">chip</th>
                <th scope="col">dose</th>
                <th scope="col">cell ids (lane → cell)</th>
              </tr>
            </thead>
            <tbody>
              {newestFirst.map((entry, i) => (
                <tr key={`${entry.ts}-${i}`}>
                  <td className="mono">{new Date(entry.ts).toLocaleString()}</td>
                  <td className="mono">{shortRowId(entry.row)}</td>
                  <td>{chipLabel(entry.chip)}</td>
                  <td className="num">{entry.dose ?? "—"}</td>
                  <td className="bench-ledger-cells">
                    {Object.entries(entry.cells)
                      .map(([lane, id]) => `${lane}: ${id}`)
                      .join(" · ")}
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
