import { useMemo, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Link2, Search, X } from "lucide-react";
import type { StationProps } from "../../app/stationProps";
import type { ChannelShard, Critic, RowRecord } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { fmt, preview, shortRowId } from "../../data/format";
import { AuditLink } from "./lib";

interface Props {
  channel: ChannelShard;
  critic: Critic;
  rows: RowRecord[];
  selectedRow: RowRecord;
  update: StationProps["update"];
}

/** Coarse verbal anchor for where in the text the activation token sits. */
function positionWord(position: number, total: number): string {
  const frac = total > 1 ? position / (total - 1) : 1;
  if (frac >= 0.98) return "the final token";
  if (frac >= 0.8) return "near the end";
  if (frac >= 0.55) return "past the middle";
  if (frac >= 0.3) return "around the middle";
  return "near the start";
}

export default function RowReaderPanel({ channel, critic, rows, selectedRow, update }: Props) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const [query, setQuery] = useState("");
  const selectedIndex = Math.max(0, rows.findIndex((row) => row.row_id === selectedRow.row_id));
  const identity = channel.identity.find(
    (item) => item.row_id === selectedRow.row_id && item.critic === critic,
  );
  const retrieval = channel.retrieval.find(
    (item) => item.row_id === selectedRow.row_id && item.critic === critic,
  );
  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return rows;
    return rows.filter((row, index) =>
      [
        String(index + 1),
        row.row_id,
        row.doc_id,
        row.source_text,
        row.av_text,
      ].some((value) => value.toLowerCase().includes(normalized)),
    );
  }, [query, rows]);

  const goTo = (index: number) => {
    const next = rows[Math.min(rows.length - 1, Math.max(0, index))];
    if (next && next.row_id !== selectedRow.row_id) update({ row: next.row_id });
  };

  const openBrowser = () => {
    setQuery("");
    dialogRef.current?.showModal();
  };

  const chooseRow = (rowId: string) => {
    update({ row: rowId });
    dialogRef.current?.close();
  };

  return (
    <Panel
      id="chan-row-reader"
      title="Selected row: source → NLA description"
      span={12}
      badges={<Badge status="qualified" label="stored-snapshot" />}
      sub="One validation example at a time. The same selection drives every row-level CHANNEL panel below."
    >
      <div className="chan-row-browser-toolbar">
        <div className="chan-row-stepper" aria-label="Step through validation examples">
          <button
            type="button"
            onClick={() => goTo(selectedIndex - 1)}
            disabled={selectedIndex === 0}
            aria-label="Previous validation example"
            title="Previous validation example"
          >
            <ChevronLeft size={16} aria-hidden />
          </button>
          <strong>Example {selectedIndex + 1} of {rows.length}</strong>
          <button
            type="button"
            onClick={() => goTo(selectedIndex + 1)}
            disabled={selectedIndex === rows.length - 1}
            aria-label="Next validation example"
            title="Next validation example"
          >
            <ChevronRight size={16} aria-hidden />
          </button>
        </div>
        <button type="button" className="chan-row-browser-button" onClick={openBrowser}>
          <Search size={15} aria-hidden />
          <span>Browse all {rows.length} examples</span>
        </button>
        <span className="chan-row-browser-note">
          Row IDs are provenance keys, not semantic labels.
        </span>
      </div>

      <dialog
        ref={dialogRef}
        className="chan-row-dialog"
        aria-labelledby="chan-row-dialog-title"
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) event.currentTarget.close();
        }}
      >
        <div className="chan-row-dialog-shell">
          <header className="chan-row-dialog-head">
            <div>
              <span className="chan-row-pane-label">Validation panel</span>
              <h3 id="chan-row-dialog-title">Choose a source example</h3>
              <p>Search the full source text, NLA description, document key, or example number.</p>
            </div>
            <button
              type="button"
              className="chan-row-dialog-close"
              onClick={() => dialogRef.current?.close()}
              aria-label="Close example browser"
              title="Close example browser"
            >
              <X size={18} aria-hidden />
            </button>
          </header>
          <label className="chan-row-search">
            <Search size={16} aria-hidden />
            <span className="visually-hidden">Search validation examples</span>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search source text, NLA text, or row key"
            />
          </label>
          <div className="chan-row-search-status" role="status">
            {filteredRows.length} of {rows.length} examples
          </div>
          <div className="chan-row-gallery" aria-label="Validation source examples">
            {filteredRows.map((row) => {
              const index = rows.findIndex((candidate) => candidate.row_id === row.row_id);
              const selected = row.row_id === selectedRow.row_id;
              return (
                <button
                  key={row.row_id}
                  type="button"
                  className="chan-row-gallery-item"
                  aria-pressed={selected}
                  onClick={() => chooseRow(row.row_id)}
                  aria-label={`Example ${index + 1}: ${preview(row.source_text, 120)}`}
                >
                  <span className="chan-row-gallery-index">
                    Example {String(index + 1).padStart(2, "0")}
                    {selected ? <Check size={15} aria-hidden /> : null}
                  </span>
                  <strong>{preview(row.source_text, 170)}</strong>
                  <small>{row.doc_id} · activation position {row.token_position}</small>
                </button>
              );
            })}
          </div>
          {filteredRows.length === 0 ? (
            <UnavailableBox>No source examples match “{query}”.</UnavailableBox>
          ) : null}
        </div>
      </dialog>

      <div className="chan-row-reader">
        <section className="chan-row-pane chan-row-source" aria-label="Full original source text">
          <div className="chan-row-pane-label">1 · Original source text</div>
          <h4>Text associated with the stored R33 activation</h4>
          <div className="chan-row-source-text" tabIndex={0}>{selectedRow.source_text}</div>
          <p className="chan-pos-caption">
            Activation taken at <strong>token {selectedRow.token_position}</strong> of{" "}
            {selectedRow.n_raw_tokens} — {positionWord(selectedRow.token_position, selectedRow.n_raw_tokens)}{" "}
            of this text (marker below; exact character alignment is not shipped for panel rows).
          </p>
          <div
            className="chan-pos-track"
            role="img"
            aria-label={`Activation position marker: token ${selectedRow.token_position} of ${selectedRow.n_raw_tokens}`}
          >
            <span
              className="chan-pos-marker"
              style={{
                left: `${Math.min(99, Math.max(0, (selectedRow.token_position / Math.max(1, selectedRow.n_raw_tokens - 1)) * 100))}%`,
              }}
            />
          </div>
          <dl className="chan-row-metadata">
            <div><dt>example</dt><dd>{selectedIndex + 1} / {rows.length}</dd></div>
            <div><dt>activation position</dt><dd>{selectedRow.token_position} of {selectedRow.n_raw_tokens} raw tokens</dd></div>
            <div><dt>document</dt><dd>{selectedRow.doc_id}</dd></div>
            <div><dt>row key</dt><dd>{shortRowId(selectedRow.row_id)}</dd></div>
          </dl>
        </section>

        <section className="chan-row-pane chan-row-output" aria-label="NLA and teacher descriptions">
          <div className="chan-row-output-head">
            <div>
              <div className="chan-row-pane-label">2 · NLA learned description</div>
              <h4>Text emitted by AV from this stored activation</h4>
            </div>
            <span className="chan-row-critic">scored by {critic} AR critic</span>
          </div>
          <div className="chan-row-link">
            <Link2 size={16} aria-hidden />
            <span>stored activation → AV text → AR reconstruction</span>
          </div>
          <div className="text-block chan-row-av-text" tabIndex={0} aria-label="NLA learned description">
            {selectedRow.av_text}
          </div>
          <p className="chan-row-caveat">
            This is a learned activation encoding, not Nano30B&apos;s visible continuation and not a
            transcript of hidden chain-of-thought.
          </p>

          <div className="chan-row-metrics" aria-label={`${critic} critic reconstruction metrics`}>
            {identity ? (
              <>
                <div><span>direction error · dMSE</span><strong>{fmt(identity.dmse)}</strong><small>lower is better</small></div>
                <div><span>direction cosine</span><strong>{fmt(identity.cosine)}</strong><small>higher is better</small></div>
                <div><span>norm ratio</span><strong>{fmt(identity.norm_ratio)}</strong><small>1.0 matches magnitude</small></div>
              </>
            ) : (
              <UnavailableBox>No identity reconstruction metric exists for this row and critic.</UnavailableBox>
            )}
            {retrieval ? (
              <div>
                <span>identity retrieval</span>
                <strong>rank {retrieval.rank}</strong>
                <small>
                  nearest: {shortRowId(retrieval.nearest_row_id)}
                  {retrieval.rank === 1 ? " (itself — a correct match)" : ""}
                </small>
              </div>
            ) : null}
          </div>
          <p className="chan-note">
            The qualified claim covers direction only —{" "}
            <AuditLink claim="magnitude" update={update} label="magnitude card" /> explains why norm
            ratio is reported but not claimed.
          </p>

          <details className="chan-row-teacher">
            <summary>Teacher reference used to supervise the language channel</summary>
            <div className="text-block" tabIndex={0}>{selectedRow.teacher_text}</div>
          </details>
        </section>
      </div>
    </Panel>
  );
}
