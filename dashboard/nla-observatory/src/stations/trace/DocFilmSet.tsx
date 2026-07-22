/**
 * Mode A — the document film set: fresh-forward learned descriptions sampled
 * along documents, with a client-side persistence readout between adjacent
 * positions. Every panel is fresh-forward exploratory evidence.
 */
import { useMemo, useRef, type MutableRefObject, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Link2 } from "lucide-react";
import type { StationProps } from "../../app/stationProps";
import type { RowsShard, TraceDoc, TracePosition, TraceShard } from "../../data/types";
import { Panel, SelectControl, UnavailableBox } from "../../components/ui";
import { fmt, shortRowId } from "../../data/format";
import { jaccardSimilarity, wordDiff } from "./textUtils";
import PersistenceChart, { type Transition } from "./PersistenceChart";
import { FreshBadge } from "./common";

interface Props extends StationProps {
  trace: TraceShard;
  rows: RowsShard;
  onEnterPoetry: () => void;
}

const EMPTY_POSITIONS: TracePosition[] = [];

function displayToken(text: string): string {
  const shown = text.replace(/\n/g, "↵");
  return shown === "" ? "·" : shown;
}

/** Humanize builder parse-state enums (e.g. usable_closed → "parsed cleanly"). */
function displayParseState(state: string): string {
  const map: Record<string, string> = {
    usable_closed: "parsed cleanly (closed)",
    usable_open: "parsed (unclosed tail)",
    unusable: "did not parse",
  };
  return map[state] ?? state.replace(/_/g, " ");
}

interface DocumentSourceSelectorProps {
  sourceText: string;
  positions: TracePosition[];
  selectedPosition: number;
  onSelect: (position: number) => void;
  tokenRefs: MutableRefObject<(HTMLButtonElement | null)[]>;
}

function DocumentSourceSelector({
  sourceText,
  positions,
  selectedPosition,
  onSelect,
  tokenRefs,
}: DocumentSourceSelectorProps) {
  const aligned = positions
    .map((position, index) => ({ position, index }))
    .filter(
      ({ position }) =>
        position.source_alignment === "exact" &&
        position.source_char_start !== null &&
        position.source_char_end !== null,
    )
    .sort(
      (a, b) =>
        (a.position.source_char_start ?? 0) - (b.position.source_char_start ?? 0),
    );

  const content: ReactNode[] = [];
  let cursor = 0;
  for (const { position, index } of aligned) {
    const start = position.source_char_start ?? cursor;
    const end = position.source_char_end ?? start;
    if (start > cursor) content.push(sourceText.slice(cursor, start));
    content.push(
      <button
        key={position.position}
        ref={(element) => {
          tokenRefs.current[index] = element;
        }}
        type="button"
        className={`trc-inline-source-token${position.usable ? "" : " trc-inline-source-token-unusable"}`}
        aria-pressed={position.position === selectedPosition}
        onClick={() => onSelect(position.position)}
        title={`Open NLA readout for token “${displayToken(position.token_text)}” at position ${position.position}`}
      >
        {sourceText.slice(start, end)}
      </button>,
    );
    cursor = Math.max(cursor, end);
  }
  if (cursor < sourceText.length) content.push(sourceText.slice(cursor));

  return (
    <div
      className="trc-full-source"
      tabIndex={0}
      aria-label="Full original document with selectable NLA sample positions"
    >
      {content}
    </div>
  );
}

export default function DocFilmSet({ trace, rows, state, update, onEnterPoetry }: Props) {
  const docs = trace.docs;
  const doc: TraceDoc | undefined = docs.find((d) => d.row_id === state.row) ?? docs[0];

  const positions = doc ? doc.positions : EMPTY_POSITIONS;
  const selIdxRaw = positions.findIndex((p) => p.position === state.position);
  // Default = first position; derived only, so the URL is untouched until the user acts.
  const selIdx = selIdxRaw >= 0 ? selIdxRaw : 0;
  const selected: TracePosition | undefined = positions[selIdx];

  const transitions = useMemo<Transition[]>(
    () =>
      positions.slice(1).map((to, i) => ({
        from: positions[i],
        to,
        similarity: jaccardSimilarity(positions[i].description, to.description),
      })),
    [positions],
  );
  const activeTransitionIdx = selIdx > 0 ? selIdx - 1 : null;
  const activeTransition = activeTransitionIdx !== null ? transitions[activeTransitionIdx] : null;
  const diff = useMemo(
    () =>
      activeTransition
        ? wordDiff(activeTransition.from.description, activeTransition.to.description)
        : null,
    [activeTransition],
  );

  const rowRecord = useMemo(
    () => (doc ? (rows.rows.find((r) => r.row_id === doc.row_id) ?? null) : null),
    [rows, doc],
  );

  const chipRefs = useRef<(HTMLButtonElement | null)[]>([]);

  if (!doc || !selected) {
    return <UnavailableBox>trace.json contains no documents — nothing to show for the film set.</UnavailableBox>;
  }

  const goTo = (index: number) => {
    const next = Math.min(positions.length - 1, Math.max(0, index));
    if (next !== selIdx) {
      update({ position: positions[next].position });
      requestAnimationFrame(() => chipRefs.current[next]?.focus());
    }
  };

  const onQualifiedPanel = rowRecord !== null;

  return (
    <div className="panel-grid">
      <Panel
        title="Document token-linked NLA reader"
        span={12}
        badges={<FreshBadge scope={trace.claim_scope} />}
        sub={`${docs.length} fresh-forward documents · ${positions.length} sampled positions in this document · ${trace.source_alignment.exact_positions}/400 trajectory positions have exact released-text alignment`}
      >
        <div className="controls-row">
          <SelectControl
            label="Document"
            value={doc.row_id}
            options={docs.map((d) => ({
              value: d.row_id,
              label: `${d.doc_id} · ${shortRowId(d.row_id)}`,
            }))}
            onChange={(v) => update({ row: v, position: null })}
          />
          <span className="trc-doc-meta mono">
            {doc.doc_id} · {shortRowId(doc.row_id)} · {doc.content_family_id}
          </span>
        </div>
        {state.row !== null && !docs.some((d) => d.row_id === state.row) ? (
          <p className="panel-sub" role="note">
            The selected row {shortRowId(state.row)} has no trace document — the film set covers{" "}
            {docs.length} of the 50 panel rows. Showing {shortRowId(doc.row_id)} instead.
          </p>
        ) : null}

        <div className="trc-document-split">
          <section className="trc-linked-pane trc-source-pane" aria-label="Full original document">
            <div className="trc-pane-label">1 · Full original document</div>
            <h4>Select a highlighted token to inspect its NLA readout</h4>
            {rowRecord && selected.source_alignment === "exact" ? (
              <>
                <DocumentSourceSelector
                  sourceText={rowRecord.source_text}
                  positions={positions}
                  selectedPosition={selected.position}
                  onSelect={(position) => update({ position })}
                  tokenRefs={chipRefs}
                />
                <div className="trc-source-selector-legend">
                  <span><i className="trc-legend-available" /> NLA readout available</span>
                  <span><i className="trc-legend-selected" /> selected token</span>
                  <span>Plain text was not sampled in this dashboard.</span>
                  <span>Tokens are model tokens — words and punctuation can split.</span>
                </div>
              </>
            ) : (
              <>
                {rowRecord ? (
                  <div className="trc-full-source" tabIndex={0}>{rowRecord.source_text}</div>
                ) : null}
                <UnavailableBox>
                  This trajectory&apos;s token IDs do not align with the released row text, so no
                  clickable token is drawn onto guessed text. Use the previous and next controls on
                  the right to inspect its stored trajectory readouts.
                </UnavailableBox>
              </>
            )}
          </section>
          <section className="trc-linked-pane trc-output-pane" aria-label="NLA verbalization">
            <div className="trc-selected-output-head">
              <div>
                <div className="trc-pane-label">2 · NLA output for selected token</div>
                <h4>“{displayToken(selected.token_text)}” <span>· position {selected.position}</span></h4>
              </div>
              <div className="trc-step-buttons" aria-label="Step through sampled tokens">
                <button type="button" onClick={() => goTo(selIdx - 1)} disabled={selIdx === 0} aria-label="Previous sampled token" title="Previous sampled token">
                  <ChevronLeft size={15} aria-hidden />
                </button>
                <span>{selIdx + 1} / {positions.length}</span>
                <button type="button" onClick={() => goTo(selIdx + 1)} disabled={selIdx === positions.length - 1} aria-label="Next sampled token" title="Next sampled token">
                  <ChevronRight size={15} aria-hidden />
                </button>
              </div>
            </div>
            <div className="trc-activation-bridge">
              <Link2 size={16} aria-hidden />
              <span>activation at the selected source token → AV verbalization</span>
            </div>
            <div className="trc-output-meta">
              <span>{selected.n_context_tokens} context tokens</span>
              <span title={`parse_state: ${selected.parse_state}`}>
                {displayParseState(selected.parse_state)}
              </span>
              <span>{selected.usable ? "usable output" : "unusable output"}</span>
            </div>
            <div className="text-block trc-desc-text" tabIndex={0} aria-label="Learned activation description">
              {selected.description}
            </div>
            <p className="trc-output-caveat">
              This is the AV model&apos;s learned text encoding of the activation, not text written by
              Nano30B and not a transcript of hidden chain-of-thought.
            </p>
          </section>
        </div>
        {!rowRecord ? (
          <UnavailableBox>
            this document&apos;s row is not in rows.json — no qualified panel text context is
            available for it.
          </UnavailableBox>
        ) : null}
      </Panel>

      <Panel
        title="Persistence of adjacent descriptions"
        span={12}
        badges={<FreshBadge scope={trace.claim_scope} />}
        sub="Jaccard word overlap of adjacent learned descriptions (computed in the browser from the displayed texts) — persistence of the encoding, not of a belief. No chance baseline ships in this bundle; compare transitions with each other."
      >
        {transitions.length === 0 ? (
          <UnavailableBox>fewer than two sampled positions — no transitions to compare.</UnavailableBox>
        ) : (
          <>
            <PersistenceChart
              transitions={transitions}
              selectedIndex={activeTransitionIdx}
              onSelect={(i) => {
                const t = transitions[i];
                if (t) update({ position: t.to.position });
              }}
            />
            {activeTransition && diff ? (
              <div className="trc-diff">
                <div className="trc-diff-cols">
                  <div>
                    <h4 className="trc-h4">
                      position {activeTransition.from.position} · “
                      {displayToken(activeTransition.from.token_text)}”
                    </h4>
                    <div className="text-block trc-diff-text" tabIndex={0}>{activeTransition.from.description}</div>
                  </div>
                  <div>
                    <h4 className="trc-h4">
                      position {activeTransition.to.position} · “
                      {displayToken(activeTransition.to.token_text)}”
                    </h4>
                    <div className="text-block trc-diff-text" tabIndex={0}>{activeTransition.to.description}</div>
                  </div>
                </div>
                <div className="trc-diff-words">
                  <span className="trc-diff-kind">removed ({diff.removed.length}):</span>
                  {diff.removed.length > 0 ? (
                    diff.removed.map((w) => (
                      <del key={w} title="word removed between these descriptions">
                        − {w}
                      </del>
                    ))
                  ) : (
                    <span>none</span>
                  )}
                </div>
                <div className="trc-diff-words">
                  <span className="trc-diff-kind">added ({diff.added.length}):</span>
                  {diff.added.length > 0 ? (
                    diff.added.map((w) => (
                      <ins key={w} title="word added between these descriptions">
                        + {w}
                      </ins>
                    ))
                  ) : (
                    <span>none</span>
                  )}
                </div>
              </div>
            ) : (
              <p className="trc-note">
                First sampled position selected — click a transition bar (or step right) to see its
                word-level diff.
              </p>
            )}
            <details className="trc-details">
              <summary>transition table (exact similarity values)</summary>
              <div className="trc-table-scroll">
                <table className="data-table" tabIndex={0}>
                  <thead>
                    <tr>
                      <th>transition</th>
                      <th className="num">similarity</th>
                      <th>
                        <span className="visually-hidden">select</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {transitions.map((t, i) => (
                      <tr key={t.to.position} aria-selected={i === activeTransitionIdx}>
                        <td className="mono">
                          {t.from.position} → {t.to.position}
                        </td>
                        <td className="num">{fmt(t.similarity, 3)}</td>
                        <td>
                          <button
                            type="button"
                            className="trc-linkbtn"
                            onClick={() => update({ position: t.to.position })}
                          >
                            view diff
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </>
        )}
      </Panel>

      <Panel
        title="Real-vs-shuffled control"
        span={4}
        badges={<FreshBadge scope={trace.claim_scope} />}
        sub="control lane — its absence here is made visible, never silently dropped"
      >
        {!trace.shuffled_control.available ? (
          <>
            <UnavailableBox>{trace.shuffled_control.note}</UnavailableBox>
            <p className="trc-note">the poetry lens carries the real-vs-shuffled control</p>
            <button type="button" className="trc-button" onClick={onEnterPoetry}>
              Open the poetry planning lens
            </button>
          </>
        ) : (
          <p className="trc-note">{trace.shuffled_control.note}</p>
        )}
      </Panel>

      <Panel
        title="Fresh-forward drift"
        span={4}
        badges={<FreshBadge scope={trace.claim_scope} />}
        sub="stored-vs-fresh final-position agreement for this document (E5)"
      >
        <dl className="kv-list">
          <dt>1 − cos</dt>
          <dd>{fmt(doc.drift.one_minus_cos, 4)}</dd>
          <dt>relative L2</dt>
          <dd>{fmt(doc.drift.relative_l2, 4)}</dd>
          <dt>RMS ratio</dt>
          <dd>{fmt(doc.drift.rms_ratio, 4)}</dd>
          <dt>max |Δ|</dt>
          <dd>{fmt(doc.drift.max_abs, 4)}</dd>
        </dl>
        <button
          type="button"
          className="trc-audit-link"
          onClick={() => update({ station: "audit", claim: "drift" })}
          title="Open the drift claim in the AUDIT station"
        >
          drift audit ↗
        </button>
      </Panel>

      <Panel
        title="Cross-station: BENCH"
        span={4}
        badges={<FreshBadge scope={trace.claim_scope} />}
        sub="jump to the precomputed intervention grid for this row"
      >
        {onQualifiedPanel ? (
          <>
            <p className="trc-note">
              {shortRowId(doc.row_id)} is one of the {rows.rows.length} qualified panel rows.
              BEHAVIOR-depth cells exist only for a subset of rows — BENCH resolves what is
              available.
            </p>
            <button
              type="button"
              className="trc-button"
              onClick={() => update({ station: "bench", row: doc.row_id })}
            >
              Open this row on BENCH ↗
            </button>
          </>
        ) : (
          <UnavailableBox>
            {shortRowId(doc.row_id)} is not one of the {rows.rows.length} qualified panel rows;
            BENCH has no page for it.
          </UnavailableBox>
        )}
      </Panel>
    </div>
  );
}
