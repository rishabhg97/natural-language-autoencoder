/**
 * Panel 3e — stored greedy continuations, baseline vs patched, with a
 * word-level diff. Both texts were precomputed and stored with the bundle;
 * nothing is generated at view time.
 */

import { useMemo } from "react";
import type { AppState } from "../../app/urlState";
import { Badge, Panel, Segmented, UnavailableBox } from "../../components/ui";
import { LANE_META, type LaneCell, type LaneKey } from "./laneModel";
import { splitWords, wordDiff } from "./diff";

function DiffBlock(props: {
  words: string[];
  marks: ("same" | "changed")[];
  changedClass: string;
  changedTitle: string;
  label: string;
}) {
  return (
    <div className="text-block" aria-label={props.label}>
      {props.words.map((w, i) => (
        <span key={`${i}-${w}`}>
          {props.marks[i] === "changed" ? (
            <span className={props.changedClass} title={props.changedTitle}>
              {w}
            </span>
          ) : (
            w
          )}{" "}
        </span>
      ))}
    </div>
  );
}

export default function ContinuationsPanel(props: {
  rack: LaneCell[];
  focusLane: LaneKey;
  update: (patch: Partial<AppState>) => void;
}) {
  const { rack, focusLane, update } = props;
  const availableLanes = rack.filter((l) => l.cell !== null).map((l) => l.lane);
  const focus = rack.find((l) => l.lane === focusLane && l.cell) ?? rack.find((l) => l.cell);
  const cell = focus?.cell ?? null;

  const diff = useMemo(() => {
    if (!cell) return null;
    const a = splitWords(cell.baseline_continuation);
    const b = splitWords(cell.patched_continuation);
    const marks = wordDiff(a, b);
    const shared =
      marks.a.filter((m) => m === "same").length / Math.max(1, marks.a.length);
    // When the outputs share almost nothing, per-word marks become stripes;
    // show plain text with an explicit statement instead.
    return { a, b, marks, sparse: shared < 0.15 };
  }, [cell]);

  return (
    <Panel
      title="Continuations — baseline vs patched"
      span={8}
      badges={<Badge status="exploratory" label="functional: validation-only" />}
      sub="Precomputed greedy continuations: baseline runs on the stored activation; patched substitutes the focused lane's re-encoded description at the boundary. Lane focus is shared with the top-k pane."
    >
      <div className="bench-controls">
        <Segmented
          options={availableLanes}
          value={availableLanes.includes(focusLane) ? focusLane : availableLanes[0] ?? "edit"}
          onChange={(v) => update({ cell: v })}
          label="focused lane (continuations)"
          format={(l) => LANE_META[l].short}
        />
      </div>
      {!cell || !diff ? (
        <UnavailableBox>no stored continuation pair for this selection.</UnavailableBox>
      ) : (
        <>
          {diff.sparse ? (
            <p className="bench-note">
              These outputs share almost no words, so the per-word diff is suppressed — read them
              as two different continuations rather than an edit of one.
            </p>
          ) : null}
          <div className="bench-cont-pair">
            <div>
              <p className="bench-cont-label">
                baseline (stored activation)
                {diff.sparse ? "" : " — words "}
                {diff.sparse ? "" : <s>struck</s>}
                {diff.sparse ? "" : " appear only here"}
              </p>
              <DiffBlock
                words={diff.a}
                marks={diff.sparse ? diff.a.map(() => "same" as const) : diff.marks.a}
                changedClass="bench-diff-removed"
                changedTitle="word only in the baseline continuation"
                label="baseline continuation"
              />
            </div>
            <div>
              <p className="bench-cont-label">
                patched ({LANE_META[focus!.lane].label} encoding)
                {diff.sparse ? "" : " — "}
                {diff.sparse ? "" : <u>underlined</u>}
                {diff.sparse ? "" : " words are new"}
              </p>
              <DiffBlock
                words={diff.b}
                marks={diff.sparse ? diff.b.map(() => "same" as const) : diff.marks.b}
                changedClass="bench-diff-added"
                changedTitle="word only in the patched continuation"
                label="patched continuation"
              />
            </div>
          </div>
          <div className="bench-kv-wrap">
            <p className="bench-cont-label">
              generation protocol (stored with the cell)
              <Badge status="qualified" label="stored-snapshot" />
            </p>
            <dl className="kv-list">
              {Object.entries(cell.generation_protocol).map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <dt>{k}</dt>
                  <dd className="mono">{String(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </>
      )}
    </Panel>
  );
}
