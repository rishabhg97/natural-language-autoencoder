/**
 * TRACE station — "what changes across tokens".
 * Mode A: document film set (fresh-forward learned descriptions along documents).
 * Mode B: poetry planning lens (real-vs-shuffled control + causal edit bench),
 * active whenever a poetry case is selected in the URL state.
 *
 * Everything in this station is fresh-forward exploratory evidence — outside
 * the qualified stored-snapshot channel claim (deep link: AUDIT claim
 * 'fresh_forward_trace').
 */
import { useEffect, useRef } from "react";
import type { StationProps } from "../../app/stationProps";
import { useShard } from "../../data/loader";
import type { PoetryShard, RowsShard, TraceShard } from "../../data/types";
import { Badge, ErrorBox, LoadingBox, Segmented, StationBrief } from "../../components/ui";
import { fmt, fmtPct } from "../../data/format";
import DocFilmSet from "./DocFilmSet";
import PoetryLens from "./PoetryLens";
import "./trace.css";

const MODES = ["Document film set", "Poetry planning lens"] as const;

export default function TraceStation({ state, update }: StationProps) {
  const trace = useShard<TraceShard>("trace.json");
  const poetry = useShard<PoetryShard>("poetry.json");
  const rows = useShard<RowsShard>("rows.json");

  // Remember the last poetry case within this session so toggling the lens
  // restores it; a URL-restored case (state.poetryCase) always wins.
  const lastCase = useRef<string | null>(null);
  useEffect(() => {
    if (state.poetryCase) lastCase.current = state.poetryCase;
  }, [state.poetryCase]);

  if (trace.status === "error") return <ErrorBox message={`trace.json — ${trace.message}`} />;
  if (poetry.status === "error") return <ErrorBox message={`poetry.json — ${poetry.message}`} />;
  if (rows.status === "error") return <ErrorBox message={`rows.json — ${rows.message}`} />;
  if (trace.status === "loading" || poetry.status === "loading" || rows.status === "loading") {
    return <LoadingBox what="TRACE evidence (trace.json · poetry.json · rows.json)" />;
  }

  const poetryActive = state.poetryCase !== null;
  const tracePositions = trace.data.docs.reduce((sum, doc) => sum + doc.positions.length, 0);
  const poetryAgg = poetry.data.aggregates;
  const enterPoetry = () => {
    const target = lastCase.current ?? poetry.data.cases[0]?.case_id ?? null;
    update({ poetryCase: target, position: null });
  };

  return (
    <div className="trc-station">
      {poetryActive ? (
        <StationBrief
          station="trace"
          question="Do descriptions reveal a future rhyme before the token is generated?"
          status="negative"
          statusLabel="weak signal · causal test negative"
          answer={
            <>
              Planning-like terms appear before the rhyme anchor in five of eight cases, but the
              average lift is small and no edited activation changes the generated rhyme family.
            </>
          }
          note="Treat this as an exploratory look at what the decoder verbalizes, not evidence that Nano30B explicitly plans future tokens."
          metrics={[
            {
              label: "cases with onset",
              value: `${poetryAgg.cases_with_planning_onset}/${poetryAgg.cases}`,
              detail: "crossed the preregistered onset threshold",
            },
            {
              label: "mean anchor lift",
              value: fmt(poetryAgg.mean_anchor_lift, 3),
              detail: "real − shuffled rate at the anchor position only",
            },
            {
              label: "causal alternate rhyme",
              value: fmtPct(poetryAgg.edited_alternate_hit_rate, 0),
              detail: `${poetryAgg.editable_cases} editable cases · doses 0, 0.5, 1`,
            },
          ]}
        />
      ) : (
        <StationBrief
          station="trace"
          question="How do learned activation descriptions change across a document?"
          status="exploratory"
          statusLabel="exploratory · fresh forward"
          answer="The film set lets you inspect when decoded concepts appear, persist, or disappear across sampled token positions. It is a hypothesis generator, not a qualified temporal claim."
          note={trace.data.shuffled_control.note}
          metrics={[
            {
              label: "documents",
              value: trace.data.docs.length.toLocaleString(),
              detail: "fresh-forward film set",
            },
            {
              label: "sampled positions",
              value: tracePositions.toLocaleString(),
              detail: `fresh activations extracted at R${trace.data.boundary}`,
            },
            {
              label: "trace control",
              value: trace.data.shuffled_control.available ? "available" : "not available",
              detail: "poetry lens carries the real-vs-shuffled control",
            },
          ]}
        />
      )}
      <div className="controls-row">
        <Segmented
          options={MODES}
          value={poetryActive ? MODES[1] : MODES[0]}
          onChange={(mode) => {
            if (mode === MODES[1]) enterPoetry();
            else update({ poetryCase: null, position: null });
          }}
          label="TRACE mode"
        />
        <Badge
          status="exploratory"
          label="fresh-forward"
          title={`claim scope: ${trace.data.claim_scope} — fresh extractions outside the qualified stored-snapshot channel claim`}
        />
        <button
          type="button"
          className="trc-audit-link"
          onClick={() => update({ station: "audit", claim: "fresh_forward_trace" })}
          title="Open the fresh-forward trace claim in the AUDIT station"
        >
          claim in AUDIT ↗
        </button>
      </div>
      {poetryActive ? (
        <PoetryLens poetry={poetry.data} state={state} update={update} />
      ) : (
        <DocFilmSet
          trace={trace.data}
          rows={rows.data}
          state={state}
          update={update}
          onEnterPoetry={enterPoetry}
        />
      )}
    </div>
  );
}
