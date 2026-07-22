/**
 * AV samples at the selected position: learned descriptions (encodings) of the
 * activation, real and shuffled-control variants. Unusable samples stay
 * listed and marked. Scored-outcome chips are the authoritative record; a
 * display-level term match that the scorer did NOT count is flagged
 * explicitly rather than left to contradict the planning curve.
 */
import { useMemo, useState } from "react";
import type { PoetryCase, PoetrySample } from "../../data/types";
import { Badge, UnavailableBox } from "../../components/ui";
import HighlightTerms from "./HighlightTerms";
import { segmentByTerms } from "./textUtils";

function hasDisplayMatch(text: string, terms: readonly string[]): boolean {
  return segmentByTerms(text, terms, []).some((seg) => seg.hit === "target");
}

function SampleCard({ sample, kase }: { sample: PoetrySample; kase: PoetryCase }) {
  // Display-level matches vs the scored flags shipped in the shard.
  const displayTarget = useMemo(
    () => hasDisplayMatch(sample.explanation, kase.target_terms),
    [sample.explanation, kase.target_terms],
  );
  const displayAlternate = useMemo(
    () => hasDisplayMatch(sample.explanation, kase.alternate_terms),
    [sample.explanation, kase.alternate_terms],
  );
  const scoredMismatch =
    (displayTarget && !sample.target_family) || (displayAlternate && !sample.alternate_family);
  return (
    <article className="trc-sample">
      <div className="trc-sample-head">
        <strong>AV sample {sample.sample_index + 1}</strong>
        {sample.variant === "shuffled" ? (
          <span className="trc-vchip trc-vchip-shuffled">donor: {sample.source_case_id}</span>
        ) : null}
        {sample.usable ? null : (
          <Badge
            status="caveat"
            label="unusable"
            title={`parse: ${sample.parse.extraction_mode}${sample.parse.repetition_loop ? " · repetition loop" : ""}${sample.parse.closed ? "" : " · unclosed"}`}
          />
        )}
        {sample.target_exact ? <span className="trc-hitchip trc-hitchip-target">scored: exact target word</span> : null}
        {sample.target_family ? <span className="trc-hitchip trc-hitchip-target">scored: target-rhyme term</span> : null}
        {sample.alternate_family ? <span className="trc-hitchip trc-hitchip-alt">scored: alternate-rhyme term</span> : null}
        {scoredMismatch ? (
          <span
            className="trc-hitchip trc-hitchip-unscored"
            title="A term string appears in this text (underlined) but the pipeline's scorer did not count it — the planning curve uses the scored flags, not the underlines."
          >
            term visible · not scored
          </span>
        ) : null}
      </div>
      <div className="text-block trc-sample-output">
        <HighlightTerms
          text={sample.explanation}
          targetTerms={kase.target_terms}
          alternateTerms={kase.alternate_terms}
        />
      </div>
    </article>
  );
}

function SampleLane(props: {
  kind: "real" | "shuffled";
  samples: PoetrySample[];
  kase: PoetryCase;
}) {
  const { kind, samples, kase } = props;
  const [expanded, setExpanded] = useState(false);
  return (
    <section className={`trc-sample-lane trc-sample-lane-${kind === "real" ? "real" : "control"}`}>
      <div className="trc-sample-lane-head">
        <span className={`trc-vchip trc-vchip-${kind === "real" ? "real" : "shuffled"}`}>
          {kind === "real" ? "real activation" : "shuffled control"}
        </span>
        <strong>
          {samples.length} {kind === "real" ? "NLA verbalizations" : "control verbalizations"}
        </strong>
        <p>
          {kind === "real"
            ? "AV received the highlighted token's activation. These samples produce the blue planning rate. Content can drift between tellings — that sampling variation is part of what the shuffled control measures."
            : "AV received a donor activation instead. These outputs form the gray comparison rate."}
        </p>
      </div>
      <div className={`trc-samples${expanded ? " trc-samples-expanded" : ""}`}>
        {samples.map((sample) => (
          <SampleCard key={`${kind}-${sample.sample_index}`} sample={sample} kase={kase} />
        ))}
      </div>
      {samples.length > 1 ? (
        <button
          type="button"
          className="trc-button trc-samples-toggle"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          {expanded ? "Collapse samples" : `Show all ${samples.length} samples`}
        </button>
      ) : null}
    </section>
  );
}

export default function PoetrySamples(props: {
  kase: PoetryCase;
  position: number;
  offset: number;
}) {
  const { kase, position } = props;
  const samples = useMemo(() => {
    const order = { real: 0, shuffled: 1 } as const;
    return kase.samples
      .filter((s) => s.position === position)
      .slice()
      .sort((a, b) => order[a.variant] - order[b.variant] || a.sample_index - b.sample_index);
  }, [kase, position]);

  if (samples.length === 0) {
    return (
      <UnavailableBox>
        no AV samples at position {position} for case {kase.case_id}.
      </UnavailableBox>
    );
  }
  const real = samples.filter((s) => s.variant === "real");
  const shuffled = samples.filter((s) => s.variant === "shuffled");

  return (
    <div className="trc-sample-lanes">
      <SampleLane kind="real" samples={real} kase={kase} />
      <SampleLane kind="shuffled" samples={shuffled} kase={kase} />
      <p className="trc-sample-footnote">
        Offset {props.offset} · position {position}. Underlines are display-level term matches in
        the text; the <em>scored</em> chips on each sample are what the planning-onset curve
        counts. Model tokens can split words, so a term may appear without being scored.
      </p>
    </div>
  );
}
