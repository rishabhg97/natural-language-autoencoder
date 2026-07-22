/**
 * Mode B — the poetry planning lens: inline source-token selector,
 * real-vs-shuffled onset curve, AV samples, the unpatched baseline lane,
 * the causal edit bench, and the all-cases summary strip.
 */
import { useMemo, useRef } from "react";
import { ChevronLeft, ChevronRight, Link2 } from "lucide-react";
import type { StationProps } from "../../app/stationProps";
import type { PoetryCase, PoetryShard } from "../../data/types";
import { Badge, Panel, SelectControl, UnavailableBox } from "../../components/ui";
import { fmt } from "../../data/format";
import { FreshBadge } from "./common";
import OnsetCurve from "./OnsetCurve";
import PoetrySamples from "./PoetrySamples";
import CausalEditBench from "./CausalEditBench";
import CaseSummary from "./CaseSummary";
import HighlightTerms from "./HighlightTerms";

interface Props extends StationProps {
  poetry: PoetryShard;
}

function displayToken(text: string): string {
  const shown = text.replace(/\n/g, "↵");
  return shown === "" ? "·" : shown;
}

export default function PoetryLens({ poetry, state, update }: Props) {
  const cases = poetry.cases;
  const kase: PoetryCase | undefined =
    cases.find((c) => c.case_id === state.poetryCase) ?? cases[0];
  const idx = kase ? cases.indexOf(kase) : -1;

  const chipRefs = useRef<(HTMLButtonElement | null)[]>([]);

  // Plain prefix head = prefix_text minus the concatenated analysis tokens;
  // null if the shard tokens do not align (then the chips render standalone).
  const prefixHead = useMemo(() => {
    if (!kase) return null;
    const joined = kase.analysis.map((t) => t.token_text).join("");
    return kase.prefix_text.endsWith(joined)
      ? kase.prefix_text.slice(0, kase.prefix_text.length - joined.length)
      : null;
  }, [kase]);

  if (!kase) {
    return <UnavailableBox>poetry.json contains no cases — the planning lens has nothing to show.</UnavailableBox>;
  }

  const anchor =
    kase.analysis.find((t) => t.relative_offset === 0) ?? kase.analysis[kase.analysis.length - 1];
  // Default = anchor position; derived only, so the URL is untouched until the user acts.
  const selTok = kase.analysis.find((t) => t.position === state.position) ?? anchor;
  const selIdx = kase.analysis.indexOf(selTok);

  const goTo = (index: number) => {
    const next = Math.min(kase.analysis.length - 1, Math.max(0, index));
    const tok = kase.analysis[next];
    if (tok && tok.position !== selTok.position) {
      update({ position: tok.position });
      requestAnimationFrame(() => chipRefs.current[next]?.focus());
    }
  };

  const selectCase = (caseId: string) => update({ poetryCase: caseId, position: null });

  return (
    <div className="trc-poetry">
      <div className="controls-row">
        <button
          type="button"
          className="trc-button"
          disabled={idx <= 0}
          onClick={() => selectCase(cases[idx - 1].case_id)}
          aria-label="previous case"
        >
          <ChevronLeft size={14} aria-hidden />
          <span>previous case</span>
        </button>
        <SelectControl
          label="Case"
          value={kase.case_id}
          options={cases.map((c) => ({ value: c.case_id, label: c.case_id }))}
          onChange={selectCase}
        />
        <button
          type="button"
          className="trc-button"
          disabled={idx >= cases.length - 1}
          onClick={() => selectCase(cases[idx + 1].case_id)}
          aria-label="next case"
        >
          <span>next case</span>
          <ChevronRight size={14} aria-hidden />
        </button>
        <FreshBadge scope={poetry.claim_scope} />
        {kase.planning_onset_position === null ? (
          <Badge
            status="negative"
            label="no onset"
            title="no analysis position cleared the planning-onset gate for this case"
          />
        ) : null}
      </div>
      <div className="controls-row trc-kchips">
        <span className="trc-kchip">
          <span className="trc-kchip-k">target word</span>
          <span className="mono">{kase.target_word}</span>
        </span>
        <span className="trc-kchip">
          <span className="trc-kchip-k">target terms</span>
          <span className="mono">{kase.target_terms.join(", ")}</span>
        </span>
        <span className="trc-kchip">
          <span className="trc-kchip-k">alternate terms</span>
          <span className="mono">{kase.alternate_terms.join(", ")}</span>
        </span>
        <span className="trc-kchip">
          <span className="trc-kchip-k">cue</span>
          <span className="mono">{kase.cue}</span>
        </span>
        <span className="trc-note trc-offset-note">
          Offsets count backward from the rhyme anchor (offset 0, the line-end token): a signal at
          a negative offset occurs before the target rhyme token enters the context.
        </span>
      </div>

      <div className="panel-grid">
        <Panel
          title="Token-linked poetry NLA reader"
          span={12}
          badges={<FreshBadge scope={poetry.claim_scope} />}
          sub="one continuous exhibit: original prefix, highlighted token, its fresh activation, and every real/control AV verbalization generated at that position"
        >
          <div className="trc-poetry-split">
            <section className="trc-linked-pane trc-source-pane" aria-label="Original poetry prefix">
              <div className="trc-pane-label">1 · Full original poetry prefix</div>
              <h4>Select any highlighted token to inspect its NLA samples</h4>
              {prefixHead !== null ? (
                <div
                  className="trc-full-source trc-full-poetry-source"
                  tabIndex={0}
                  aria-label="Original poetry prefix with selectable analyzed tokens"
                >
                  <span>{prefixHead}</span>
                  {kase.analysis.map((token, tokenIndex) => {
                    const isAnchor = token.relative_offset === 0;
                    const isOnset = kase.planning_onset_position === token.position;
                    const isFuture = tokenIndex > selIdx;
                    return (
                      <button
                        key={token.position}
                        ref={(element) => {
                          chipRefs.current[tokenIndex] = element;
                        }}
                        type="button"
                        className={`trc-inline-source-token${isFuture ? " trc-inline-token-future" : ""}${isAnchor ? " trc-inline-token-anchor" : ""}${isOnset ? " trc-inline-token-onset" : ""}`}
                        aria-pressed={token.position === selTok.position}
                        onClick={() => update({ position: token.position })}
                        title={`Open NLA samples for “${displayToken(token.token_text)}” · offset ${token.relative_offset}${isAnchor ? " · rhyme anchor" : ""}${isOnset ? " · measured planning onset" : ""}`}
                      >
                        {token.token_text}
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="trc-full-source" tabIndex={0}>{kase.prefix_text}</div>
              )}
              <div className="trc-source-selector-legend">
                <span><i className="trc-legend-available" /> analyzed token</span>
                <span><i className="trc-legend-selected" /> selected token</span>
                <span><i className="trc-legend-future" /> after the selected position; not yet seen</span>
                <span>Tokens are model tokens — words can split (e.g., a final letter on its own).</span>
              </div>
              <div className="trc-heldout-poem">
                <span>Held-out target continuation · never shown at any analyzed position</span>
                <strong>
                  <HighlightTerms
                    text={kase.second_line}
                    targetTerms={kase.target_terms}
                    alternateTerms={kase.alternate_terms}
                  />
                </strong>
              </div>
            </section>
            <section className="trc-linked-pane trc-poetry-output-pane" aria-label="NLA verbalizations and shuffled controls">
              <div className="trc-selected-output-head">
                <div>
                  <div className="trc-pane-label">2 · NLA samples for selected token</div>
                  <h4>“{displayToken(selTok.token_text)}” <span>· offset {selTok.relative_offset}</span></h4>
                </div>
                <div className="trc-step-buttons" aria-label="Step through analyzed poetry tokens">
                  <button type="button" onClick={() => goTo(selIdx - 1)} disabled={selIdx === 0} aria-label="Previous analyzed token" title="Previous analyzed token">
                    <ChevronLeft size={15} aria-hidden />
                  </button>
                  <span>{selIdx + 1} / {kase.analysis.length}</span>
                  <button type="button" onClick={() => goTo(selIdx + 1)} disabled={selIdx === kase.analysis.length - 1} aria-label="Next analyzed token" title="Next analyzed token">
                    <ChevronRight size={15} aria-hidden />
                  </button>
                </div>
              </div>
              <div className="trc-activation-bridge">
                <Link2 size={16} aria-hidden />
                <span>fresh activation at the selected token → 4 real AV samples + 2 shuffled controls</span>
              </div>
              <PoetrySamples kase={kase} position={selTok.position} offset={selTok.relative_offset} />
            </section>
          </div>
        </Panel>

        <Panel
          title="Planning-onset curve"
          span={12}
          badges={<FreshBadge scope={poetry.claim_scope} />}
          sub={`real vs shuffled target-family rate across the analysis window · onset gate ${fmt(poetry.gates.planning_onset_rate, 2)} (from the shard)`}
        >
          <OnsetCurve
            kase={kase}
            gate={poetry.gates.planning_onset_rate}
            selectedOffset={selTok.relative_offset}
            onSelectOffset={(o) => {
              const tok = kase.analysis.find((t) => t.relative_offset === o);
              if (tok) update({ position: tok.position });
            }}
          />
        </Panel>

        <Panel
          title="Baseline continuation"
          span={12}
          badges={<FreshBadge scope={poetry.claim_scope} />}
          sub="unpatched greedy continuation — behavior agreement lane; disagreement stays visible."
        >
          <div className="trc-baseline-flow">
            <section className="trc-linked-pane trc-source-pane">
              <div className="trc-pane-label">1 · Input</div>
              <h4>Original prefix shown to Nano30B</h4>
              <div className="trc-source-text trc-poetry-source-text">{kase.prefix_text}</div>
              <div className="trc-heldout-poem">
                <span>Reference second line · not supplied to the model</span>
                <strong>{kase.second_line}</strong>
              </div>
            </section>
            <div className="trc-physical-link">
              <span>unpatched greedy generation</span>
              <strong>no NLA edit applied</strong>
            </div>
            <section className="trc-linked-pane trc-output-pane">
              <div className="trc-pane-label">2 · Baseline model output</div>
              <h4>What Nano30B generated from the original prefix</h4>
              <div className="text-block trc-baseline-output">
                <HighlightTerms
                  text={kase.baseline_continuation}
                  targetTerms={kase.target_terms}
                  alternateTerms={kase.alternate_terms}
                />
              </div>
              <p className="trc-verdict-line">
                {kase.baseline_hits_target_family ? (
                  <Badge status="qualified" label="lands in target rhyme family" />
                ) : (
                  <Badge status="negative" label="does not land in target rhyme family" />
                )}
              </p>
            </section>
          </div>
        </Panel>

        <Panel
          title="Causal edit bench"
          span={12}
          badges={<FreshBadge scope={poetry.claim_scope} />}
          sub="dose 0 is a pure gold-replacement patch-fidelity control; direction 'random' is a norm-matched random-direction control."
        >
          <CausalEditBench kase={kase} />
        </Panel>

        <Panel
          title="All cases · original text and observed outcomes"
          span={12}
          badges={<FreshBadge scope={poetry.claim_scope} />}
          sub="every case stays on the board — zero lifts, missing onsets, and failed steering included. Stored outputs end at the generation cap, so some stop mid-sentence."
        >
          <CaseSummary poetry={poetry} currentCaseId={kase.case_id} onSelect={selectCase} />
        </Panel>
      </div>
    </div>
  );
}
