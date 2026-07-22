/**
 * Causal edit bench for one poetry case: original-vs-edited explanation,
 * edit map, reconstruction metrics, and the steering interventions table.
 * Control rows (dose 0, random direction) always render and cannot be
 * dismissed; the failed-steering verdict stays visible.
 */
import { useMemo } from "react";
import type { PoetryCase, PoetryIntervention } from "../../data/types";
import { Badge, UnavailableBox } from "../../components/ui";
import { fmt } from "../../data/format";
import HighlightTerms from "./HighlightTerms";

const DIRECTIONS: { key: PoetryIntervention["direction"]; label: string }[] = [
  { key: "edited", label: "edited direction" },
  { key: "random", label: "random control (norm-matched)" },
];

export default function CausalEditBench(props: { kase: PoetryCase }) {
  const { kase } = props;
  const recon = kase.reconstruction;

  const doses = useMemo(
    () => [...new Set(kase.interventions.map((i) => i.dose))].sort((a, b) => a - b),
    [kase],
  );
  const editedTerms = useMemo(
    () =>
      recon
        ? recon.changed_terms.map((t) => kase.edit_map[t]).filter((t): t is string => Boolean(t))
        : [],
    [kase, recon],
  );

  if (!recon) {
    return (
      <UnavailableBox>
        this case was not editable (no anchor sample contained an editable target-family term)
      </UnavailableBox>
    );
  }

  const edited = kase.interventions.filter((i) => i.direction === "edited");
  const steeringFailed = edited.length > 0 && edited.every((i) => !i.hits_alternate_family);

  return (
    <>
      <div className="trc-process-step">
        <span>Step 1</span>
        <div><strong>Edit the NLA verbalization</strong><small>Only the highlighted rhyme-family words are replaced.</small></div>
      </div>
      <div className="trc-diff-cols">
        <div>
          <h4 className="trc-h4">Original AV output from the real activation</h4>
          <div className="text-block trc-diff-text" tabIndex={0}>
            <HighlightTerms
              text={recon.original_explanation}
              targetTerms={recon.changed_terms}
              targetTitle="edited term (original)"
            />
          </div>
        </div>
        <div>
          <h4 className="trc-h4">Counterfactual NLA text after the word edit</h4>
          <div className="text-block trc-diff-text" tabIndex={0}>
            <HighlightTerms
              text={recon.edited_explanation}
              targetTerms={editedTerms}
              targetTitle="replacement term (edited)"
            />
          </div>
        </div>
      </div>
      <div className="trc-chiprow">
        <span className="trc-poem-k">changed terms</span>
        {recon.changed_terms.map((t) => (
          <span key={t} className="trc-hitchip trc-hitchip-target">
            {t} → {kase.edit_map[t] ?? "?"}
          </span>
        ))}
      </div>
      <details className="trc-details">
        <summary>full edit map</summary>
        <div className="trc-table-scroll">
          <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th>from</th>
                <th>to</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(kase.edit_map).map(([from, to]) => (
                <tr key={from}>
                  <td className="mono">{from}</td>
                  <td className="mono">{to}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <div className="trc-process-step">
        <span>Step 2</span>
        <div><strong>AR turns the text back into an activation</strong><small>The edited-minus-original activation direction becomes the patch.</small></div>
      </div>
      <dl className="kv-list trc-metrics">
        <dt title="cosine between AR's reconstruction of the original anchor sample and the fresh anchor activation — the base the edit direction is computed from">
          original cosine
        </dt>
        <dd>{fmt(recon.original_cosine, 3)}</dd>
        <dt title="directional MSE of the original anchor-sample reconstruction">original dMSE</dt>
        <dd>{fmt(recon.original_dmse, 3)}</dd>
        <dt title="absolute L2 length of the edited-minus-original reconstruction difference, in activation units (no normalized scale ships in this bundle)">
          edit Δ norm (absolute)
        </dt>
        <dd>{fmt(recon.edit_delta_norm, 2)}</dd>
      </dl>
      <div className="trc-process-step">
        <span>Step 3</span>
        <div><strong>Patch Nano30B and generate from the same original prefix</strong><small>Edited and random-control outputs are shown in full at every dose.</small></div>
      </div>
      {edited.length > 0 ? (
        steeringFailed ? (
          <p className="trc-verdict">
            <Badge status="negative" label="steering failed" />
            No alternate-family rhyme at any dose — steering failed on this case.
          </p>
        ) : (
          <p className="trc-verdict">
            <Badge status="exploratory" label="alternate-family hit" />
            At least one edited-direction dose produced an alternate-family rhyme.
          </p>
        )
      ) : null}
      {doses.length === 0 ? (
        <UnavailableBox>no steering interventions recorded for this case.</UnavailableBox>
      ) : (
        <>
          {doses.includes(0) ? (
            <p className="trc-note">
              Dose 0 patches the unedited reconstruction — a patch-fidelity control shared by both
              directions, so its outputs coincide by construction.
            </p>
          ) : null}
          <div className="trc-outcome-lanes">
          {DIRECTIONS.map((dir) => (
            <section key={dir.key} className={`trc-outcome-lane trc-outcome-lane-${dir.key}`}>
              <div className="trc-outcome-lane-head">
                <strong>{dir.label}</strong>
                <span>{dir.key === "edited" ? "intended semantic direction" : "same-size unrelated direction"}</span>
              </div>
              <div className="trc-outcome-grid">
                {doses.map((dose) => {
                  const iv = kase.interventions.find(
                    (item) => item.direction === dir.key && item.dose === dose,
                  );
                  return (
                    <article key={dose} className="trc-outcome-card">
                      <div className="trc-outcome-card-head">
                        <strong>dose {dose}</strong>
                        <span>{iv ? (iv.hits_alternate_family ? "alternate rhyme hit" : "no alternate rhyme") : "not recorded"}</span>
                      </div>
                      {iv ? (
                        <>
                          <div className="text-block trc-outcome-text">
                            <HighlightTerms
                              text={iv.continuation_text}
                              targetTerms={kase.target_terms}
                              alternateTerms={kase.alternate_terms}
                            />
                          </div>
                          <div className="trc-outcome-flags">
                            <span>{iv.hits_target_family ? "✓ target-family" : "✗ target-family"}</span>
                            <span>{iv.hits_alternate_family ? "✓ alternate-family" : "✗ alternate-family"}</span>
                          </div>
                        </>
                      ) : <UnavailableBox>no output recorded</UnavailableBox>}
                    </article>
                  );
                })}
              </div>
            </section>
          ))}
          </div>
        </>
      )}
    </>
  );
}
