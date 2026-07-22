/**
 * Claim ledger — the scoped claims of the bundle, each with its exact
 * status string from the shard, plus the evidence-status legend and the
 * verbatim limitations list. Each card carries the DOM id other stations
 * deep-link to (claim=stored_snapshot_channel | fresh_forward_trace |
 * functional_interventions | test_set).
 */

import type { AuditShard, EvidenceStatus } from "../../data/types";
import { Badge, Panel } from "../../components/ui";
import { humanize } from "./util";

const CLAIM_ORDER = [
  "stored_snapshot_channel",
  "matched_online_rl_roundtrip",
  "fresh_forward_trace",
  "functional_interventions",
  "test_set",
] as const;

/**
 * One-sentence neutral meaning per claim key. Prose only — every number and
 * status string on the cards comes from the shard.
 */
const CLAIM_MEANING: Record<string, string> = {
  stored_snapshot_channel:
    "Learned descriptions (encodings) of stored activation snapshots carry enough information for the critics to reconstruct the target activation direction on the qualified validation panel.",
  matched_online_rl_roundtrip:
    "On the same 122 held-out validation families and 384-token protocol, the jointly trained RL actor and critic reduced round-trip directional error by 27.4% versus clean SFT. This is validation evidence, not a sealed test or component attribution.",
  fresh_forward_trace:
    "Per-token learned descriptions decoded during fresh forward passes; stored-vs-fresh activation drift bounds this view, so it sits outside the qualified claim.",
  functional_interventions:
    "Activation-patching outcomes were measured on validation rows only; no unexposed confirmatory boundary backs a functional claim.",
  test_set:
    "The test split was not opened to build this dashboard's evidence lattice, so no claim of any strength is made about it here.",
};

function claimBadge(status: string) {
  switch (status) {
    case "qualified":
      return <Badge status="qualified" />;
    case "exploratory":
      return <Badge status="exploratory" />;
    case "validation_only_exploratory":
      return <Badge status="exploratory" label="validation-only exploratory" />;
    case "validation_only_confirmatory":
      return <Badge status="exploratory" label="matched validation" />;
    case "not_opened_for_dashboard_lattice":
      return <Badge status="unavailable" label="not opened" />;
    default:
      return <Badge status="unavailable" label={humanize(status)} />;
  }
}

const LEGEND_ORDER: readonly EvidenceStatus[] = [
  "qualified",
  "exploratory",
  "negative",
  "unavailable",
];

export default function ClaimLedgerPanel(props: { audit: AuditShard }) {
  const { claims, limitations } = props.audit.claim_ledger;
  const legend = props.audit.evidence_status_legend;
  const keys: string[] = [
    ...CLAIM_ORDER.filter((k) => k in claims),
    ...Object.keys(claims)
      .filter((k) => !(CLAIM_ORDER as readonly string[]).includes(k))
      .sort(),
  ];
  return (
    <Panel
      id="claim_ledger"
      span={12}
      title="Claim ledger"
      sub="What each layer of evidence is allowed to claim. A permalink to any card cites its scope; deeper stations link back here."
    >
      <div className="audit-claim-cards">
        {keys.map((key) => (
          <div className="audit-claim-card" id={key} key={key}>
            <h4>
              {humanize(key)} {claimBadge(claims[key])}
            </h4>
            <div className="audit-claim-status" title="status string, verbatim from the bundle">
              {claims[key]}
            </div>
            <p>{CLAIM_MEANING[key] ?? ""}</p>
          </div>
        ))}
      </div>
      <div className="audit-legend-row" aria-label="Evidence status legend">
        {LEGEND_ORDER.filter((s) => s in legend).map((s) => (
          <span key={s}>
            <Badge status={s} /> {legend[s]}
          </span>
        ))}
      </div>
      <h4 className="audit-subhead">Limitations</h4>
      <ul className="audit-limitations">
        {limitations.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </Panel>
  );
}
