/**
 * AUDIT — "what is safe to claim". Claim ledger, negative results, cipher
 * court docket, parse health, drift, null-text almanac, magnitude card,
 * poetry pipeline status, and the provenance browser.
 *
 * Deep-link contract: other stations navigate here with
 * update({ station: "audit", claim: "<id>" }). Every major panel (and each
 * claim-ledger card) carries that id as a DOM anchor; when state.claim
 * matches, the element scrolls into view and flashes the selected wash.
 */

import { useEffect } from "react";
import type { StationProps } from "../../app/stationProps";
import type { AuditShard } from "../../data/types";
import { useShard } from "../../data/loader";
import { ErrorBox, LoadingBox, StationBrief } from "../../components/ui";
import ClaimLedgerPanel from "./ClaimLedgerPanel";
import NegativeResultsPanel from "./NegativeResultsPanel";
import CourtPanel from "./CourtPanel";
import ParseHealthPanel from "./ParseHealthPanel";
import DriftPanel from "./DriftPanel";
import NullTextPanel from "./NullTextPanel";
import MagnitudePanel from "./MagnitudePanel";
import PoetryStatusPanel from "./PoetryStatusPanel";
import ProvenancePanel from "./ProvenancePanel";
import "./station.css";

export default function AuditStation(props: StationProps) {
  const audit = useShard<AuditShard>("audit.json");
  if (audit.status === "loading") return <LoadingBox what="AUDIT evidence (audit.json)" />;
  if (audit.status === "error") return <ErrorBox message={audit.message} />;
  return <AuditContent audit={audit.data} state={props.state} update={props.update} />;
}

function AuditContent(props: { audit: AuditShard } & StationProps) {
  const { audit, state, update } = props;
  const claimStatuses = Object.values(audit.claim_ledger.claims);
  const qualifiedClaims = claimStatuses.filter((status) => status === "qualified").length;
  const validationClaims = claimStatuses.filter(
    (status) => status.includes("exploratory") || status.includes("validation_only"),
  ).length;

  useEffect(() => {
    if (!state.claim) return;
    const el = document.getElementById(state.claim);
    if (!el) return;
    el.scrollIntoView({ block: "start" });
    el.classList.add("audit-flash");
    const timer = window.setTimeout(() => el.classList.remove("audit-flash"), 2000);
    return () => {
      window.clearTimeout(timer);
      el.classList.remove("audit-flash");
    };
  }, [state.claim]);

  return (
    <div className="audit-station">
      <StationBrief
        station="audit"
        question="Which NLA claims survive the evidence boundaries?"
        status="qualified"
        statusLabel="one bounded claim qualified"
        answer="Stored-snapshot direction recovery is qualified. A matched validation eval also shows 27.4% lower round-trip error after online RL, while fresh-forward traces and functional interventions remain exploratory. The dashboard lattice has not opened the test set."
        note="A completed pipeline is not a successful hypothesis. Negative findings, missing controls, and publication blockers remain visible below."
        metrics={[
          {
            label: "qualified claims",
            value: qualifiedClaims.toString(),
            detail: `of ${claimStatuses.length} scoped claims`,
          },
          {
            label: "validation-only claims",
            value: validationClaims.toString(),
            detail: "matched RL and exploratory evidence",
          },
          {
            label: "weak / negative / caveat",
            value: audit.negative_results.length.toString(),
            detail: "kept visible in the evidence record",
          },
        ]}
      />
      <div className="panel-grid audit-grid">
        <ClaimLedgerPanel audit={audit} />
        <NegativeResultsPanel results={audit.negative_results} />
        <ParseHealthPanel ph={audit.parse_health} />
        <CourtPanel audit={audit} state={state} update={update} />
        <DriftPanel drift={audit.drift} state={state} update={update} />
        <NullTextPanel nt={audit.null_text} />
        <MagnitudePanel mag={audit.magnitude} />
        <PoetryStatusPanel status={audit.poetry_status} />
        <ProvenancePanel prov={audit.provenance} />
      </div>
    </div>
  );
}
