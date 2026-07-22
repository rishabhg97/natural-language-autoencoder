import { describe, expect, it } from "vitest";
import { panelGuideFor, TERM_DEFINITIONS } from "../src/components/panelGuides";

const PANEL_TITLES = [
  "Selected row: source → NLA description",
  "Did online RL improve reconstruction?",
  "Information waterfall",
  "Real-vs-control AV loss",
  "Twin critics",
  "Capacity ladder",
  "Retrieval",
  "Words buy direction (rate-distortion)",
  "Word / section attribution - validation row",
  "Alternate-tellings fan - validation row",
  "Rewrite stress test",
  "Document token-linked NLA reader",
  "Learned description - position 18",
  "Persistence of adjacent descriptions",
  "Real-vs-shuffled control",
  "Fresh-vs-stored drift",
  "Cross-station: BENCH",
  "Prefix and analysis window",
  "Token-linked poetry NLA reader",
  "Planning-onset curve",
  "AV samples - offset -6",
  "Baseline continuation",
  "Causal edit bench",
  "All cases",
  "Row + move picker",
  "Control rack - lanes travel together",
  "Reconstruction compass",
  "Next-token top-k movement",
  "Divergence gauges",
  "Causal wake",
  "Continuations - baseline vs patched",
  "Dose comparison",
  "Session ledger - 1 experiment opened",
  "Claim ledger",
  "Negative & weak results",
  "Parse health & control completeness",
  "Cipher court docket",
  "Magnitude card",
  "Null-text almanac",
  "Poetry pipeline status",
  "Provenance browser",
  "Row evidence",
] as const;

describe("panel guide registry", () => {
  it.each(PANEL_TITLES)("explains %s", (title) => {
    const guide = panelGuideFor(title);
    expect(guide, `${title} has no reader guide`).not.toBeNull();
    expect(guide?.what.length).toBeGreaterThan(40);
    expect(guide?.read.length).toBeGreaterThan(40);
    for (const term of guide?.terms ?? []) {
      expect(TERM_DEFINITIONS[term], `${title} references undefined term ${term}`).toBeTruthy();
    }
  });

  it("does not invent a generic explanation for unknown panels", () => {
    expect(panelGuideFor("Unknown future panel")).toBeNull();
  });
});
