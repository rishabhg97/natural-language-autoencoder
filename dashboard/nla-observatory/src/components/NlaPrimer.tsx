import { useState } from "react";
import { ArrowRight, BookOpen, CornerDownRight } from "lucide-react";
import { TERM_DEFINITIONS } from "./panelGuides";
import { Badge } from "./ui";
import type { Station } from "../app/urlState";
import type { EvidenceStatus } from "../data/types";

const CORE_TERMS = [
  "activation",
  "AV",
  "learned description",
  "AR",
  "RL",
  "critic",
  "stored snapshot",
  "fresh forward",
  "control",
  "dMSE",
  "cosine",
] as const;

/**
 * The four rooms, with the same bounded one-line answers the station briefs
 * carry. Copy here must stay in sync with each StationBrief.
 */
const ORIENTATION: {
  station: Station;
  label: string;
  question: string;
  answer: string;
  status: EvidenceStatus;
  statusLabel: string;
}[] = [
  {
    station: "channel",
    label: "CHANNEL",
    question: "What survives the trip through language?",
    answer:
      "On matched validation rows, online RL lowered round-trip error 27.4% versus SFT. Stored-snapshot tests separately show the language channel beats every control.",
    status: "exploratory",
    statusLabel: "matched validation",
  },
  {
    station: "trace",
    label: "TRACE",
    question: "What changes across tokens?",
    answer:
      "Fresh-forward readouts along documents, plus a poetry lens whose planning-like signal is weak and whose causal test failed.",
    status: "exploratory",
    statusLabel: "exploratory",
  },
  {
    station: "bench",
    label: "BENCH",
    question: "Do language edits cause distinct model change?",
    answer:
      "Precomputed patches move the model, but a credible effect must separate from paraphrase, random, and identity controls.",
    status: "exploratory",
    statusLabel: "validation-only",
  },
  {
    station: "audit",
    label: "AUDIT",
    question: "What is safe to claim?",
    answer:
      "One bounded stored-snapshot claim is qualified; the matched RL gain is validation-only, with negative results and provenance kept visible.",
    status: "qualified",
    statusLabel: "boundaries",
  },
];

const PRIMER_OPEN_KEY = "nla-primer-open";

function initialOpen(): boolean {
  try {
    return window.localStorage.getItem(PRIMER_OPEN_KEY) !== "false";
  } catch {
    return true;
  }
}

function FlowArrow({ label }: { label: string }) {
  return (
    <div className="nla-primer-arrow" aria-label={label}>
      <span>{label}</span>
      <ArrowRight size={18} aria-hidden />
    </div>
  );
}

export function NlaPrimer(props: {
  station: Station;
  onOpenStation: (station: Station) => void;
}) {
  const [open, setOpen] = useState(initialOpen);

  return (
    <section className="nla-primer" aria-labelledby="nla-primer-title">
      <div className="nla-primer-heading">
        <div>
          <div className="nla-primer-kicker">The shared experiment</div>
          <h2 id="nla-primer-title">What the NLA is doing</h2>
          <p>
            It learns a language bottleneck between a hidden activation and a reconstructed
            activation. The generated text is an encoding to test, not a transcript of private
            chain-of-thought.
          </p>
        </div>
        <details className="nla-glossary">
          <summary>
            <BookOpen size={14} aria-hidden /> Dashboard glossary
          </summary>
          <dl>
            {CORE_TERMS.map((term) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{TERM_DEFINITIONS[term]}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>

      <details
        className="nla-primer-details"
        open={open}
        onToggle={(event) => {
          const next = (event.currentTarget as HTMLDetailsElement).open;
          setOpen(next);
          try {
            window.localStorage.setItem(PRIMER_OPEN_KEY, String(next));
          } catch {
            /* private mode */
          }
        }}
      >
        <summary>How the pipeline fits together</summary>
        <div className="nla-primer-flow" aria-label="Natural-language autoencoder pipeline">
          <div className="nla-primer-node">
            <strong>Activation h</strong>
            <span>2,688 numbers from Nano30B layer R33</span>
          </div>
          <FlowArrow label="AV reads" />
          <div className="nla-primer-node nla-primer-node-text">
            <strong>Learned description z</strong>
            <span>natural language generated from h</span>
          </div>
          <FlowArrow label="AR reconstructs" />
          <div className="nla-primer-node">
            <strong>Reconstruction h_hat</strong>
            <span>predicted activation compared with h</span>
          </div>
        </div>
        <div className="nla-primer-extension">
          <CornerDownRight size={16} aria-hidden />
          <span>
            <strong>BENCH adds a causal question:</strong> edit z, reconstruct h_hat, patch it into
            Nano30B, and compare the resulting behavior with matched controls.
          </span>
        </div>
      </details>

      <nav className="nla-orientation" aria-label="One story in four stations">
        {ORIENTATION.map((room) => {
          const current = room.station === props.station;
          return (
            <button
              key={room.station}
              type="button"
              className="nla-orientation-card"
              aria-pressed={current}
              aria-current={current ? "page" : undefined}
              onClick={() => props.onOpenStation(room.station)}
            >
              <span className="nla-orientation-head">
                <strong>{room.label}</strong>
                <Badge status={room.status} label={room.statusLabel} />
              </span>
              <span className="nla-orientation-question">{room.question}</span>
              <span className="nla-orientation-answer">{room.answer}</span>
            </button>
          );
        })}
      </nav>
    </section>
  );
}
