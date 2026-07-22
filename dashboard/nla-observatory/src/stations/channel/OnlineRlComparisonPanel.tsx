import { ArrowRight, CheckCircle2, ShieldAlert } from "lucide-react";
import type { AppState } from "../../app/urlState";
import { Badge, Panel } from "../../components/ui";
import type { ChannelShard, MatchedOnlineRlControl } from "../../data/types";
import { fmt, fmtPct } from "../../data/format";
import { AuditLink } from "./lib";

const CONTROL_EXPLANATIONS: Record<string, string> = {
  av_shuffled: "description from the wrong activation",
  av_zero: "description from a zero activation",
  av_mean: "description from the dataset-average activation",
  av_none: "description with activation input removed",
};

export default function OnlineRlComparisonPanel(props: {
  channel: ChannelShard;
  update: (patch: Partial<AppState>) => void;
}) {
  const result = props.channel.matched_online_rl;
  const comparisonRows: MatchedOnlineRlControl[] = [
    {
      key: "real",
      label: "matching RL description",
      roundtrip_nmse: result.rl.roundtrip_nmse,
    },
    {
      key: "teacher",
      label: "teacher reference text",
      roundtrip_nmse: result.rl.teacher_nmse,
    },
    ...result.rl.controls,
  ];
  const maxError = Math.max(...comparisonRows.map((row) => row.roundtrip_nmse));

  return (
    <Panel
      id="chan-online-rl"
      title="Did online RL improve reconstruction?"
      span={12}
      badges={
        <>
          <Badge
            status="exploratory"
            label="matched validation"
            title="Same held-out validation rows and generation protocol; not a sealed test result"
          />
          <AuditLink claim="matched_online_rl_roundtrip" update={props.update} />
        </>
      }
      sub={`${result.row_count} held-out validation rows from ${result.independent_family_count} independent content families. SFT and RL used the same rows and ${result.max_new_tokens}-token generation budget.`}
    >
      <div className="chan-rl-answer">
        <CheckCircle2 size={22} aria-hidden />
        <div>
          <strong>Yes. Online RL lowered round-trip error by {fmtPct(result.improvement.nmse_relative)}.</strong>
          <p>
            The generated description was decoded back into an activation closer to the original.
            Lower error is better.
          </p>
        </div>
      </div>

      <div
        className="chan-rl-comparison"
        aria-label={`Round-trip directional error improved from ${fmt(result.sft.roundtrip_nmse, 3)} with SFT to ${fmt(result.rl.roundtrip_nmse, 3)} with online RL`}
      >
        <div className="chan-rl-stage">
          <span>Before RL</span>
          <strong>{fmt(result.sft.roundtrip_nmse, 3)}</strong>
          <small>clean SFT checkpoint</small>
        </div>
        <div className="chan-rl-change" aria-hidden>
          <ArrowRight size={24} />
          <strong>{fmtPct(result.improvement.nmse_relative)} lower</strong>
        </div>
        <div className="chan-rl-stage chan-rl-stage-selected">
          <span>After 342 RL updates</span>
          <strong>{fmt(result.rl.roundtrip_nmse, 3)}</strong>
          <small>selected actor + critic pair</small>
        </div>
      </div>
      <p className="chan-rl-metric-definition">
        <strong>Round-trip directional error:</strong> encode an activation as language, decode that
        language back into an activation, then compare its direction with the original.
      </p>

      <div className="chan-rl-support" aria-label="Supporting matched results">
        <div>
          <span>Raw activation error</span>
          <strong>
            {fmt(result.sft.raw_mse, 2)} <ArrowRight size={14} aria-label="to" /> {fmt(result.rl.raw_mse, 2)}
          </strong>
          <small>{fmtPct(result.improvement.raw_mse_relative)} lower</small>
        </div>
        <div>
          <span>Generated text beats teacher text</span>
          <strong>
            {result.sft.teacher_win_count}/{result.row_count} <ArrowRight size={14} aria-label="to" />{" "}
            {result.rl.teacher_win_count}/{result.row_count}
          </strong>
          <small>{fmtPct(result.rl.teacher_win_fraction)} of RL rows</small>
        </div>
        <div>
          <span>Explanation closed correctly</span>
          <strong>
            {result.sft.parse.closed_count}/{result.row_count} <ArrowRight size={14} aria-label="to" />{" "}
            {result.rl.parse.closed_count}/{result.row_count}
          </strong>
          <small>RL still had usable text on {result.rl.parse.usable_count}/{result.row_count}</small>
        </div>
      </div>

      <div className="chan-rl-controls">
        <div className="chan-rl-controls-heading">
          <div>
            <h4>Sanity check: does the matching activation matter?</h4>
            <p>Yes. The real RL round trip has much lower error than every bad-input control.</p>
          </div>
          <span>lower is better</span>
        </div>
        <div className="chan-rl-control-list">
          {comparisonRows.map((row) => (
            <div className={`chan-rl-control${row.key === "real" ? " is-real" : ""}`} key={row.key}>
              <div>
                <strong>{row.label}</strong>
                <small>{CONTROL_EXPLANATIONS[row.key] ?? (row.key === "teacher" ? "human-written reference explanation" : "true activation paired with its generated description")}</small>
              </div>
              <span className="chan-rl-control-track" aria-hidden>
                <span style={{ width: `${(row.roundtrip_nmse / maxError) * 100}%` }} />
              </span>
              <output aria-label={`${row.label} round-trip directional error`}>
                {fmt(row.roundtrip_nmse, 3)}
              </output>
            </div>
          ))}
        </div>
      </div>

      <div className="chan-rl-scope-note">
        <ShieldAlert size={17} aria-hidden />
        <p>{result.scope_note}</p>
      </div>
    </Panel>
  );
}
