/**
 * CHANNEL station — "what survives the trip through English".
 *
 * Evidence: channel.json (qualified stored-snapshot channel metrics),
 * rows.json (50-row qualified panel), and rewrites.json (transform explorer;
 * lazy-loaded inside the RewritePanel).
 *
 * All selections (critic, row, rewrite cell) round-trip through update() so
 * permalinks restore the exact view. Every displayed number comes from the
 * loaded shards; explanations are learned descriptions/encodings of a stored
 * activation — never "thoughts".
 */

import { useEffect } from "react";
import type { StationProps } from "../../app/stationProps";
import { useShard } from "../../data/loader";
import type { ChannelShard, Critic, RowsShard } from "../../data/types";
import {
  ErrorBox,
  LoadingBox,
  Segmented,
  StationBrief,
  UnavailableBox,
} from "../../components/ui";
import { fmt, fmtPct } from "../../data/format";
import WaterfallPanel from "./WaterfallPanel";
import RealVsControlPanel from "./RealVsControlPanel";
import TwinCriticsPanel from "./TwinCriticsPanel";
import CapacityPanel from "./CapacityPanel";
import RetrievalPanel from "./RetrievalPanel";
import TruncationPanel from "./TruncationPanel";
import AttributionPanel from "./AttributionPanel";
import TellingsPanel from "./TellingsPanel";
import RewritePanel from "./RewritePanel";
import RowReaderPanel from "./RowReaderPanel";
import OnlineRlComparisonPanel from "./OnlineRlComparisonPanel";
import "./channel.css";

const CRITIC_OPTIONS = ["primary", "independent"] as const;

export default function ChannelStation({ state, update }: StationProps) {
  const channel = useShard<ChannelShard>("channel.json");
  const rows = useShard<RowsShard>("rows.json");

  // Deep-linked view anchor (#/channel?view=…) scrolls to the matching panel.
  useEffect(() => {
    if (!state.view || channel.status !== "ready" || rows.status !== "ready") return;
    document.getElementById(`chan-${state.view}`)?.scrollIntoView({ block: "start" });
  }, [state.view, channel.status, rows.status]);

  if (channel.status === "loading" || rows.status === "loading") {
    return <LoadingBox what="CHANNEL evidence (channel.json, rows.json)" />;
  }
  if (channel.status === "error") return <ErrorBox message={channel.message} />;
  if (rows.status === "error") return <ErrorBox message={rows.message} />;

  const rowList = rows.data.rows;
  if (rowList.length === 0) {
    return <UnavailableBox>rows.json contains no qualified panel rows.</UnavailableBox>;
  }

  // Default to the first panel row for display; the URL is only written once
  // the user actively selects a row.
  const selectedRow = rowList.find((r) => r.row_id === state.row) ?? rowList[0];
  const critic: Critic = state.critic;
  const realDmse = channel.data.waterfall.variants.av_real.dmse;
  const teacherDmse = channel.data.waterfall.variants.teacher.dmse;
  const controlLosses = Object.entries(channel.data.real_vs_control.e2.mean_loss).filter(
    ([name]) => name !== "real",
  );
  const bestControlLoss = Math.min(...controlLosses.map(([, value]) => value));
  const matchedRl = channel.data.matched_online_rl;

  return (
    <div className="chan-station">
      <StationBrief
        station="channel"
        question="Can language preserve an activation, and did online RL improve that round trip?"
        status="exploratory"
        statusLabel="matched validation result"
        answer={
          <>
            On the same held-out examples, online RL reduced round-trip directional error by{" "}
            {fmtPct(matchedRl.improvement.nmse_relative)} versus clean SFT. Separate stored-snapshot
            tests show that learned descriptions beat null and mismatched-text controls.
          </>
        }
        note="The RL comparison is validation-only and measures the jointly updated AV actor plus AR critic. It is not a sealed test result or an actor-only attribution."
        metrics={[
          {
            label: "online-RL round-trip dMSE",
            value: fmt(matchedRl.rl.roundtrip_nmse, 3),
            detail: `SFT ${fmt(matchedRl.sft.roundtrip_nmse, 3)} · ${fmtPct(matchedRl.improvement.nmse_relative)} lower`,
          },
          {
            label: "learned-description dMSE",
            value: fmt(realDmse, 3),
            detail: `teacher text ${fmt(teacherDmse, 3)} · lower is better`,
          },
          {
            label: "real AV token loss",
            value: fmt(channel.data.real_vs_control.e2.mean_loss.real, 3),
            detail: `best non-real control ${fmt(bestControlLoss, 3)} · lower is better`,
          },
        ]}
      />

      <div className="station-section-heading">
        <div>
          <span className="station-section-index">01</span>
          <h3>Matched training result</h3>
        </div>
        <p>Same examples and generation protocol; only the selected checkpoint pair changes.</p>
      </div>
      <div className="panel-grid">
        <OnlineRlComparisonPanel channel={channel.data} update={update} />
      </div>

      <div className="controls-row">
        <Segmented
          options={CRITIC_OPTIONS}
          value={critic}
          onChange={(v) => update({ critic: v })}
          label="Critic lens"
        />
        <span className="chan-controls-note">
          The critic lens scores the same learned descriptions with a different AR critic. Choose
          the validation example in the full-text reader below.
        </span>
      </div>

      <div className="panel-grid chan-row-reader-wrap">
        <RowReaderPanel
          channel={channel.data}
          critic={critic}
          rows={rowList}
          selectedRow={selectedRow}
          update={update}
        />
      </div>

      <div className="station-section-heading">
        <div>
          <span className="station-section-index">02</span>
          <h3>Core channel evidence</h3>
        </div>
        <p>Population-level comparisons first; every chart below uses the validation panel.</p>
      </div>
      <div className="panel-grid">
        <WaterfallPanel channel={channel.data} update={update} />
        <RealVsControlPanel channel={channel.data} update={update} />
        <TwinCriticsPanel
          channel={channel.data}
          critic={critic}
          selectedRowId={selectedRow.row_id}
          update={update}
        />
        <CapacityPanel channel={channel.data} update={update} />
      </div>
      <details
        className="station-detail-group"
        open={state.view || state.cell ? true : undefined}
      >
        <summary>
          <span>
            <span className="station-section-index">03</span>
            Explore what the descriptions encode
          </span>
          <small>row retrieval, word budget, attribution, alternate tellings, and rewrites</small>
        </summary>
        <div className="panel-grid">
          <RetrievalPanel
            channel={channel.data}
            critic={critic}
            selectedRowId={selectedRow.row_id}
            update={update}
          />
          <TruncationPanel channel={channel.data} selectedRowId={selectedRow.row_id} />
          <AttributionPanel channel={channel.data} row={selectedRow} />
          <TellingsPanel channel={channel.data} critic={critic} selectedRowId={selectedRow.row_id} />
          <RewritePanel
            channel={channel.data}
            critic={critic}
            row={selectedRow}
            state={state}
            update={update}
          />
        </div>
      </details>
    </div>
  );
}
