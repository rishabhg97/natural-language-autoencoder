/**
 * BENCH — precomputed counterfactuals.
 *
 * Selection tuple (all restored from the URL): state.row = panel row,
 * state.variant = intervention chip ("syntax", …, or "teacher"),
 * state.dose = "0.5" | "1". The edit-lane cell is the primary selection; its
 * paraphrase-placebo + random-edit partners at the same dose and the identity
 * ("teacher") cell always load with it. state.cell carries the focused lane
 * for the top-k/continuation panes and state.view the wake metric.
 */

import { useEffect, useMemo } from "react";
import "./bench.css";
import type { StationProps } from "../../app/stationProps";
import type { AppState } from "../../app/urlState";
import { useShard } from "../../data/loader";
import type { BenchIndexRow, BenchIndexShard, BenchRowShard } from "../../data/types";
import {
  ErrorBox,
  LoadingBox,
  Panel,
  StationBrief,
  UnavailableBox,
} from "../../components/ui";
import { shortRowId } from "../../data/format";
import {
  chipsForRow,
  dosesForRow,
  isLaneKey,
  resolveRack,
  TEACHER_CHIP,
  type LaneKey,
} from "./laneModel";
import { useLedger, type LedgerEntry } from "./useLedger";
import BannerBar from "./BannerBar";
import PickerPanel from "./PickerPanel";
import ControlRack from "./ControlRack";
import CompassPanel from "./CompassPanel";
import TopKPanel from "./TopKPanel";
import GaugesPanel from "./GaugesPanel";
import WakePanel, { isWakeMetric, type WakeMetric } from "./WakePanel";
import ContinuationsPanel from "./ContinuationsPanel";
import DosePanel from "./DosePanel";
import LedgerPanel from "./LedgerPanel";
import { LinkButton } from "./benchUi";

interface Selection {
  row: BenchIndexRow;
  chip: string;
  dose: string;
  chips: string[];
  doses: string[];
  focusLane: LaneKey;
  wakeMetric: WakeMetric;
}

function resolveSelection(index: BenchIndexShard, state: AppState): Selection | null {
  const rows = index.rows;
  if (rows.length === 0) return null;
  const row =
    rows.find((r) => r.row_id === state.row) ??
    rows.find((r) => r.row_id === index.behavior_rows[0]) ??
    rows[0];
  const chips = chipsForRow(row);
  const doses = dosesForRow(row);
  const chip =
    state.variant && (chips.includes(state.variant) || state.variant === TEACHER_CHIP)
      ? state.variant
      : chips[0] ?? TEACHER_CHIP;
  const dose =
    state.dose && doses.includes(state.dose)
      ? state.dose
      : doses.includes("1")
        ? "1"
        : doses[doses.length - 1] ?? "1";
  const focusLane: LaneKey =
    chip === TEACHER_CHIP
      ? "teacher"
      : isLaneKey(state.cell)
        ? state.cell
        : "edit";
  const wakeMetric: WakeMetric = isWakeMetric(state.view) ? state.view : "kl";
  return { row, chip, dose, chips, doses, focusLane, wakeMetric };
}

/** Behavior rows: load the per-row shard lazily, then render the rack + panes. */
function BehaviorPanes(props: {
  selection: Selection;
  update: (patch: Partial<AppState>) => void;
  record: (entry: Omit<LedgerEntry, "ts">) => void;
}) {
  const { selection, update, record } = props;
  const { row, chip, dose } = selection;
  const shardPath = `bench/row-${row.row_id}.json`;
  const rowShard = useShard<BenchRowShard>(shardPath);

  const rack = useMemo(
    () => (rowShard.status === "ready" ? resolveRack(rowShard.data, chip, dose) : null),
    [rowShard, chip, dose],
  );

  // Ledger: append every opened experiment (cell ids + selection tuple only).
  useEffect(() => {
    if (!rack) return;
    const cells: Record<string, string> = {};
    for (const laneCell of rack) {
      if (laneCell.cell) cells[laneCell.lane] = laneCell.cell.cell_id;
    }
    record({
      row: row.row_id,
      chip,
      dose: chip === TEACHER_CHIP ? null : dose,
      cells,
    });
  }, [rack, row.row_id, chip, dose, record]);

  if (rowShard.status === "loading") {
    return (
      <Panel title="Row evidence" span={12}>
        <LoadingBox what={shardPath} />
      </Panel>
    );
  }
  if (rowShard.status === "error") {
    return (
      <Panel title="Row evidence" span={12}>
        <ErrorBox message={rowShard.message} />
      </Panel>
    );
  }
  if (!rack || rack.every((l) => l.cell === null)) {
    return (
      <Panel title="Row evidence" span={12}>
        <UnavailableBox>
          the grid holds no cells for {chip === TEACHER_CHIP ? "the identity move" : `${chip} @ dose ${dose}`}{" "}
          on {shortRowId(row.row_id)} — absence stays visible; nothing is substituted.
        </UnavailableBox>
      </Panel>
    );
  }

  return (
    <>
      <ControlRack rack={rack} rowId={row.row_id} chip={chip} dose={dose} update={update} />
      <CompassPanel rack={rack} target={rowShard.data.target_geometry} rowId={row.row_id} />
      <TopKPanel rack={rack} focusLane={selection.focusLane} chip={chip} update={update} />
      <GaugesPanel rack={rack} />
      <WakePanel rack={rack} metric={selection.wakeMetric} update={update} />
      <ContinuationsPanel rack={rack} focusLane={selection.focusLane} update={update} />
      <DosePanel shard={rowShard.data} chip={chip} doses={selection.doses} />
    </>
  );
}

/**
 * METRIC-only rows: this bundle publishes per-cell shards (geometry, top-k,
 * wake, continuations) only for the designated behavior rows, so every
 * consequence pane shows explicit absence rather than a placeholder.
 */
function MetricOnlyPanes(props: {
  rowId: string;
  behaviorRows: number;
  update: (patch: Partial<AppState>) => void;
}) {
  const { rowId, behaviorRows, update } = props;
  const channelLink = (
    <LinkButton
      onClick={() => update({ station: "channel", row: rowId })}
      title="Reconstruction metrics for this row's METRIC families live on CHANNEL"
    >
      view row on CHANNEL
    </LinkButton>
  );
  // First pane states the coverage boundary in full; the rest stay visible
  // but reference it instead of repeating the same sentence six times.
  const absence = (what: string) => (
    <UnavailableBox>
      {shortRowId(rowId)} has METRIC-depth cells only, and this bundle ships per-cell shards
      solely for the {behaviorRows} behavior rows — no {what} exists for it here. {channelLink}
    </UnavailableBox>
  );
  const absenceShort = (what: string) => (
    <UnavailableBox>same coverage boundary as above — no {what} for this row.</UnavailableBox>
  );
  return (
    <>
      <Panel title="Control rack — lanes travel together" span={12}>
        {absence("edit/placebo/random/identity behavior rack")}
      </Panel>
      <Panel
        title="Reconstruction compass"
        span={6}
        sub="validation-fitted PCA projection — display only; metrics are computed in native 2688-d."
      >
        {absenceShort("precomputed cell geometry")}
      </Panel>
      <Panel title="Next-token top-k movement" span={6}>
        {absenceShort("stored top-k readout")}
      </Panel>
      <Panel title="Divergence gauges" span={6}>
        {absenceShort("stored divergence readout")}
      </Panel>
      <Panel title="Causal wake" span={6}>
        {absenceShort("stored wake trajectory")}
      </Panel>
      <Panel title="Continuations — baseline vs patched" span={8}>
        {absenceShort("stored continuation pair")}
      </Panel>
      <Panel title="Dose comparison" span={4}>
        {absenceShort("dose gradient")}
      </Panel>
    </>
  );
}

export default function BenchStation({ state, update }: StationProps) {
  const index = useShard<BenchIndexShard>("bench/index.json");
  const ledger = useLedger();

  if (index.status === "loading") {
    return <LoadingBox what="bench grid index (bench/index.json)" />;
  }
  if (index.status === "error") {
    return <ErrorBox message={index.message} />;
  }

  const selection = resolveSelection(index.data, state);
  if (!selection) {
    return (
      <UnavailableBox>the bench grid index lists no rows — nothing can be selected.</UnavailableBox>
    );
  }

  return (
    <div className="bench-station">
      <StationBrief
        station="bench"
        question="Does changing the language encoding cause a distinct model change?"
        status="exploratory"
        statusLabel="functional · validation only"
        answer="The workbench compares each semantic edit with a same-meaning paraphrase, a norm-matched random direction, and the unedited teacher encoding. A useful causal result must separate from those controls, not merely move the model."
        note="Every choice below opens a precomputed experiment. No model is running and no intervention is recomputed in the browser."
        metrics={[
          {
            label: "precomputed experiments",
            value: index.data.banner.total_cells.toLocaleString(),
            detail: "complete validation intervention lattice",
          },
          {
            label: "functional readouts",
            value: index.data.banner.behavior_cells.toLocaleString(),
            detail: `across ${index.data.banner.behavior_rows} behavior rows`,
          },
          {
            label: "required control lanes",
            value: "3 + identity",
            detail: "edit · paraphrase · random · teacher",
          },
        ]}
      />
      <BannerBar banner={index.data.banner} update={update} />
      <div className="panel-grid">
        <PickerPanel
          index={index.data}
          row={selection.row}
          chip={selection.chip}
          dose={selection.dose}
          chips={selection.chips}
          doses={selection.doses}
          update={update}
        />
        {selection.row.has_behavior ? (
          <BehaviorPanes selection={selection} update={update} record={ledger.record} />
        ) : (
          <MetricOnlyPanes
            rowId={selection.row.row_id}
            behaviorRows={index.data.banner.behavior_rows}
            update={update}
          />
        )}
        <LedgerPanel entries={ledger.entries} onClear={ledger.clear} />
      </div>
    </div>
  );
}
