/**
 * Panel 1 — row + move picker with a coverage map. Rows with BEHAVIOR-depth
 * cells lead the list; METRIC-only rows stay selectable but announce their
 * missing behavior coverage instead of hiding it.
 */

import { useMemo } from "react";
import type { BenchIndexRow, BenchIndexShard, RowsShard } from "../../data/types";
import type { AppState } from "../../app/urlState";
import { useShard } from "../../data/loader";
import { Panel, Segmented, SelectControl, UnavailableBox } from "../../components/ui";
import { preview, shortRowId } from "../../data/format";
import { chipLabel, TEACHER_CHIP } from "./laneModel";

function CoverageMap(props: { row: BenchIndexRow }) {
  const families = useMemo(
    () => Object.entries(props.row.families).sort(([a], [b]) => a.localeCompare(b)),
    [props.row],
  );
  return (
    <div className="bench-scroll">
      <table className="data-table bench-coverage" tabIndex={0} aria-label="Precomputed experiment coverage map">
        <caption>
          Coverage map for {shortRowId(props.row.row_id)}: which precomputed cells exist, at
          which depth. METRIC = reconstruction metrics only; BEHAVIOR = full functional
          readout (top-k, wake, continuations).
        </caption>
        <thead>
          <tr>
            <th scope="col">family</th>
            <th scope="col" className="num">
              METRIC cells
            </th>
            <th scope="col" className="num">
              BEHAVIOR cells
            </th>
            <th scope="col">variants</th>
          </tr>
        </thead>
        <tbody>
          {families.map(([family, info]) => {
            const isBehavior = info.depths.includes("BEHAVIOR");
            const isMetric = info.depths.includes("METRIC");
            const previewList = info.variants.slice(0, 4).join(", ");
            const more = info.variants.length > 4 ? ` … +${info.variants.length - 4}` : "";
            return (
              <tr key={family}>
                <td className="mono">{family}</td>
                <td className="num">{isMetric ? info.variants.length : "—"}</td>
                <td className="num">{isBehavior ? info.variants.length : "—"}</td>
                <td className="bench-variant-preview" title={info.variants.join(", ")}>
                  {previewList}
                  {more}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function PickerPanel(props: {
  index: BenchIndexShard;
  row: BenchIndexRow;
  chip: string;
  dose: string;
  chips: string[];
  doses: string[];
  update: (patch: Partial<AppState>) => void;
}) {
  const { index, row, chip, dose, chips, doses, update } = props;
  const rows = useShard<RowsShard>("rows.json");
  const rowRecord = useMemo(
    () =>
      rows.status === "ready"
        ? (rows.data.rows.find((r) => r.row_id === row.row_id) ?? null)
        : null,
    [rows, row.row_id],
  );

  const rowOptions = useMemo(() => {
    const sorted = [...index.rows].sort(
      (a, b) => Number(b.has_behavior) - Number(a.has_behavior) || a.row_id.localeCompare(b.row_id),
    );
    return sorted.map((r) => ({
      value: r.row_id,
      label: `${shortRowId(r.row_id)} — ${r.has_behavior ? "BEHAVIOR" : "metric-only"}`,
    }));
  }, [index.rows]);

  const chipOptions = useMemo(() => [...chips, TEACHER_CHIP], [chips]);

  return (
    <Panel
      title="Row + move picker"
      span={12}
      sub="Pick a stored row, then an intervention chip and dose. Every choice selects a precomputed grid cell — nothing is recomputed."
    >
      <div className="bench-controls">
        <SelectControl
          label="row"
          value={row.row_id}
          options={rowOptions}
          onChange={(v) => update({ row: v })}
        />
        {row.has_behavior ? (
          <>
            <Segmented
              options={chipOptions}
              value={chipOptions.includes(chip) ? chip : chipOptions[0]}
              onChange={(v) => update({ variant: v })}
              label="intervention chip"
              format={chipLabel}
            />
            {chip === TEACHER_CHIP ? (
              <span className="bench-note">identity view — dose does not apply</span>
            ) : doses.length > 0 ? (
              <Segmented
                options={doses}
                value={doses.includes(dose) ? dose : doses[doses.length - 1]}
                onChange={(v) => update({ dose: v })}
                label="dose"
                format={(d) => `dose ${d}`}
              />
            ) : null}
          </>
        ) : (
          <UnavailableBox>
            no BEHAVIOR cells for this row — the grid holds METRIC-depth cells only, so the
            consequence panes below show explicit absence rather than substitutes.
          </UnavailableBox>
        )}
      </div>
      {rowRecord ? (
        <p className="bench-note bench-row-preview">
          Row source: “{preview(rowRecord.source_text, 160)}” — activation at token{" "}
          {rowRecord.token_position} of {rowRecord.n_raw_tokens}.
        </p>
      ) : null}
      <CoverageMap row={row} />
    </Panel>
  );
}
