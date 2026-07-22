/**
 * BENCH lane model.
 *
 * The control rack is furniture: any resolver that returns the edit lane also
 * returns its paraphrase-placebo and random-edit partners plus the identity
 * ("teacher") cell, so no pane can display the edit lane without its controls.
 * All lane texts are learned descriptions/encodings of a stored activation —
 * never the model reporting on itself.
 */

import type {
  BenchCell,
  BenchIndexRow,
  BenchRowShard,
  Critic,
  SlimMetric,
} from "../../data/types";

export type LaneKey = "edit" | "paraphrase_placebo" | "random_edit" | "teacher";

export const LANE_ORDER: readonly LaneKey[] = [
  "edit",
  "paraphrase_placebo",
  "random_edit",
  "teacher",
];

export const LANE_META: Record<
  LaneKey,
  { label: string; short: string; color: string; role: string }
> = {
  edit: {
    label: "edit",
    short: "edit",
    color: "var(--series-1)",
    role: "semantic edit — the counterfactual under test",
  },
  paraphrase_placebo: {
    label: "paraphrase placebo",
    short: "placebo",
    color: "var(--series-2)",
    role: "control — same wording change, same meaning",
  },
  random_edit: {
    label: "random edit",
    short: "random",
    // Distinct hue: the two control lanes must not share a color (the old
    // series-4/ink-3 pair was indistinguishable, incl. under CVD).
    color: "var(--series-3)",
    role: "control — donor section from an unrelated row",
  },
  teacher: {
    label: "identity (teacher)",
    short: "identity",
    color: "var(--ink-3)",
    role: "harness floor — unedited stored description, re-encoded",
  },
};

/** Canonical chip presentation order; actual chips are derived from the shard. */
export const CHIP_ORDER: readonly string[] = [
  "syntax",
  "discourse",
  "register",
  "final_token",
  "syntax_final",
  "discourse_register",
];

export const TEACHER_CHIP = "teacher";

export function chipLabel(chip: string): string {
  return chip === TEACHER_CHIP ? "identity (teacher)" : chip.replace(/_/g, " ");
}

/** Chips available for a row, derived from its clause_swap variants. */
export function chipsForRow(row: BenchIndexRow): string[] {
  const fam = row.families["clause_swap"];
  if (!fam) return [];
  const found = new Set<string>();
  for (const v of fam.variants) {
    const chip = v.split(":")[0];
    if (chip) found.add(chip);
  }
  const ordered = CHIP_ORDER.filter((c) => found.has(c));
  const extra = [...found].filter((c) => !CHIP_ORDER.includes(c)).sort();
  return [...ordered, ...extra];
}

/** Doses available for a row, derived from its clause_swap variants (a0.5 → "0.5"). */
export function dosesForRow(row: BenchIndexRow): string[] {
  const fam = row.families["clause_swap"];
  if (!fam) return [];
  const found = new Set<string>();
  for (const v of fam.variants) {
    const seg = v.split(":")[2];
    if (seg && seg.startsWith("a")) found.add(seg.slice(1));
  }
  return [...found].sort((a, b) => Number(a) - Number(b));
}

export interface LaneCell {
  lane: LaneKey;
  /** null = the grid has no such precomputed cell; absence stays visible. */
  cell: BenchCell | null;
}

function findCell(shard: BenchRowShard, family: string, variant: string): BenchCell | null {
  return shard.cells.find((c) => c.family === family && c.variant === variant) ?? null;
}

/**
 * Resolve the full rack for a selection. The edit lane is the primary
 * selection; controls and identity always travel with it.
 */
export function resolveRack(shard: BenchRowShard, chip: string, dose: string): LaneCell[] {
  if (chip === TEACHER_CHIP) {
    return [{ lane: "teacher", cell: findCell(shard, "identity", "teacher") }];
  }
  return [
    { lane: "edit", cell: findCell(shard, "clause_swap", `${chip}:edit:a${dose}`) },
    {
      lane: "paraphrase_placebo",
      cell: findCell(shard, "clause_swap", `${chip}:paraphrase_placebo:a${dose}`),
    },
    {
      lane: "random_edit",
      cell: findCell(shard, "clause_swap", `${chip}:random_edit:a${dose}`),
    },
    { lane: "teacher", cell: findCell(shard, "identity", "teacher") },
  ];
}

/**
 * Metrics may carry only the primary critic on clause cells; the independent
 * critic did not score them. Returns null rather than inventing values.
 */
export function criticMetrics(cell: BenchCell, critic: Critic): SlimMetric | null {
  const m = cell.metrics as Partial<Record<Critic, SlimMetric>>;
  return m[critic] ?? null;
}

export function isLaneKey(v: string | null): v is LaneKey {
  return v === "edit" || v === "paraphrase_placebo" || v === "random_edit" || v === "teacher";
}
