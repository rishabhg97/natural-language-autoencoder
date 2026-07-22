/**
 * Typed contracts for the generated static data shards under public/data/.
 * These mirror scripts/build_static_data.py exactly; the builder fails closed,
 * so the app may trust these shapes after the manifest check passes.
 */

export type Critic = "primary" | "independent";

export type EvidenceStatus = "qualified" | "exploratory" | "negative" | "unavailable";

export type Family =
  | "identity"
  | "clause_swap"
  | "section_ablation"
  | "word_occlusion"
  | "truncation"
  | "paraphrase"
  | "corruption"
  | "alternate_telling";

export type Lane = "edit" | "paraphrase_placebo" | "random_edit";

export interface SlimMetric {
  dmse: number;
  raw_mse: number;
  cosine: number;
  norm_ratio: number;
}

/* ------------------------------ manifest.json ------------------------------ */

export interface ManifestFile {
  path: string;
  sha256: string;
  bytes: number;
  schema_version: string;
}

export interface DashboardManifest {
  schema_version: string;
  generated_at: string;
  source: {
    bundle_id: string;
    source_config_sha256: string;
    bundle_config_sha256: string;
    population: string;
    split: string;
    counts: Record<string, number>;
    manifest_sha256: string;
    excluded_files: string[];
  };
  poetry: { config_sha256: string; phases_passed: string[] };
  online_rl: {
    status: "validation_only_matched";
    row_count: number;
    independent_family_count: number;
    max_new_tokens: number;
    generation_protocol_sha256: string;
    report_sha256: { sft: string; rl: string };
  };
  tokenizer: {
    path: string;
    sha256: string;
    vocab_size: number;
    spot_check: { tokens: number; mismatches: number };
  };
  counts: Record<string, number>;
  files: ManifestFile[];
}

/* -------------------------------- rows.json -------------------------------- */

export interface RowRecord {
  row_id: string;
  row_index: number;
  doc_id: string;
  content_family_id: string;
  n_raw_tokens: number;
  token_position: number;
  activation_norm: number;
  source_text: string;
  teacher_text: string;
  av_text: string;
  release_status: string;
  claim_scope: string;
  stratum: Record<string, unknown>;
}

export interface RowsShard {
  schema_version: string;
  kind: "rows";
  rows: RowRecord[];
}

/* ------------------------------ channel.json ------------------------------- */

export interface AggregateEntry {
  mean: number;
  ci_low: number;
  ci_high: number;
  rows: number;
  families: number;
  bootstrap_samples: number;
}

export interface CourtThreshold {
  threshold: number;
  balanced_accuracy: number;
  positive_recall: number;
  negative_recall: number;
}

export interface IdentityMetricRow extends SlimMetric {
  row_id: string;
  critic: Critic;
}

export interface TwinCriticRow {
  row_id: string;
  primary_dmse: number;
  independent_dmse: number;
  primary_cosine: number;
  independent_cosine: number;
}

export interface RetrievalRow {
  row_id: string;
  critic: Critic;
  rank: number;
  nearest_row_id: string;
  expected_cosine: number;
}

export interface WaterfallVariant {
  dmse: number;
  cosine_mean: number;
  ci_low: number;
  ci_high: number;
  rows: number;
  families: number;
  norm_ratio_mean: number;
}

export interface CapacityLadderRung {
  gallery_size: number;
  gallery_bits: number;
  top1_accuracy: number;
  top5_accuracy: number;
  median_rank: number;
  mean_reciprocal_rank: number;
  fano_information_lower_bound_bits: number;
}

export interface TruncationPoint {
  fraction: number;
  words: number;
  dmse: number;
  cosine: number;
}

export interface OcclusionCell {
  word_index: number;
  word: string;
  char_start: number;
  char_end: number;
  dmse: number;
  d_dmse: number;
}

export interface TellingSample {
  cell_id: string;
  sample_index: number;
  text: string;
  dmse: number;
  cosine: number;
}

export interface MatchedOnlineRlControl {
  key: string;
  label: string;
  roundtrip_nmse: number;
}

export interface MatchedOnlineRlStage {
  roundtrip_nmse: number;
  raw_mse: number;
  cosine: number;
  centered_r2: number;
  norm_ratio: number;
  teacher_nmse: number;
  teacher_win_count: number;
  teacher_win_fraction: number;
  parse: {
    closed_count: number;
    closed_fraction: number;
    usable_count: number;
    usable_fraction: number;
    row_count: number;
  };
  controls: MatchedOnlineRlControl[];
}

export interface MatchedOnlineRlResult {
  status: "validation_only_matched";
  row_count: number;
  independent_family_count: number;
  max_new_tokens: number;
  generation_protocol_sha256: string;
  sft: MatchedOnlineRlStage;
  rl: MatchedOnlineRlStage;
  improvement: {
    nmse_absolute: number;
    nmse_relative: number;
    raw_mse_absolute: number;
    raw_mse_relative: number;
    teacher_win_fraction_gain: number;
  };
  source_reports: { sft: string; rl: string };
  scope_note: string;
}

export interface ChannelShard {
  schema_version: string;
  kind: "channel";
  matched_online_rl: MatchedOnlineRlResult;
  aggregates: Record<string, AggregateEntry>;
  court_thresholds: Record<Critic, CourtThreshold>;
  fit_split: string;
  identity: IdentityMetricRow[];
  twin_critics: {
    per_row: TwinCriticRow[];
    e3_summaries: Record<Critic, Record<string, number>>;
    p2_summaries: Record<Critic, Record<string, number>>;
    confound: string;
  };
  retrieval: RetrievalRow[];
  waterfall: {
    metric: string;
    split: string;
    variants: Record<string, WaterfallVariant>;
    source_report: string;
  };
  capacity_ladder: {
    ladder: CapacityLadderRung[];
    assumptions: Record<string, string>;
    top_confusions: { count: number; retrieved_family: string; source_family: string }[];
    distance: string;
    variant: string;
    source_report: string;
  };
  real_vs_control: {
    e1_av: {
      losses: Record<string, number>;
      rows: number;
      parse: Record<string, { closed_fraction: number; usable_fraction: number }>;
      source_report: string;
    };
    e2: {
      mean_loss: Record<string, number>;
      paired: Record<string, { mean_real_minus_control: number; real_win_fraction: number }>;
      rows: number;
      records: number;
      source_report: string;
    };
  };
  truncation: Record<string, TruncationPoint[]>;
  occlusion: Record<string, OcclusionCell[]>;
  tellings: Record<string, TellingSample[]>;
  shapley: Record<string, { sections: Record<string, number>; efficiency_error: number }>;
}

/* ------------------------------ rewrites.json ------------------------------ */

export interface RewriteCourt {
  identity_cosine: number;
  calibration_label: "positive" | "negative" | "context";
  semanticity_verdict: boolean;
}

export interface RewriteCell {
  cell_id: string;
  row_id: string;
  family: "paraphrase" | "corruption";
  variant: string;
  text: string;
  spec: { kind: string; rate?: number; semantic_intent?: string };
  metrics: Record<Critic, SlimMetric>;
  court: Record<Critic, RewriteCourt>;
}

export interface RewritesShard {
  schema_version: string;
  kind: "rewrites";
  identity: { row_id: string; metrics: Record<Critic, SlimMetric>; text: string }[];
  cells: RewriteCell[];
}

/* -------------------------------- trace.json ------------------------------- */

export interface TracePosition {
  position: number;
  n_context_tokens: number;
  token_id: number;
  token_text: string;
  source_alignment: "exact" | "unavailable";
  source_char_start: number | null;
  source_char_end: number | null;
  source_before: string;
  source_token: string;
  source_after: string;
  source_prefix_omitted: boolean;
  source_suffix_omitted: boolean;
  description: string;
  parse_state: string;
  usable: boolean;
}

export interface TraceDoc {
  row_id: string;
  doc_id: string;
  content_family_id: string;
  positions: TracePosition[];
  drift: { one_minus_cos: number; relative_l2: number; rms_ratio: number; max_abs: number };
}

export interface TraceShard {
  schema_version: string;
  kind: "trace";
  claim_scope: string;
  boundary: number;
  shuffled_control: { available: boolean; note: string };
  source_alignment: { exact_positions: number; unavailable_positions: number; note: string };
  docs: TraceDoc[];
}

/* ------------------------------- poetry.json ------------------------------- */

export interface PoetryAnalysisToken {
  position: number;
  relative_offset: number;
  token_id: number;
  token_text: string;
}

export interface PoetryPositionScore {
  position: number;
  relative_offset: number;
  variant: "real" | "shuffled";
  samples: number;
  usable_rate: number;
  target_exact_rate: number;
  target_family_rate: number;
  alternate_family_rate: number;
}

export interface PoetrySample {
  position: number;
  relative_offset: number;
  variant: "real" | "shuffled";
  sample_index: number;
  source_case_id: string;
  usable: boolean;
  target_exact: boolean;
  target_family: boolean;
  alternate_family: boolean;
  explanation: string;
  parse: { closed: boolean; extraction_mode: string; repetition_loop: boolean };
}

export interface PoetryIntervention {
  direction: "edited" | "random";
  dose: number;
  continuation_text: string;
  hits_target_family: boolean;
  hits_alternate_family: boolean;
}

export interface PoetryCase {
  case_id: string;
  framing: string;
  first_line: string;
  second_line: string;
  prefix_text: string;
  cue: string;
  target_word: string;
  target_terms: string[];
  alternate_terms: string[];
  edit_map: Record<string, string>;
  anchor_position: number;
  analysis: PoetryAnalysisToken[];
  baseline_continuation: string;
  baseline_hits_target_family: boolean;
  planning_onset_position: number | null;
  planning_onset_relative_offset: number | null;
  anchor_lift: number;
  anchor_real_target_family_rate: number;
  anchor_shuffled_target_family_rate: number;
  position_scores: PoetryPositionScore[];
  samples: PoetrySample[];
  reconstruction: {
    original_explanation: string;
    edited_explanation: string;
    changed_terms: string[];
    original_cosine: number;
    original_dmse: number;
    edit_delta_norm: number;
  } | null;
  interventions: PoetryIntervention[];
}

export interface PoetryShard {
  schema_version: string;
  kind: "poetry";
  claim_scope: string;
  config_sha256: string;
  gates: { planning_onset_rate: number; minimum_usable_fraction: number };
  aggregates: {
    cases: number;
    positions: number;
    samples: number;
    usable_fraction: number;
    mean_anchor_lift: number;
    cases_with_planning_onset: number;
    cases_with_baseline_target_rhyme: number;
    editable_cases: number;
    steering_doses: number[];
    edited_alternate_hit_rate: number;
    random_alternate_hit_rate: number;
    mean_original_dmse: number;
  };
  interpretation: { signal: string; notes: string[] };
  reports: Record<string, { passed: boolean; config_sha256: string }>;
  cases: PoetryCase[];
}

/* ---------------------------- bench/index.json ----------------------------- */

export interface BenchIndexRow {
  row_id: string;
  has_behavior: boolean;
  families: Record<string, { variants: string[]; depths: string[] }>;
}

export interface BenchControlGroup {
  row_id: string;
  chip: string;
  cells: Record<Lane, Record<string, string>>;
}

export interface BenchIndexShard {
  schema_version: string;
  kind: "bench_index";
  banner: {
    statement: string;
    total_cells: number;
    behavior_cells: number;
    behavior_rows: number;
    grid_spec_sha256: string;
    claim_scope: string;
    functional_claim_status: string;
  };
  rows: BenchIndexRow[];
  behavior_rows: string[];
  control_groups: Record<string, BenchControlGroup>;
}

/* --------------------------- bench/row-*.json ------------------------------ */

export interface TopKToken {
  id: number;
  text: string;
  p: number;
}

export interface WakePoint {
  offset: number;
  js: number;
  kl: number;
  top_10_overlap: number;
  top_50_overlap: number;
}

export interface BenchCell {
  cell_id: string;
  family: string;
  variant: string;
  depth: "METRIC" | "BEHAVIOR";
  control_group_id: string | null;
  spec: Record<string, unknown>;
  text: string;
  metrics: Record<Critic, SlimMetric> | { primary: SlimMetric };
  geometry: { x: number; y: number; z: number } | null;
  behavior: {
    js_divergence: number;
    kl_original_to_patched: number;
    logit_pearson: number;
    top_10_overlap: number;
    top_50_overlap: number;
    original_top1_rank: number;
    vocab_size: number;
  };
  topk: { original: TopKToken[]; patched: TopKToken[] };
  wake: WakePoint[];
  baseline_continuation: string;
  patched_continuation: string;
  generation_protocol: Record<string, unknown>;
}

export interface BenchRowShard {
  schema_version: string;
  kind: "bench_row";
  row_id: string;
  claim_scope: string;
  target_geometry: { x: number; y: number; z: number } | null;
  cells: BenchCell[];
}

/* -------------------------------- audit.json ------------------------------- */

export interface CourtDocketCell {
  variant: string;
  identity_cosine: number;
  label: "positive" | "negative" | "context";
  verdict: boolean;
}

export interface CourtDocketRow {
  row_id: string;
  critic: Critic;
  paraphrase_min_identity_cosine: number;
  paraphrase_mean_identity_cosine: number;
  paraphrase_verdicts_true: number;
  paraphrase_cells: CourtDocketCell[];
  corruption_cells: CourtDocketCell[];
  row_verdict: "honest" | "mixed" | "suspect";
}

export interface NegativeResult {
  id: string;
  status: "weak" | "negative" | "caveat";
  statement: string;
  source: string;
}

export interface AuditShard {
  schema_version: string;
  kind: "audit";
  claim_ledger: { claims: Record<string, string>; limitations: string[] };
  evidence_status_legend: Record<EvidenceStatus, string>;
  provenance: {
    bundle_id: string;
    source_config_sha256: string;
    bundle_config_sha256: string;
    population: string;
    split: string;
    counts: Record<string, number>;
    files: ManifestFile[];
    excluded_files: string[];
    report_bindings: Record<string, { source_path: string; sha256: string }>;
    code_bindings: Record<string, string>;
    runtime: Record<string, string>;
    privacy_card: Record<string, unknown>;
    source_provenance: Record<string, string>;
    tokenizer: DashboardManifest["tokenizer"];
  };
  court: {
    thresholds: Record<Critic, CourtThreshold>;
    fit_split: string;
    confound: string;
    docket: CourtDocketRow[];
  };
  parse_health: {
    explanations_by_kind: Record<string, Record<string, number>>;
    trace_descriptions_usable: boolean;
    e1_av_parse: Record<string, { closed_fraction: number; usable_fraction: number }>;
    almanac_parse_health: Record<string, { closed_fraction: number; usable_fraction: number }>;
    poetry_usable_fraction: number;
  };
  drift: {
    card: Record<string, unknown>;
    e5_per_doc: { row_id: string; one_minus_cos: number; relative_l2: number }[];
  };
  magnitude: {
    claim_boundary: string;
    publication_status: string;
    fit: Record<string, unknown>;
  };
  null_text: {
    scope: string;
    row_count: number;
    real_enriched_words: { token: string; log_odds_real_vs_zero: number }[];
    zero_enriched_words: { token: string; log_odds_real_vs_zero: number }[];
    e1_av_losses: Record<string, number>;
    e2_mean_loss: Record<string, number>;
    e2_paired: Record<string, { mean_real_minus_control: number; real_win_fraction: number }>;
    backfill_note: string;
  };
  negative_results: NegativeResult[];
  poetry_status: {
    claim_scope: string;
    pipeline_passed: boolean;
    pipeline_note: string;
    config_sha256: string;
  };
}
