/**
 * Panel 3f — dose comparison. For the selected chip, KL and JS at every
 * available dose per lane as slope (dumbbell) charts: a graded edit-lane
 * effect with flat placebo lanes is the pattern of interest — but whatever
 * the data shows is what renders.
 */

import { useMemo } from "react";
import type { BenchRowShard, WakePoint } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { fmt } from "../../data/format";
import { Dot, linePath, makeFrame, Axes } from "../../components/charts";
import { LANE_META, resolveRack, TEACHER_CHIP, type LaneKey } from "./laneModel";
import { LaneLegend } from "./benchUi";

type DoseMetric = Extract<keyof WakePoint, "kl" | "js">;
const METRICS: { key: DoseMetric; behaviorKey: "kl_original_to_patched" | "js_divergence"; label: string }[] = [
  { key: "kl", behaviorKey: "kl_original_to_patched", label: "KL(orig → patched)" },
  { key: "js", behaviorKey: "js_divergence", label: "JS divergence" },
];

const DOSE_LANES: readonly LaneKey[] = ["edit", "paraphrase_placebo", "random_edit"];
const W = 320;
const H = 170;

interface Series {
  lane: LaneKey;
  points: { dose: number; value: number }[];
}

function SlopeChart(props: { label: string; series: Series[]; doses: string[] }) {
  const { label, series, doses } = props;
  const frame = useMemo(() => {
    const values = series.flatMap((s) => s.points.map((p) => p.value));
    const doseNums = doses.map(Number);
    const lo = Math.min(...doseNums);
    const hi = Math.max(...doseNums);
    const span = hi - lo || 1;
    return makeFrame({
      width: W,
      height: H,
      xDomain: [lo - span * 0.18, hi + span * 0.18],
      yDomain: [0, Math.max(1e-9, ...values) * 1.15],
      margin: { left: 44, right: 52, bottom: 28 },
    });
  }, [series, doses]);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="chart-svg bench-chart"
      role="img"
      aria-label={`${label} at dose ${doses.join(" vs ")} per lane; exact values labeled at each point`}
    >
      <Axes
        frame={frame}
        xLabel="dose"
        yLabel={label}
        xTicks={doses.map(Number)}
        xFormat={(v) => String(v)}
        yFormat={(v) => fmt(v, 2)}
      />
      {(() => {
        // Label selectively: coincident labels are dropped (first series
        // wins); every value stays reachable via the point title/aria-label.
        const placed: { x: number; y: number }[] = [];
        const showLabel = (px: number, py: number): boolean => {
          if (placed.some((q) => Math.abs(q.x - px) < 42 && Math.abs(q.y - py) < 12)) {
            return false;
          }
          placed.push({ x: px, y: py });
          return true;
        };
        return series.map((s) => (
        <g key={s.lane}>
          <path
            d={linePath(frame, s.points.map((p) => ({ x: p.dose, y: p.value })))}
            fill="none"
            stroke={LANE_META[s.lane].color}
            strokeWidth={2}
          />
          {s.points.map((p, i) => (
            <g
              key={p.dose}
              role="img"
              aria-label={`${LANE_META[s.lane].label} at dose ${p.dose}: ${fmt(p.value, 4)}`}
            >
              <title>{`${LANE_META[s.lane].label} at dose ${p.dose}: ${fmt(p.value, 4)}`}</title>
              <Dot
                cx={frame.x(p.dose)}
                cy={frame.y(p.value)}
                r={4}
                fill={LANE_META[s.lane].color}
              />
              {showLabel(frame.x(p.dose), frame.y(p.value)) ? (
                <text
                  x={frame.x(p.dose) + (i === s.points.length - 1 ? 7 : -7)}
                  y={frame.y(p.value) - 5}
                  textAnchor={i === s.points.length - 1 ? "start" : "end"}
                >
                  {fmt(p.value, 2)}
                </text>
              ) : null}
            </g>
          ))}
        </g>
        ));
      })()}
    </svg>
  );
}

export default function DosePanel(props: { shard: BenchRowShard; chip: string; doses: string[] }) {
  const { shard, chip, doses } = props;

  const data = useMemo(() => {
    if (chip === TEACHER_CHIP || doses.length < 2) return null;
    const racks = doses.map((dose) => ({ dose, rack: resolveRack(shard, chip, dose) }));
    return METRICS.map((metric) => ({
      metric,
      series: DOSE_LANES.map((lane) => ({
        lane,
        points: racks
          .map(({ dose, rack }) => {
            const cell = rack.find((l) => l.lane === lane)?.cell;
            return cell ? { dose: Number(dose), value: cell.behavior[metric.behaviorKey] } : null;
          })
          .filter((p): p is { dose: number; value: number } => p !== null),
      })).filter((s) => s.points.length > 0),
    }));
  }, [shard, chip, doses]);

  return (
    <Panel
      title="Dose comparison"
      span={4}
      badges={<Badge status="exploratory" label="functional: validation-only" />}
      sub="is the effect graded with dose? placebo lanes should stay flat. Identity (teacher) has no dose axis — it is the floor in the other panes."
    >
      {chip === TEACHER_CHIP ? (
        <UnavailableBox>
          dose comparison needs an intervention chip; the identity lane carries no dose.
        </UnavailableBox>
      ) : !data || doses.length < 2 ? (
        <UnavailableBox>
          fewer than two doses exist for this chip on this row — no dose gradient can be shown.
        </UnavailableBox>
      ) : (
        <>
          <LaneLegend lanes={DOSE_LANES} />
          {doses.length === 2 ? (
            <p className="bench-note">
              Only two doses exist for this chip — read the lines as endpoints, not a fitted curve.
            </p>
          ) : null}
          {data.map(({ metric, series }) =>
            series.length > 0 ? (
              <SlopeChart key={metric.key} label={metric.label} series={series} doses={doses} />
            ) : (
              <UnavailableBox key={metric.key}>
                no cells resolved for {metric.label} at these doses.
              </UnavailableBox>
            ),
          )}
        </>
      )}
    </Panel>
  );
}
