/**
 * Chart primitives: an SVG frame with linear axes, plus a shared tooltip.
 * Marks follow the dataviz specs: bars <= 24px with 4px rounded data ends,
 * 2px lines, >= 8px markers with a 2px surface ring, hairline solid grid.
 */

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { linearScale, ticks } from "../data/format";

/* --------------------------------- tooltip -------------------------------- */

export interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

const TooltipCtx = createContext<{
  show: (t: TooltipState) => void;
  hide: () => void;
}>({ show: () => {}, hide: () => {} });

export function TooltipProvider(props: { children: ReactNode }) {
  const [tip, setTip] = useState<TooltipState | null>(null);
  const show = useCallback((t: TooltipState) => setTip(t), []);
  const hide = useCallback(() => setTip(null), []);
  const value = useMemo(() => ({ show, hide }), [show, hide]);
  return (
    <TooltipCtx.Provider value={value}>
      {props.children}
      {tip ? (
        <div
          className="viz-tooltip"
          role="tooltip"
          style={{
            left: Math.min(tip.x + 12, window.innerWidth - 360),
            top: Math.min(tip.y + 12, window.innerHeight - 120),
          }}
        >
          {tip.content}
        </div>
      ) : null}
    </TooltipCtx.Provider>
  );
}

export function useTooltip() {
  return useContext(TooltipCtx);
}

/* ---------------------------------- frame --------------------------------- */

export interface Frame {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  x: (v: number) => number;
  y: (v: number) => number;
  xDomain: [number, number];
  yDomain: [number, number];
  innerWidth: number;
  innerHeight: number;
}

export function makeFrame(opts: {
  width: number;
  height: number;
  xDomain: [number, number];
  yDomain: [number, number];
  margin?: Partial<Frame["margin"]>;
}): Frame {
  const margin = { top: 8, right: 12, bottom: 26, left: 44, ...opts.margin };
  const innerWidth = Math.max(10, opts.width - margin.left - margin.right);
  const innerHeight = Math.max(10, opts.height - margin.top - margin.bottom);
  return {
    width: opts.width,
    height: opts.height,
    margin,
    xDomain: opts.xDomain,
    yDomain: opts.yDomain,
    innerWidth,
    innerHeight,
    x: linearScale(opts.xDomain, [margin.left, margin.left + innerWidth]),
    y: linearScale(opts.yDomain, [margin.top + innerHeight, margin.top]),
  };
}

export function Axes(props: {
  frame: Frame;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number[];
  yTicks?: number[];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
}) {
  const { frame } = props;
  const xs = props.xTicks ?? ticks(frame.xDomain, 5);
  const ys = props.yTicks ?? ticks(frame.yDomain, 4);
  const xf = props.xFormat ?? ((v: number) => String(v));
  const yf = props.yFormat ?? ((v: number) => String(v));
  return (
    <g aria-hidden>
      {ys.map((t) => (
        <g key={`y${t}`}>
          <line
            className="grid-line"
            x1={frame.margin.left}
            x2={frame.margin.left + frame.innerWidth}
            y1={frame.y(t)}
            y2={frame.y(t)}
          />
          <text className="tick-label" x={frame.margin.left - 6} y={frame.y(t) + 3} textAnchor="end">
            {yf(t)}
          </text>
        </g>
      ))}
      <line
        className="axis-line"
        x1={frame.margin.left}
        x2={frame.margin.left + frame.innerWidth}
        y1={frame.y(frame.yDomain[0])}
        y2={frame.y(frame.yDomain[0])}
      />
      {xs.map((t) => (
        <text
          key={`x${t}`}
          className="tick-label"
          x={frame.x(t)}
          y={frame.margin.top + frame.innerHeight + 16}
          textAnchor="middle"
        >
          {xf(t)}
        </text>
      ))}
      {props.xLabel ? (
        <text
          x={frame.margin.left + frame.innerWidth / 2}
          y={frame.height - 2}
          textAnchor="middle"
          className="tick-label"
        >
          {props.xLabel}
        </text>
      ) : null}
      {props.yLabel ? (
        <text
          transform={`translate(10 ${frame.margin.top + frame.innerHeight / 2}) rotate(-90)`}
          textAnchor="middle"
          className="tick-label"
        >
          {props.yLabel}
        </text>
      ) : null}
    </g>
  );
}

/** Polyline path for a series of points in data space. */
export function linePath(frame: Frame, pts: { x: number; y: number }[]): string {
  return pts
    .map((p, i) => `${i === 0 ? "M" : "L"}${frame.x(p.x).toFixed(1)},${frame.y(p.y).toFixed(1)}`)
    .join("");
}

/** Horizontal bar with a 4px rounded data end and a square baseline end. */
export function HBar(props: {
  x0: number;
  x1: number;
  y: number;
  height: number;
  fill: string;
  opacity?: number;
}) {
  const w = Math.max(0.5, Math.abs(props.x1 - props.x0));
  const x = Math.min(props.x0, props.x1);
  const r = Math.min(4, w / 2, props.height / 2);
  const rightRounded = props.x1 >= props.x0;
  const path = rightRounded
    ? `M${x},${props.y} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${props.height - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-(w - r)} z`
    : `M${x + w},${props.y} v${props.height} h${-(w - r)} a${r},${r} 0 0 1 ${-r},${-r} v${-(props.height - 2 * r)} a${r},${r} 0 0 1 ${r},${-r} z`;
  return <path d={path} fill={props.fill} opacity={props.opacity ?? 1} />;
}

/** Marker dot with the 2px surface ring. */
export function Dot(props: {
  cx: number;
  cy: number;
  r?: number;
  fill: string;
  onHover?: (e: React.PointerEvent) => void;
  onLeave?: () => void;
  label?: string;
}) {
  return (
    <circle
      cx={props.cx}
      cy={props.cy}
      r={props.r ?? 4}
      fill={props.fill}
      stroke="var(--surface-1)"
      strokeWidth={2}
      onPointerMove={props.onHover}
      onPointerLeave={props.onLeave}
      aria-label={props.label}
    >
      {props.label ? <title>{props.label}</title> : null}
    </circle>
  );
}
