/**
 * Selection tuple <-> URL fragment. A permalink is a citation: restoring the
 * URL restores the same station and selected evidence.
 *
 * Encoding: #/<station>?key=value&… using URLSearchParams. Only the known
 * AppState fields are serialized; unknown query keys are dropped on the next
 * update() call.
 */

import { useCallback, useEffect, useState } from "react";

export const STATIONS = ["channel", "trace", "bench", "audit"] as const;
export type Station = (typeof STATIONS)[number];

export interface AppState {
  station: Station;
  /** Selected qualified panel row (all stations). */
  row: string | null;
  /** Critic lens: primary | independent. */
  critic: "primary" | "independent";
  /**
   * CHANNEL: deep-link-only panel anchor (e.g. waterfall | rewrite); the UI
   * reads it for scrolling but does not mint it. BENCH reuses the slot for
   * the wake-metric choice.
   */
  view: string | null;
  /** CHANNEL rewrite explorer: selected transform cell id. */
  cell: string | null;
  /** TRACE: selected document position / poetry token position. */
  position: number | null;
  /** TRACE: poetry case id (poetry lens active when set). */
  poetryCase: string | null;
  /** BENCH: selected intervention chip/variant. */
  variant: string | null;
  /** BENCH: selected dose (as written in variant specs, e.g. "1"). */
  dose: string | null;
  /** AUDIT: selected claim id for deep links. */
  claim: string | null;
}

export const DEFAULT_STATE: AppState = {
  station: "channel",
  row: null,
  critic: "primary",
  view: null,
  cell: null,
  position: null,
  poetryCase: null,
  variant: null,
  dose: null,
  claim: null,
};

export function parseHash(hash: string): AppState {
  const cleaned = hash.replace(/^#\/?/, "");
  const [stationPart, query = ""] = cleaned.split("?");
  const station = (STATIONS as readonly string[]).includes(stationPart)
    ? (stationPart as Station)
    : DEFAULT_STATE.station;
  const params = new URLSearchParams(query);
  const num = (key: string): number | null => {
    const raw = params.get(key);
    if (raw === null || raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  };
  const critic = params.get("critic");
  return {
    station,
    row: params.get("row"),
    critic: critic === "independent" ? "independent" : "primary",
    view: params.get("view"),
    cell: params.get("cell"),
    position: num("pos"),
    poetryCase: params.get("poem"),
    variant: params.get("variant"),
    dose: params.get("dose"),
    claim: params.get("claim"),
  };
}

export function serializeState(state: AppState): string {
  const params = new URLSearchParams();
  if (state.row) params.set("row", state.row);
  if (state.critic !== "primary") params.set("critic", state.critic);
  if (state.view) params.set("view", state.view);
  if (state.cell) params.set("cell", state.cell);
  if (state.position !== null) params.set("pos", String(state.position));
  if (state.poetryCase) params.set("poem", state.poetryCase);
  if (state.variant) params.set("variant", state.variant);
  if (state.dose) params.set("dose", state.dose);
  if (state.claim) params.set("claim", state.claim);
  const query = params.toString();
  return `#/${state.station}${query ? `?${query}` : ""}`;
}

export function useAppState(): [AppState, (patch: Partial<AppState>) => void] {
  const [state, setState] = useState<AppState>(() =>
    typeof window === "undefined" ? DEFAULT_STATE : parseHash(window.location.hash),
  );

  useEffect(() => {
    const onHashChange = () => setState(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const update = useCallback((patch: Partial<AppState>) => {
    setState((prev) => {
      const next = { ...prev, ...patch };
      const hash = serializeState(next);
      if (typeof window !== "undefined" && window.location.hash !== hash) {
        // pushState keeps back/forward working without a hashchange feedback loop.
        window.history.pushState(null, "", hash);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const onPop = () => setState(parseHash(window.location.hash));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return [state, update];
}
