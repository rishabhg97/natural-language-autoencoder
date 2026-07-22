import { lazy, Suspense, useEffect, useRef } from "react";
import { Activity, FlaskConical, Radio, ShieldCheck } from "lucide-react";
import { useManifest } from "../data/loader";
import { useAppState, STATIONS, type Station } from "./urlState";
import { Badge, ErrorBox, HashChip, LoadingBox, Segmented } from "../components/ui";
import { NlaPrimer } from "../components/NlaPrimer";
import { TooltipProvider } from "../components/charts";
import { useTheme } from "./theme";

const ChannelStation = lazy(() => import("../stations/channel/ChannelStation"));
const TraceStation = lazy(() => import("../stations/trace/TraceStation"));
const BenchStation = lazy(() => import("../stations/bench/BenchStation"));
const AuditStation = lazy(() => import("../stations/audit/AuditStation"));

const STATION_META: Record<
  Station,
  { label: string; hint: string; icon: typeof Radio }
> = {
  channel: { label: "CHANNEL", hint: "what survives language", icon: Radio },
  trace: { label: "TRACE", hint: "what changes across tokens", icon: Activity },
  bench: { label: "BENCH", hint: "precomputed counterfactuals", icon: FlaskConical },
  audit: { label: "AUDIT", hint: "what is safe to claim", icon: ShieldCheck },
};

/** Publish the sticky header height so anchored panels can scroll clear of it. */
function useHeaderHeightVar() {
  useEffect(() => {
    const header = document.querySelector<HTMLElement>(".shell-header");
    if (!header) return;
    const apply = () =>
      document.documentElement.style.setProperty("--header-h", `${header.offsetHeight}px`);
    apply();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(apply) : null;
    ro?.observe(header);
    window.addEventListener("resize", apply);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", apply);
      document.documentElement.style.removeProperty("--header-h");
    };
  }, []);
}

export default function App() {
  const [state, update] = useAppState();
  const manifest = useManifest();
  const [theme, setTheme] = useTheme();
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  useHeaderHeightVar();

  // Roving arrow-key navigation for the station tablist.
  const onTabKeyDown = (event: React.KeyboardEvent, index: number) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const delta = event.key === "ArrowRight" ? 1 : -1;
    const next = (index + delta + STATIONS.length) % STATIONS.length;
    tabRefs.current[next]?.focus();
    update({ station: STATIONS[next] });
  };

  return (
    <TooltipProvider>
      <a className="skip-link" href="#station-main">
        Skip to station content
      </a>
      <div className="shell">
        <header className="shell-header">
          <div className="shell-brand">
            <span className="shell-brand-mark" aria-hidden>NLA</span>
            <h1 className="shell-title">
              Activation Observatory <span className="sub">Nano30B · R33</span>
            </h1>
          </div>
          <nav className="station-tabs" role="tablist" aria-label="Stations">
            {STATIONS.map((s, i) => {
              const Icon = STATION_META[s].icon;
              return (
                <button
                  key={s}
                  ref={(el) => {
                    tabRefs.current[i] = el;
                  }}
                  role="tab"
                  aria-selected={state.station === s}
                  tabIndex={state.station === s ? 0 : -1}
                  className="station-tab"
                  title={STATION_META[s].hint}
                  onClick={() => update({ station: s })}
                  onKeyDown={(e) => onTabKeyDown(e, i)}
                >
                  <Icon size={13} aria-hidden />
                  {STATION_META[s].label}
                </button>
              );
            })}
          </nav>
          <div className="shell-meta">
            <Segmented
              options={["light", "system", "dark"] as const}
              value={theme}
              onChange={setTheme}
              label="Color theme"
            />
          </div>
        </header>

        <div className="evidence-rail" aria-label="Evidence build rail">
          {manifest.status === "ready" ? (
            <>
              <Badge
                status="qualified"
                label={`${manifest.data.source.population} · ${manifest.data.source.split}`}
                title="Stored-snapshot channel claim on the 50-row qualified validation panel"
              />
              <span>
                {manifest.data.source.counts.rows} rows ·{" "}
                {manifest.data.source.counts.interventions.toLocaleString()} precomputed
                interventions · {manifest.data.source.counts.behavior} behavior cells
              </span>
              <span className="evidence-rail-snapshot">
                Static snapshot · no model runs at view time
              </span>
              <details className="evidence-rail-provenance">
                <summary>provenance</summary>
                <div>
                  <HashChip
                    hash={manifest.data.source.bundle_id}
                    label="bundle"
                    title={`Observatory bundle id ${manifest.data.source.bundle_id} — click to copy`}
                  />
                  <HashChip
                    hash={manifest.data.poetry.config_sha256}
                    label="poetry cfg"
                    title={`Poetry planning config hash ${manifest.data.poetry.config_sha256} — click to copy`}
                  />
                </div>
              </details>
            </>
          ) : manifest.status === "error" ? (
            <span role="alert" style={{ color: "var(--status-negative)" }}>
              Bundle manifest failed to load: {manifest.message}
            </span>
          ) : (
            <span>Verifying dashboard bundle manifest…</span>
          )}
        </div>

        <main className="station-main" id="station-main">
          <NlaPrimer station={state.station} onOpenStation={(s) => update({ station: s })} />
          {manifest.status === "error" ? (
            <ErrorBox
              message={`${manifest.message}\n\nGenerate the local data with:\n  python3 scripts/build_static_data.py\n  python3 scripts/verify_static_data.py`}
            />
          ) : (
            <Suspense fallback={<LoadingBox what={`${STATION_META[state.station].label} station`} />}>
              {state.station === "channel" && <ChannelStation state={state} update={update} />}
              {state.station === "trace" && <TraceStation state={state} update={update} />}
              {state.station === "bench" && <BenchStation state={state} update={update} />}
              {state.station === "audit" && <AuditStation state={state} update={update} />}
            </Suspense>
          )}
        </main>
      </div>
    </TooltipProvider>
  );
}
