/**
 * Fixed BENCH banner. Always visible (sticky under the app header), never
 * dismissible: users are choosing among precomputed experiments, not editing
 * a live model.
 */

import { useEffect } from "react";
import type { BenchIndexShard } from "../../data/types";
import type { AppState } from "../../app/urlState";
import { Badge, HashChip } from "../../components/ui";
import { LinkButton } from "./benchUi";

/**
 * Keep the banner pinned just below the (sticky, wrap-height) app header, and
 * publish the banner's own height so anchored panels can scroll clear of the
 * double sticky stack.
 */
function useHeaderOffset() {
  useEffect(() => {
    const header = document.querySelector<HTMLElement>(".shell-header");
    if (!header) return;
    const apply = () => {
      document.documentElement.style.setProperty(
        "--bench-header-h",
        `${header.offsetHeight}px`,
      );
      const banner = document.querySelector<HTMLElement>(".bench-banner");
      document.documentElement.style.setProperty(
        "--bench-banner-h",
        `${banner?.offsetHeight ?? 0}px`,
      );
    };
    apply();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(apply) : null;
    ro?.observe(header);
    const banner = document.querySelector<HTMLElement>(".bench-banner");
    if (banner) ro?.observe(banner);
    window.addEventListener("resize", apply);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", apply);
      document.documentElement.style.removeProperty("--bench-header-h");
      document.documentElement.style.removeProperty("--bench-banner-h");
    };
  }, []);
}

export default function BannerBar(props: {
  banner: BenchIndexShard["banner"];
  update: (patch: Partial<AppState>) => void;
}) {
  useHeaderOffset();
  const { banner, update } = props;
  return (
    <div className="bench-banner" role="note" aria-label="bench claim banner">
      <p className="bench-banner-statement">
        Precomputed evidence only <span>· viewer, not a live model editor</span>
      </p>
      <div className="bench-banner-meta">
        <HashChip
          hash={banner.grid_spec_sha256}
          label="grid spec"
          title={`grid spec sha256 ${banner.grid_spec_sha256} — click to copy`}
        />
        <Badge
          status="qualified"
          label="stored-snapshot"
          title={`claim scope: ${banner.claim_scope}`}
        />
        <Badge
          status="exploratory"
          label="functional: validation-only"
          title={`functional claim status: ${banner.functional_claim_status}`}
        />
        <LinkButton
          onClick={() => update({ station: "audit", claim: "functional_interventions" })}
          title="Open the functional-interventions claim on AUDIT"
        >
          AUDIT: functional claim
        </LinkButton>
      </div>
    </div>
  );
}
