/**
 * Poetry pipeline status — phase completeness only. A pipeline 'passed' flag
 * means the phases ran to completion; it is not evidence that a scientific
 * planning hypothesis passed (the pipeline note says exactly this, verbatim).
 */

import type { AuditShard } from "../../data/types";
import { useManifest } from "../../data/loader";
import { Badge, ErrorBox, HashChip, LoadingBox, Panel } from "../../components/ui";
import { humanize } from "./util";

export default function PoetryStatusPanel(props: { status: AuditShard["poetry_status"] }) {
  const manifest = useManifest();
  const s = props.status;
  return (
    <Panel
      span={6}
      title="Poetry pipeline status"
      sub="Completeness of the poetry planning pipeline. See TRACE for the evidence itself and 'Negative & weak results' for its outcomes."
      badges={
        <>
          <Badge status="exploratory" label={humanize(s.claim_scope)} />
          <HashChip hash={s.config_sha256} label="config" />
        </>
      }
    >
      {manifest.status === "loading" ? (
        <LoadingBox what="bundle manifest (poetry phases)" />
      ) : manifest.status === "error" ? (
        <ErrorBox message={manifest.message} />
      ) : (
        <>
          <p className="audit-poetry-phases">
            {s.pipeline_passed
              ? `${manifest.data.poetry.phases_passed.length}/${manifest.data.poetry.phases_passed.length} phases complete`
              : `${manifest.data.poetry.phases_passed.length} phases recorded as passed; pipeline incomplete`}
          </p>
          <div className="audit-chip-row">
            {manifest.data.poetry.phases_passed.map((p) => (
              <span className="audit-chip" key={p}>
                {p}
              </span>
            ))}
          </div>
        </>
      )}
      <p className="audit-note">
        <Badge status="caveat" label="scope" /> {s.pipeline_note}
      </p>
    </Panel>
  );
}
