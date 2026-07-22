/**
 * Provenance browser — bundle identity, tokenizer, report/code bindings,
 * runtime, privacy triage card, and the full file inventories. Every hash is
 * click-to-copy; every report is directly openable as static JSON.
 */

import { useMemo } from "react";
import type { AuditShard, ManifestFile } from "../../data/types";
import { useManifest } from "../../data/loader";
import { Badge, ErrorBox, HashChip, LoadingBox, Panel, UnavailableBox } from "../../components/ui";
import { KvList, asString, humanize } from "./util";

function FilesTable(props: { files: ManifestFile[] }) {
  return (
    <div className="audit-table-scroll">
      <table className="data-table" tabIndex={0}>
        <thead>
          <tr>
            <th scope="col">path</th>
            <th scope="col" className="num">
              bytes
            </th>
            <th scope="col">sha256</th>
          </tr>
        </thead>
        <tbody>
          {props.files.map((f) => (
            <tr key={f.path}>
              <td className="mono audit-wrap">{f.path}</td>
              <td className="num">{f.bytes.toLocaleString()}</td>
              <td>
                <HashChip hash={f.sha256} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ProvenancePanel(props: { prov: AuditShard["provenance"] }) {
  const { prov } = props;
  const manifest = useManifest();

  const reportRows = useMemo(
    () =>
      Object.entries(prov.report_bindings)
        .map(([name, b]) => ({
          name,
          sourcePath: b.source_path,
          sha256: b.sha256,
          href: `data/reports/${name}.json`,
        }))
        .sort((a, b) => a.name.localeCompare(b.name)),
    [prov.report_bindings],
  );

  const poetryReports = useMemo(() => {
    if (manifest.status !== "ready") return [];
    return manifest.data.files
      .filter((f) => /^reports\/poetry_.*\.json$/.test(f.path))
      .map((f) => ({
        name: f.path.replace(/^reports\//, "").replace(/\.json$/, ""),
        sourcePath: f.path,
        sha256: f.sha256,
        href: `data/${f.path}`,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [manifest]);

  const codeRows = useMemo(
    () => Object.entries(prov.code_bindings).sort(([a], [b]) => a.localeCompare(b)),
    [prov.code_bindings],
  );

  const pc = prov.privacy_card;
  const privacyBoundary = asString(pc.claim_boundary);
  const privacySha = asString(pc.sha256);

  return (
    <Panel
      id="provenance"
      span={12}
      title="Provenance browser"
      sub="Where every number on this dashboard comes from. All hashes copy on click; all reports open as the exact static JSON the builder shipped."
    >
      <div className="audit-prov-grid">
        <div className="audit-prov-card">
          <h4 className="audit-subhead">Bundle identity</h4>
          <div className="audit-chip-row">
            <HashChip hash={prov.bundle_id} label="bundle" />
            <HashChip hash={prov.source_config_sha256} label="source cfg" />
            <HashChip hash={prov.bundle_config_sha256} label="bundle cfg" />
          </div>
          <dl className="kv-list audit-spaced">
            <dt>population</dt>
            <dd>{prov.population}</dd>
            <dt>split</dt>
            <dd>{prov.split}</dd>
          </dl>
          <div className="audit-chip-row audit-spaced" aria-label="Bundle counts">
            {Object.entries(prov.counts)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([k, v]) => (
                <span className="audit-chip" key={k}>
                  {humanize(k)} {v.toLocaleString()}
                </span>
              ))}
          </div>
          <h4 className="audit-subhead">Excluded files</h4>
          {prov.excluded_files.length === 0 ? (
            <UnavailableBox>No files were excluded from the web bundle.</UnavailableBox>
          ) : (
            <>
              <ul className="audit-limitations">
                {prov.excluded_files.map((f) => (
                  <li className="mono audit-wrap" key={f}>
                    {f}
                  </li>
                ))}
              </ul>
              <p className="audit-fineprint">
                Heavy activation vectors excluded from the web bundle by design.
              </p>
            </>
          )}
        </div>

        <div className="audit-prov-card">
          <h4 className="audit-subhead">Tokenizer</h4>
          <p className="audit-path">{prov.tokenizer.path}</p>
          <div className="audit-chip-row">
            <HashChip hash={prov.tokenizer.sha256} label="tokenizer" />
          </div>
          <dl className="kv-list audit-spaced">
            <dt>vocab size</dt>
            <dd>{prov.tokenizer.vocab_size.toLocaleString()}</dd>
            <dt>spot check</dt>
            <dd>
              {prov.tokenizer.spot_check.tokens.toLocaleString()} tokens /{" "}
              {prov.tokenizer.spot_check.mismatches.toLocaleString()} mismatches
            </dd>
          </dl>
          <h4 className="audit-subhead">Runtime</h4>
          <KvList record={prov.runtime} />
        </div>

        <div className="audit-prov-card">
          <h4 className="audit-subhead">
            Privacy & memorization triage{" "}
            {pc.human_review_required === true ? (
              <Badge status="caveat" label="human review required" />
            ) : null}
            {privacySha ? <HashChip hash={privacySha} label="report" /> : null}
          </h4>
          <dl className="kv-list">
            {typeof pc.automatic_gate_passed === "boolean" ? (
              <>
                <dt>automatic gate</dt>
                <dd>{pc.automatic_gate_passed ? "passed (completeness triage)" : "failed"}</dd>
              </>
            ) : null}
            {typeof pc.human_review_required === "boolean" ? (
              <>
                <dt>human review required</dt>
                <dd>{pc.human_review_required ? "yes" : "no"}</dd>
              </>
            ) : null}
          </dl>
          {privacyBoundary ? <p className="audit-boundary">{privacyBoundary}</p> : null}
          <KvList
            record={pc}
            skip={["automatic_gate_passed", "human_review_required", "claim_boundary", "sha256"]}
          />
          <h4 className="audit-subhead">Source caches</h4>
          <div className="audit-chip-row">
            {Object.entries(prov.source_provenance)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([k, v]) => (
                <HashChip hash={v} label={humanize(k).replace(/ sha256$/, "")} key={k} />
              ))}
          </div>
        </div>
      </div>

      <details className="audit-details">
        <summary>Report bindings ({reportRows.length + poetryReports.length})</summary>
        <div className="audit-table-scroll">
          <table className="data-table" tabIndex={0}>
          <thead>
            <tr>
              <th scope="col">report</th>
              <th scope="col">source path</th>
              <th scope="col">sha256</th>
              <th scope="col">open</th>
            </tr>
          </thead>
          <tbody>
            {reportRows.map((r) => (
              <tr key={r.name}>
                <td className="mono">{r.name}</td>
                <td className="mono audit-wrap">{r.sourcePath}</td>
                <td>
                  <HashChip hash={r.sha256} />
                </td>
                <td>
                  <a href={r.href} target="_blank" rel="noreferrer">
                    open
                  </a>
                </td>
              </tr>
            ))}
            {poetryReports.map((r) => (
              <tr key={r.name}>
                <td className="mono">{r.name}</td>
                <td className="mono audit-wrap">{r.sourcePath} (dashboard copy)</td>
                <td>
                  <HashChip hash={r.sha256} />
                </td>
                <td>
                  <a href={r.href} target="_blank" rel="noreferrer">
                    open
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
          </table>
        </div>
        {manifest.status === "loading" ? (
          <LoadingBox what="bundle manifest (poetry report hashes)" />
        ) : manifest.status === "error" ? (
          <ErrorBox message={manifest.message} />
        ) : null}
      </details>

      <details className="audit-details">
        <summary>Code bindings ({codeRows.length})</summary>
        <div className="audit-table-scroll">
          <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th scope="col">path</th>
                <th scope="col">sha256</th>
              </tr>
            </thead>
            <tbody>
              {codeRows.map(([path, hash]) => (
                <tr key={path}>
                  <td className="mono audit-wrap">{path}</td>
                  <td>
                    <HashChip hash={hash} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <details className="audit-details">
        <summary>
          Dashboard shard files
          {manifest.status === "ready" ? ` (${manifest.data.files.length})` : ""}
        </summary>
        {manifest.status === "loading" ? (
          <LoadingBox what="bundle manifest (shard files)" />
        ) : manifest.status === "error" ? (
          <ErrorBox message={manifest.message} />
        ) : (
          <FilesTable files={manifest.data.files} />
        )}
      </details>

      <details className="audit-details">
        <summary>Source bundle files ({prov.files.length})</summary>
        <FilesTable files={prov.files} />
      </details>
    </Panel>
  );
}
