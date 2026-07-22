/**
 * Parse health & control completeness — parse-state counts per explanation
 * kind and closed/usable fractions per control lane. Control lanes render
 * always; a 'passed' fraction is completeness, not scientific success.
 */

import { useMemo } from "react";
import type { AuditShard } from "../../data/types";
import { Badge, Panel, UnavailableBox } from "../../components/ui";
import { fmtPct } from "../../data/format";
import { humanize } from "./util";

type ParseFractions = Record<string, { closed_fraction: number; usable_fraction: number }>;

function controlOrder(keys: string[]): string[] {
  return keys.slice().sort((a, b) => {
    if (a === "real") return -1;
    if (b === "real") return 1;
    return a.localeCompare(b);
  });
}

function FractionTable(props: { title: string; data: ParseFractions }) {
  const keys = controlOrder(Object.keys(props.data));
  if (keys.length === 0) {
    return <UnavailableBox>No parse-health records for {props.title}.</UnavailableBox>;
  }
  return (
    <div>
      <h4 className="audit-subhead">{props.title}</h4>
      <div className="audit-table-scroll">
        <table className="data-table" tabIndex={0}>
          <thead>
            <tr>
              <th scope="col">condition</th>
              <th scope="col" className="num">
                closed
              </th>
              <th scope="col" className="num">
                usable
              </th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k}>
                <td className="mono">{k}</td>
                <td className="num">{fmtPct(props.data[k].closed_fraction, 1)}</td>
                <td className="num">{fmtPct(props.data[k].usable_fraction, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Plain-language labels for explanation kinds (raw key stays in the title). */
const KIND_LABEL: Record<string, string> = {
  alternate_telling: "alternate tellings",
  qualified_av: "AV descriptions (qualified rows)",
  teacher: "teacher references",
  trace_description: "trace descriptions (film set)",
};

export default function ParseHealthPanel(props: { ph: AuditShard["parse_health"] }) {
  const { ph } = props;
  const kinds = useMemo(() => Object.keys(ph.explanations_by_kind).sort(), [ph]);
  const parseStates = useMemo(
    () =>
      Array.from(
        new Set(kinds.flatMap((k) => Object.keys(ph.explanations_by_kind[k]))),
      ).sort(),
    [ph, kinds],
  );

  return (
    <Panel
      id="parse_health"
      span={6}
      title="Parse health & control completeness"
      sub="Whether learned descriptions parsed cleanly, per kind and per control lane. Completeness only — not evidence of semantic success."
    >
      {kinds.length === 0 ? (
        <UnavailableBox>No explanation parse counts in this bundle.</UnavailableBox>
      ) : (
        <div className="audit-table-scroll">
          <table className="data-table" tabIndex={0}>
            <thead>
              <tr>
                <th scope="col">explanation kind</th>
                {parseStates.map((s) => (
                  <th scope="col" className="num" key={s}>
                    {humanize(s)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {kinds.map((k) => (
                <tr key={k}>
                  <td title={`kind: ${k}`}>
                    {KIND_LABEL[k] ?? k.replace(/_/g, " ")}{" "}
                    <span className="mono audit-kind-key">{k}</span>
                  </td>
                  {parseStates.map((s) => (
                    <td className="num" key={s}>
                      {ph.explanations_by_kind[k][s] !== undefined
                        ? ph.explanations_by_kind[k][s].toLocaleString()
                        : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <dl className="kv-list audit-spaced">
        <dt>trace descriptions usable</dt>
        <dd>
          {ph.trace_descriptions_usable ? "yes" : "no"}{" "}
          {ph.trace_descriptions_usable ? null : <Badge status="caveat" label="not usable" />}
        </dd>
        <dt>poetry usable fraction</dt>
        <dd>{fmtPct(ph.poetry_usable_fraction, 1)}</dd>
      </dl>

      <FractionTable title="E1-AV parse (per control)" data={ph.e1_av_parse} />
      <FractionTable title="Null-text almanac parse (per control)" data={ph.almanac_parse_health} />
    </Panel>
  );
}
