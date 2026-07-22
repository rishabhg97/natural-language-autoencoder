import { useMemo } from "react";
import { segmentByTerms } from "./textUtils";

/**
 * Renders text with labeled highlights on target/alternate term hits.
 * Highlights pair an underline style and a title with the background wash so
 * the encoding is never color-alone.
 */
export default function HighlightTerms(props: {
  text: string;
  targetTerms: readonly string[];
  alternateTerms?: readonly string[];
  targetTitle?: string;
  alternateTitle?: string;
}) {
  const { text, targetTerms, alternateTerms } = props;
  const segments = useMemo(
    () => segmentByTerms(text, targetTerms, alternateTerms ?? []),
    [text, targetTerms, alternateTerms],
  );
  return (
    <>
      {segments.map((seg, i) =>
        seg.hit === null ? (
          <span key={i}>{seg.text}</span>
        ) : (
          <mark
            key={i}
            className={`trc-hit trc-hit-${seg.hit}`}
            title={
              seg.hit === "target"
                ? (props.targetTitle ?? "target-family term")
                : (props.alternateTitle ?? "alternate-family term")
            }
          >
            {seg.text}
          </mark>
        ),
      )}
    </>
  );
}
