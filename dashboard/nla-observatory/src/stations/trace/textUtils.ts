/**
 * Client-side text derivations for the TRACE station.
 * Similarity/diff operate on simple whitespace-split, lowercased word sets —
 * deliberately naive so the persistence readout stays transparent and auditable.
 */

export function wordSet(text: string): Set<string> {
  const out = new Set<string>();
  for (const w of text.toLowerCase().split(/\s+/)) {
    if (w) out.add(w);
  }
  return out;
}

/** Word-level Jaccard similarity between two learned descriptions. */
export function jaccardSimilarity(a: string, b: string): number {
  const sa = wordSet(a);
  const sb = wordSet(b);
  if (sa.size === 0 && sb.size === 0) return 1;
  let inter = 0;
  for (const w of sa) if (sb.has(w)) inter += 1;
  const union = sa.size + sb.size - inter;
  return union === 0 ? 1 : inter / union;
}

export interface WordDiff {
  added: string[];
  removed: string[];
}

/** Word-set difference between two adjacent learned descriptions. */
export function wordDiff(from: string, to: string): WordDiff {
  const sa = wordSet(from);
  const sb = wordSet(to);
  const added: string[] = [];
  const removed: string[] = [];
  for (const w of sb) if (!sa.has(w)) added.push(w);
  for (const w of sa) if (!sb.has(w)) removed.push(w);
  return { added, removed };
}

export interface TermSegment {
  text: string;
  hit: "target" | "alternate" | null;
}

function escapeRegExp(term: string): string {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Split text into segments, marking case-insensitive whole-word occurrences of
 * target / alternate terms. Target terms win when a term appears in both lists.
 */
export function segmentByTerms(
  text: string,
  targetTerms: readonly string[],
  alternateTerms: readonly string[],
): TermSegment[] {
  const terms = [...targetTerms, ...alternateTerms].filter((t) => t.trim().length > 0);
  if (!text || terms.length === 0) return [{ text, hit: null }];
  const targets = new Set(targetTerms.map((t) => t.toLowerCase()));
  const re = new RegExp(`\\b(?:${terms.map(escapeRegExp).join("|")})\\b`, "gi");
  const segments: TermSegment[] = [];
  let cursor = 0;
  for (const match of text.matchAll(re)) {
    const start = match.index ?? 0;
    if (start > cursor) segments.push({ text: text.slice(cursor, start), hit: null });
    segments.push({
      text: match[0],
      hit: targets.has(match[0].toLowerCase()) ? "target" : "alternate",
    });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), hit: null });
  return segments;
}
