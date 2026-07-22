/** Word-level LCS diff for the continuation pair (short sequences only). */

export type DiffMark = "same" | "changed";

export interface DiffResult {
  /** one mark per word of `a`: "changed" = only in baseline (removed). */
  a: DiffMark[];
  /** one mark per word of `b`: "changed" = only in patched (added). */
  b: DiffMark[];
}

export function splitWords(text: string): string[] {
  return text.split(/\s+/).filter((w) => w.length > 0);
}

export function wordDiff(a: string[], b: string[]): DiffResult {
  const n = a.length;
  const m = b.length;
  // LCS length table
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const marksA: DiffMark[] = new Array<DiffMark>(n).fill("changed");
  const marksB: DiffMark[] = new Array<DiffMark>(m).fill("changed");
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      marksA[i] = "same";
      marksB[j] = "same";
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }
  return { a: marksA, b: marksB };
}
