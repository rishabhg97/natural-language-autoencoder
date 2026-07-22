/**
 * Session ledger: an append-only local record of every precomputed experiment
 * the user opens on BENCH. Stores selection tuples + cell ids ONLY — never
 * evidence values. Data lives in this browser's localStorage; the only way it
 * leaves is the user copying it.
 */

import { useCallback, useState } from "react";

export interface LedgerEntry {
  ts: string;
  row: string;
  chip: string;
  dose: string | null;
  cells: Record<string, string>;
}

export const LEDGER_KEY = "nla-bench-ledger";

function readStore(): LedgerEntry[] {
  try {
    const raw = window.localStorage.getItem(LEDGER_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as LedgerEntry[]) : [];
  } catch {
    return [];
  }
}

function writeStore(entries: LedgerEntry[]): void {
  try {
    window.localStorage.setItem(LEDGER_KEY, JSON.stringify(entries));
  } catch {
    // Storage may be unavailable (private mode); the in-memory copy remains.
  }
}

export interface Ledger {
  entries: LedgerEntry[];
  record: (entry: Omit<LedgerEntry, "ts">) => void;
  clear: () => void;
}

export function useLedger(): Ledger {
  const [entries, setEntries] = useState<LedgerEntry[]>(readStore);

  const record = useCallback((entry: Omit<LedgerEntry, "ts">) => {
    setEntries((prev) => {
      const last = prev[prev.length - 1];
      if (
        last &&
        last.row === entry.row &&
        last.chip === entry.chip &&
        last.dose === entry.dose &&
        JSON.stringify(last.cells) === JSON.stringify(entry.cells)
      ) {
        return prev; // same experiment still open — append only on change
      }
      const next = [...prev, { ...entry, ts: new Date().toISOString() }];
      writeStore(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setEntries(() => {
      writeStore([]);
      return [];
    });
  }, []);

  return { entries, record, clear };
}
