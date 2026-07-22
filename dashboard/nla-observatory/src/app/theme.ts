import { useCallback, useEffect, useState } from "react";

export type ThemeChoice = "light" | "system" | "dark";
const KEY = "nla-observatory-theme";

function apply(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
}

export function useTheme(): [ThemeChoice, (t: ThemeChoice) => void] {
  const [choice, setChoice] = useState<ThemeChoice>(() => {
    const stored = typeof window === "undefined" ? null : window.localStorage.getItem(KEY);
    return stored === "light" || stored === "dark" ? stored : "system";
  });

  useEffect(() => {
    apply(choice);
  }, [choice]);

  const set = useCallback((t: ThemeChoice) => {
    setChoice(t);
    if (t === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, t);
  }, []);

  return [choice, set];
}
