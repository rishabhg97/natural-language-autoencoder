/**
 * Vitest setup (wired via vite.config.ts `test.setupFiles`).
 * Registers jest-dom matchers and fills small jsdom gaps that the app shell
 * touches at render time.
 */

import "@testing-library/jest-dom/vitest";

// jsdom has no matchMedia; theme/system-preference code paths may query it.
if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  window.matchMedia = (query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList;
}

// This jsdom build exposes a non-functional localStorage; the theme hook
// reads/writes it, so install a tiny in-memory implementation when needed.
if (
  typeof window !== "undefined" &&
  (!window.localStorage || typeof window.localStorage.getItem !== "function")
) {
  const store = new Map<string, string>();
  const memoryStorage: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key: string) => (store.has(key) ? (store.get(key) as string) : null),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    removeItem: (key: string) => {
      store.delete(key);
    },
    setItem: (key: string, value: string) => {
      store.set(key, String(value));
    },
  };
  Object.defineProperty(window, "localStorage", { configurable: true, value: memoryStorage });
}

// jsdom has no ResizeObserver; charts or layout hooks may construct one.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}

// jsdom does not implement scrollIntoView; deep-link anchors call it.
if (
  typeof Element !== "undefined" &&
  typeof Element.prototype.scrollIntoView !== "function"
) {
  Element.prototype.scrollIntoView = () => {};
}

// HashChip copies hashes; jsdom has no clipboard implementation.
if (typeof navigator !== "undefined" && !navigator.clipboard) {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: async () => {} },
  });
}
