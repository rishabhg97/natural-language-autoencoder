/**
 * App-level smoke + behavior + accessibility tests over the synthetic fixture
 * bundle. These are deliberately resilient to station internals: stations are
 * developed in parallel, so each test only asserts that
 *   (a) the station tab activates and the app shell stays alive,
 *   (b) no ErrorBox (role=alert) appears with the fixture bundle, and
 *   (c) axe finds no serious/critical violations.
 * A station rendering an UnavailableBox for fixture-scale data is acceptable;
 * a crash or an error box is not.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import App from "../src/app/App";
import { resetLoaderForTests } from "../src/data/loader";
import { installFixtureFetch } from "./mockFetch";

const STATION_TABS = [
  { station: "channel", tabName: /channel/i, readyText: /rewrite explorer/i },
  { station: "trace", tabName: /trace/i, readyText: /document token-linked nla reader/i },
  { station: "bench", tabName: /bench/i, readyText: /control rack/i },
  { station: "audit", tabName: /audit/i, readyText: /phases complete/i },
] as const;

const TEST_TIMEOUT = 30_000;

const settle = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function expectNoSeriousAxeViolations(container: HTMLElement): Promise<void> {
  const results = await axe.run(container, {
    // jsdom has no layout/canvas, so color-contrast cannot be computed here;
    // the Playwright e2e suite covers contrast in a real browser.
    rules: { "color-contrast": { enabled: false } },
  });
  const severe = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  const summary = severe.map((v) => `${v.id}: ${v.help} (${v.nodes.length} nodes)`);
  expect(summary).toEqual([]);
}

describe("App stations over the fixture bundle", () => {
  beforeEach(() => {
    resetLoaderForTests();
    installFixtureFetch();
    window.history.replaceState(null, "", "#/");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    resetLoaderForTests();
  });

  it(
    "renders the shell with fixture-driven manifest evidence",
    async () => {
      render(<App />);
      // Evidence rail text comes straight from the fixture manifest counts.
      await screen.findByText(/2 rows · 8 precomputed interventions · 4 behavior cells/);
      expect(screen.getAllByRole("tab")).toHaveLength(4);
      expect(screen.queryAllByRole("alert")).toHaveLength(0);
    },
    TEST_TIMEOUT,
  );

  it(
    "states the matched online-RL result in plain language",
    async () => {
      render(<App />);
      await screen.findByText(/Did online RL improve reconstruction/i);
      expect(screen.getByText(/Online RL lowered round-trip error by 25.0%/i)).toBeInTheDocument();
      expect(screen.getAllByText(/not a sealed test result/i).length).toBeGreaterThan(0);
    },
    TEST_TIMEOUT,
  );

  for (const { station, tabName, readyText } of STATION_TABS) {
    it(
      `${station} station activates without an error box and passes axe`,
      async () => {
        const { container } = render(<App />);
        // Wait for the manifest gate to pass (fixture-driven rail content).
        await screen.findByText(/4 behavior cells/);

        const tab = await screen.findByRole("tab", { name: tabName });
        await userEvent.click(tab);
        await waitFor(() => expect(tab).toHaveAttribute("aria-selected", "true"));

        // Wait for the deepest lazy shard used by the station, then flush any
        // immediately queued manifest/cache updates before axe inspects it.
        await screen.findByText(readyText, {}, { timeout: 8_000 });
        await act(async () => settle(0));
        await waitFor(
          () => {
            // No ErrorBox anywhere: the fixture bundle is complete, so any
            // alert means a shard failed to load or a contract was violated.
            expect(screen.queryAllByRole("alert")).toHaveLength(0);
            // The app did not crash: the shell plus station render real text.
            expect(document.body.textContent?.length ?? 0).toBeGreaterThan(80);
          },
          { timeout: 8_000 },
        );

        // Selection state round-trips into the permalink hash.
        expect(window.location.hash.startsWith(`#/${station}`)).toBe(true);

        await expectNoSeriousAxeViolations(container);
      },
      TEST_TIMEOUT,
    );
  }

  it(
    "restores the station from a permalink hash on first render",
    async () => {
      window.history.replaceState(null, "", "#/audit?claim=provenance");
      render(<App />);
      await screen.findByText(/4 behavior cells/);
      const tab = await screen.findByRole("tab", { name: /audit/i });
      await waitFor(() => expect(tab).toHaveAttribute("aria-selected", "true"));
      expect(window.location.hash).toContain("claim=provenance");
    },
    TEST_TIMEOUT,
  );

  it(
    "shows an explicit alert when the manifest fails its schema gate",
    async () => {
      vi.unstubAllGlobals();
      installFixtureFetch({
        overrides: { "manifest.json": { schema_version: "bogus.v0", files: [] } },
      });
      render(<App />);
      const alerts = await screen.findAllByRole("alert");
      expect(alerts.length).toBeGreaterThan(0);
      expect(alerts.map((a) => a.textContent).join(" ")).toMatch(/bundle version mismatch/);
    },
    TEST_TIMEOUT,
  );
});
