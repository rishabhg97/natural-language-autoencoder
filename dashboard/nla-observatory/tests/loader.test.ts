import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import {
  EXPECTED_SCHEMA,
  loadManifest,
  loadShard,
  resetLoaderForTests,
  useShard,
} from "../src/data/loader";
import type { ChannelShard, DashboardManifest, RowsShard } from "../src/data/types";
import { fixtureJson, installFixtureFetch, type FixtureFetchMock } from "./mockFetch";

function callsFor(mock: FixtureFetchMock, path: string): number {
  return mock.mock.calls.filter(([input]) => String(input).includes(path)).length;
}

describe("loader", () => {
  beforeEach(() => {
    resetLoaderForTests();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    resetLoaderForTests();
  });

  it("loads the fixture manifest and passes the schema gate", async () => {
    installFixtureFetch();
    const manifest = await loadManifest();
    expect(manifest.schema_version).toBe(EXPECTED_SCHEMA);
    expect(manifest.files.map((f) => f.path)).toContain("channel.json");
    expect(manifest.source.counts.rows).toBe(2);
  });

  it("rejects a manifest with the wrong schema_version", async () => {
    const manifest = fixtureJson<DashboardManifest>("manifest.json");
    installFixtureFetch({
      overrides: {
        "manifest.json": { ...manifest, schema_version: "nla_observatory_dashboard.v999" },
      },
    });
    await expect(loadManifest()).rejects.toThrow(/bundle version mismatch/);
  });

  it("surfaces the manifest schema gate as an error state in useShard", async () => {
    const manifest = fixtureJson<DashboardManifest>("manifest.json");
    installFixtureFetch({
      overrides: { "manifest.json": { ...manifest, schema_version: "bogus.v0" } },
    });
    const { result } = renderHook(() => useShard<RowsShard>("rows.json"));
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("error"));
    if (result.current.status === "error") {
      expect(result.current.message).toMatch(/bundle version mismatch/);
    }
  });

  it("loads a shard and resolves fixture row ids", async () => {
    installFixtureFetch();
    const rows = await loadShard<RowsShard>("rows.json");
    expect(rows.kind).toBe("rows");
    expect(rows.rows.map((r) => r.row_id)).toEqual(["validation-1", "validation-2"]);
  });

  it("caches shards: fetch fires once per path across repeat and concurrent loads", async () => {
    const mock = installFixtureFetch();
    await Promise.all([
      loadShard<RowsShard>("rows.json"),
      loadShard<RowsShard>("rows.json"),
    ]);
    await loadShard<RowsShard>("rows.json");
    await loadShard<ChannelShard>("channel.json");
    await loadShard<ChannelShard>("channel.json");
    expect(callsFor(mock, "manifest.json")).toBe(1);
    expect(callsFor(mock, "rows.json")).toBe(1);
    expect(callsFor(mock, "channel.json")).toBe(1);
  });

  it("rejects shard paths that are not listed in the manifest", async () => {
    installFixtureFetch();
    await expect(loadShard("not-a-shard.json")).rejects.toThrow(
      /not listed in the dashboard manifest/,
    );
  });

  it("rejects malformed JSON bodies", async () => {
    installFixtureFetch({ rawBodies: { "channel.json": '{"schema_version": ' } });
    await expect(loadShard<ChannelShard>("channel.json")).rejects.toThrow(/malformed JSON/);
  });

  it("rejects HTTP failures with the status code", async () => {
    installFixtureFetch({ missing: ["trace.json"] });
    await expect(loadShard("trace.json")).rejects.toThrow(/HTTP 404/);
  });

  it("rejects shard bodies that fail the schema check", async () => {
    installFixtureFetch({
      overrides: { "rows.json": { schema_version: "wrong.v0", kind: "rows", rows: [] } },
    });
    await expect(loadShard<RowsShard>("rows.json")).rejects.toThrow(/failed its schema check/);
  });

  it("does not cache failures: a shard can recover after a transient error", async () => {
    installFixtureFetch({ missing: ["trace.json"] });
    await expect(loadShard("trace.json")).rejects.toThrow(/HTTP 404/);
    // allow the cache-eviction .catch handler to run
    await new Promise((r) => setTimeout(r, 0));
    vi.unstubAllGlobals();
    installFixtureFetch();
    const trace = await loadShard<{ schema_version: string; kind: string }>("trace.json");
    expect(trace.kind).toBe("trace");
  });

  it("useShard reaches ready with fixture data on the happy path", async () => {
    installFixtureFetch();
    const { result } = renderHook(() => useShard<RowsShard>("rows.json"));
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    if (result.current.status === "ready") {
      expect(result.current.data.rows).toHaveLength(2);
      expect(result.current.data.rows[0].row_id).toBe("validation-1");
    }
  });

  it("useShard reports an error state for missing shards", async () => {
    installFixtureFetch({ missing: ["audit.json"] });
    const { result } = renderHook(() => useShard("audit.json"));
    await waitFor(() => expect(result.current.status).toBe("error"));
    if (result.current.status === "error") {
      expect(result.current.message).toMatch(/HTTP 404/);
    }
  });

  it("useShard stays loading when given a null path", async () => {
    installFixtureFetch();
    const { result } = renderHook(() => useShard(null));
    await new Promise((r) => setTimeout(r, 10));
    expect(result.current.status).toBe("loading");
  });
});
