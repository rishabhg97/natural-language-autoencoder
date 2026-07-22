/**
 * Fixture-backed fetch mock for unit tests.
 *
 * The app loader fetches `data/<shard-path>`; this helper serves the synthetic
 * JSON committed under tests/fixtures/ for those paths via
 * `vi.stubGlobal("fetch", ...)`. Tests can override individual shard bodies
 * (parsed or raw text) or force 404s to exercise the loader's failure modes.
 *
 * Fixtures are imported statically (resolveJsonModule) so this file needs no
 * Node built-ins; regenerate the JSON with tests/fixtures/make_fixture.py.
 */

import { vi } from "vitest";

import auditFixture from "./fixtures/audit.json";
import benchIndexFixture from "./fixtures/bench/index.json";
import benchRowValidation1Fixture from "./fixtures/bench/row-validation-1.json";
import channelFixture from "./fixtures/channel.json";
import manifestFixture from "./fixtures/manifest.json";
import poetryFixture from "./fixtures/poetry.json";
import rewritesFixture from "./fixtures/rewrites.json";
import rowsFixture from "./fixtures/rows.json";
import traceFixture from "./fixtures/trace.json";

const FIXTURES: Record<string, unknown> = {
  "audit.json": auditFixture,
  "bench/index.json": benchIndexFixture,
  "bench/row-validation-1.json": benchRowValidation1Fixture,
  "channel.json": channelFixture,
  "manifest.json": manifestFixture,
  "poetry.json": poetryFixture,
  "rewrites.json": rewritesFixture,
  "rows.json": rowsFixture,
  "trace.json": traceFixture,
};

export interface FixtureFetchOptions {
  /** Replace the JSON body served for a shard path, e.g. { "manifest.json": {...} }. */
  overrides?: Record<string, unknown>;
  /** Serve raw text for a shard path (lets tests exercise malformed JSON). */
  rawBodies?: Record<string, string>;
  /** Shard paths that respond 404. */
  missing?: string[];
}

/** Map any fetched URL (relative or absolute, with query) to a shard path. */
export function shardPathFromUrl(input: RequestInfo | URL): string {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  const noQuery = url.split(/[?#]/)[0];
  const marker = noQuery.indexOf("data/");
  return marker >= 0 ? noQuery.slice(marker + "data/".length) : noQuery.replace(/^\.?\//, "");
}

/** Deep-cloned fixture shard (safe for tests to mutate before overriding). */
export function fixtureJson<T>(shardPath: string): T {
  const fixture = FIXTURES[shardPath];
  if (fixture === undefined) {
    throw new Error(`no fixture registered for shard path: ${shardPath}`);
  }
  return structuredClone(fixture) as T;
}

function makeResponse(bodyText: string, status: number): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => JSON.parse(bodyText) as unknown,
    text: async () => bodyText,
  } as unknown as Response;
}

export type FixtureFetchMock = ReturnType<
  typeof vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>
>;

/**
 * Install the fixture-serving fetch stub. Returns the mock so tests can
 * inspect call counts/URLs. Pair with `vi.unstubAllGlobals()` in afterEach.
 */
export function installFixtureFetch(options: FixtureFetchOptions = {}): FixtureFetchMock {
  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, _init?: RequestInit): Promise<Response> => {
      const shardPath = shardPathFromUrl(input);
      if (options.missing?.includes(shardPath)) {
        return makeResponse(JSON.stringify({ error: "not found" }), 404);
      }
      if (options.rawBodies && shardPath in options.rawBodies) {
        return makeResponse(options.rawBodies[shardPath], 200);
      }
      if (options.overrides && shardPath in options.overrides) {
        return makeResponse(JSON.stringify(options.overrides[shardPath]), 200);
      }
      const fixture = FIXTURES[shardPath];
      if (fixture === undefined) {
        return makeResponse(JSON.stringify({ error: `no fixture for ${shardPath}` }), 404);
      }
      return makeResponse(JSON.stringify(fixture), 200);
    },
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
