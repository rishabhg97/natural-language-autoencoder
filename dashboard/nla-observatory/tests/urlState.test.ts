import { describe, expect, it } from "vitest";
import {
  DEFAULT_STATE,
  parseHash,
  serializeState,
  STATIONS,
  type AppState,
} from "../src/app/urlState";

describe("parseHash / serializeState", () => {
  it("returns defaults for an empty hash", () => {
    expect(parseHash("")).toEqual(DEFAULT_STATE);
    expect(parseHash("#")).toEqual(DEFAULT_STATE);
    expect(parseHash("#/")).toEqual(DEFAULT_STATE);
  });

  it("serializes the default state to the bare channel station", () => {
    expect(serializeState(DEFAULT_STATE)).toBe("#/channel");
  });

  it("round-trips a state with every field populated", () => {
    const full: AppState = {
      station: "bench",
      row: "validation-1",
      critic: "independent",
      view: "waterfall",
      cell: "cell-rw-1",
      position: 12,
      poetryCase: "fixture-alpha",
      variant: "syntax",
      dose: "1",
      claim: "stored_snapshot_channel",
    };
    expect(parseHash(serializeState(full))).toEqual(full);
  });

  it("round-trips each station with default selections", () => {
    for (const station of STATIONS) {
      const state: AppState = { ...DEFAULT_STATE, station };
      expect(serializeState(state)).toBe(`#/${station}`);
      expect(parseHash(serializeState(state))).toEqual(state);
    }
  });

  it("round-trips a partially populated trace selection", () => {
    const state: AppState = {
      ...DEFAULT_STATE,
      station: "trace",
      row: "validation-2",
      position: 0, // zero must survive (it is not null)
      poetryCase: "fixture-beta",
    };
    expect(parseHash(serializeState(state))).toEqual(state);
  });

  it("falls back to the channel station for unknown stations", () => {
    const parsed = parseHash("#/warp-core?row=validation-1");
    expect(parsed.station).toBe("channel");
    // the rest of the query still parses
    expect(parsed.row).toBe("validation-1");
  });

  it("parses bad numeric positions as null", () => {
    expect(parseHash("#/trace?pos=abc").position).toBeNull();
    expect(parseHash("#/trace?pos=NaN").position).toBeNull();
    expect(parseHash("#/trace?pos=Infinity").position).toBeNull();
    expect(parseHash("#/trace?pos=").position).toBeNull();
  });

  it("parses valid numeric positions including negatives and zero", () => {
    expect(parseHash("#/trace?pos=7").position).toBe(7);
    expect(parseHash("#/trace?pos=0").position).toBe(0);
    expect(parseHash("#/trace?pos=-3").position).toBe(-3);
  });

  it("falls back to the primary critic for unknown critic values", () => {
    expect(parseHash("#/channel?critic=bogus").critic).toBe("primary");
    expect(parseHash("#/channel?critic=independent").critic).toBe("independent");
    expect(parseHash("#/channel").critic).toBe("primary");
  });

  it("omits default/null fields from the serialized query", () => {
    const state: AppState = { ...DEFAULT_STATE, station: "audit", claim: "provenance" };
    expect(serializeState(state)).toBe("#/audit?claim=provenance");
  });

  it("keeps missing query fields null after parsing", () => {
    const parsed = parseHash("#/bench?variant=syntax");
    expect(parsed.variant).toBe("syntax");
    expect(parsed.row).toBeNull();
    expect(parsed.cell).toBeNull();
    expect(parsed.dose).toBeNull();
    expect(parsed.claim).toBeNull();
    expect(parsed.position).toBeNull();
    expect(parsed.poetryCase).toBeNull();
  });

  it("round-trips URL-hostile characters in values", () => {
    const state: AppState = {
      ...DEFAULT_STATE,
      station: "bench",
      variant: "syntax:edit:a1",
      dose: "0.5",
    };
    expect(parseHash(serializeState(state))).toEqual(state);
  });
});
