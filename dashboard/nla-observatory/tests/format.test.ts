import { describe, expect, it } from "vitest";
import {
  extent,
  fmt,
  fmtFixed,
  fmtPct,
  fmtSigned,
  linearScale,
  ticks,
} from "../src/data/format";

describe("fmt", () => {
  it("renders non-finite values as an em dash", () => {
    expect(fmt(Number.NaN)).toBe("—");
    expect(fmt(Number.POSITIVE_INFINITY)).toBe("—");
    expect(fmt(Number.NEGATIVE_INFINITY)).toBe("—");
  });

  it("renders zero without trailing decimals", () => {
    expect(fmt(0)).toBe("0");
  });

  it("strips an all-zero decimal tail from round numbers", () => {
    expect(fmt(2)).toBe("2");
    expect(fmt(-3, 2)).toBe("-3");
  });

  it("keeps fixed decimals when any decimal digit is significant", () => {
    expect(fmt(1.234567, 3)).toBe("1.235");
    expect(fmt(1.5)).toBe("1.500");
    expect(fmt(-0.25)).toBe("-0.250");
  });

  it("switches to exponential notation below the digit resolution", () => {
    expect(fmt(0.0004, 3)).toBe("4.0e-4");
    expect(fmt(-0.0004, 3)).toBe("-4.0e-4");
  });
});

describe("fmtFixed", () => {
  it("renders non-finite values as an em dash", () => {
    expect(fmtFixed(Number.NaN)).toBe("—");
  });

  it("keeps a fixed number of decimals", () => {
    expect(fmtFixed(0, 2)).toBe("0.00");
    expect(fmtFixed(1.005, 1)).toBe("1.0");
  });
});

describe("fmtPct", () => {
  it("renders non-finite values as an em dash", () => {
    expect(fmtPct(Number.NaN)).toBe("—");
    expect(fmtPct(Number.POSITIVE_INFINITY)).toBe("—");
  });

  it("formats fractions as percentages", () => {
    expect(fmtPct(0.5)).toBe("50.0%");
    expect(fmtPct(1)).toBe("100.0%");
    expect(fmtPct(0)).toBe("0.0%");
    expect(fmtPct(-0.25, 0)).toBe("-25%");
  });
});

describe("fmtSigned", () => {
  it("renders non-finite values as an em dash", () => {
    expect(fmtSigned(Number.NaN)).toBe("—");
  });

  it("prefixes positives with plus and keeps minus for negatives", () => {
    expect(fmtSigned(0.1234)).toBe("+0.123");
    expect(fmtSigned(-0.1234)).toBe("-0.123");
  });

  it("does not sign zero", () => {
    expect(fmtSigned(0)).toBe("0.000");
  });
});

describe("extent", () => {
  it("falls back to [0, 1] for an empty list", () => {
    expect(extent([])).toEqual([0, 1]);
  });

  it("widens a single value to a non-degenerate domain", () => {
    expect(extent([2])).toEqual([1, 3]);
    expect(extent([0])).toEqual([-1, 1]);
  });

  it("finds min and max including negatives", () => {
    expect(extent([3, -1, 2])).toEqual([-1, 3]);
  });

  it("ignores NaN entries (falls back when nothing is comparable)", () => {
    expect(extent([Number.NaN])).toEqual([0, 1]);
  });
});

describe("linearScale", () => {
  it("maps the domain linearly onto the range", () => {
    const scale = linearScale([0, 10], [0, 100]);
    expect(scale(0)).toBe(0);
    expect(scale(5)).toBe(50);
    expect(scale(10)).toBe(100);
  });

  it("supports inverted ranges (SVG y-axes)", () => {
    const scale = linearScale([0, 1], [100, 0]);
    expect(scale(0.25)).toBe(75);
  });

  it("supports negative domains", () => {
    const scale = linearScale([-1, 1], [0, 10]);
    expect(scale(0)).toBe(5);
  });

  it("does not divide by zero on a degenerate domain", () => {
    const scale = linearScale([3, 3], [0, 10]);
    expect(Number.isFinite(scale(3))).toBe(true);
    expect(scale(3)).toBe(0);
  });
});

describe("ticks", () => {
  it("produces round ticks spanning the domain", () => {
    expect(ticks([0, 1])).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });

  it("returns the low endpoint for a zero-span domain", () => {
    expect(ticks([0, 0])).toEqual([0]);
    expect(ticks([5, 3])).toEqual([5]);
  });

  it("handles negative domains", () => {
    const out = ticks([-1, 1]);
    expect(out[0]).toBeGreaterThanOrEqual(-1);
    expect(out[out.length - 1]).toBeLessThanOrEqual(1 + 1e-9);
    expect(out).toContain(0);
  });

  it("keeps every tick inside the domain", () => {
    for (const domain of [[0, 0.37], [12, 480], [-3.2, -0.4]] as [number, number][]) {
      for (const t of ticks(domain)) {
        expect(t).toBeGreaterThanOrEqual(domain[0] - 1e-9);
        expect(t).toBeLessThanOrEqual(domain[1] + 1e-9);
      }
    }
  });
});
