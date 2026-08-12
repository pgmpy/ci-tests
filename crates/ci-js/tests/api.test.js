// API-surface tests for the data-bound JS bindings: implicit construction
// (raw columns object vs shared Dataset), string categoricals, the strict
// NaN policy, query validation, and fisher_z.

import { describe, test, expect, beforeAll } from "vitest";
import pkg from "../pkg/ci_js.js";
const {
  Dataset,
  ChiSquared,
  FisherZ,
  PearsonCorrelation,
  PearsonEquivalence,
  init,
} = pkg;

beforeAll(() => init());

const DISCRETE_COLS = {
  A: { kind: "discrete", values: [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0] },
  B: { kind: "discrete", values: [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0] },
};

describe("implicit construction", () => {
  test("raw columns object and shared Dataset give identical results", () => {
    const viaRaw = new ChiSquared(DISCRETE_COLS);
    const data = new Dataset(DISCRETE_COLS);
    const viaDataset = new ChiSquared(data);
    const a = viaRaw.runTest("A", "B", []);
    const b = viaDataset.runTest("A", "B", []);
    expect(a.statistic).toBeCloseTo(b.statistic, 12);
    expect(a.pValue).toBeCloseTo(b.pValue, 12);
  });

  test("one Dataset can be shared across several tests without being consumed", () => {
    const data = new Dataset(DISCRETE_COLS);
    const chi1 = new ChiSquared(data);
    const chi2 = new ChiSquared(data, { yates: false });
    expect(chi1.runTest("A", "B", []).pValue).toBeGreaterThan(0);
    expect(chi2.runTest("A", "B", []).pValue).toBeGreaterThan(0);
    // The caller's Dataset must remain usable afterwards.
    expect(data.nRows()).toBe(12);
  });
});

describe("string categoricals", () => {
  test("discrete columns accept string values", () => {
    const data = new Dataset({
      A: {
        kind: "discrete",
        values: ["yes", "no", "yes", "no", "maybe", "yes"],
      },
      B: { kind: "discrete", values: [0, 1, 0, 1, 0, 1] },
    });
    const r = new ChiSquared(data).runTest("A", "B", []);
    expect(r.pValue).toBeGreaterThanOrEqual(0);
    expect(r.pValue).toBeLessThanOrEqual(1);
  });

  test("continuous columns reject string values", () => {
    expect(
      () => new Dataset({ X: { kind: "continuous", values: ["a", "b"] } }),
    ).toThrow(/numeric/);
  });

  test("mixed string/number arrays are rejected either way around", () => {
    expect(
      () => new Dataset({ A: { kind: "discrete", values: [1, "a"] } }),
    ).toThrow(/mixed string\/number/);
    expect(
      () => new Dataset({ A: { kind: "discrete", values: ["a", 1] } }),
    ).toThrow(/mixed string\/number/);
  });
});

describe("typed arrays", () => {
  test("Float64Array values match the plain-array form", () => {
    const plain = [0.4, 1.3, -0.2, 0.9, 0.1, -0.7, 0.6, 1.1];
    const other = [1.0, -0.3, 0.8, 0.2, -0.9, 0.5, -0.1, 0.7];
    const viaArray = new Dataset({
      X: { kind: "continuous", values: plain },
      Y: { kind: "continuous", values: other },
    });
    const viaTyped = new Dataset({
      X: { kind: "continuous", values: new Float64Array(plain) },
      Y: { kind: "continuous", values: new Float64Array(other) },
    });
    const a = new PearsonCorrelation(viaArray).runTest("X", "Y", []);
    const b = new PearsonCorrelation(viaTyped).runTest("X", "Y", []);
    expect(b.statistic).toBeCloseTo(a.statistic, 14);
    expect(b.pValue).toBeCloseTo(a.pValue, 14);
  });
});

describe("strict NaN policy and validation", () => {
  test("NaN errors at Dataset construction", () => {
    expect(
      () => new Dataset({ X: { kind: "continuous", values: [1, NaN, 2] } }),
    ).toThrow(/missing data/);
  });

  test("null errors at Dataset construction", () => {
    expect(
      () => new Dataset({ X: { kind: "continuous", values: [1, null, 2] } }),
    ).toThrow();
  });

  test("x == y is an invalid query", () => {
    const chi = new ChiSquared(DISCRETE_COLS);
    expect(() => chi.runTest("A", "A", [])).toThrow(/invalid query/);
  });
});

describe("fisher_z", () => {
  test("runs and reports no dof", () => {
    const n = 60;
    const X = [],
      Y = [],
      Z = [];
    let s = 42;
    const rand = () => {
      // deterministic LCG in (-1, 1)
      s = (s * 48271) % 2147483647;
      return (s / 2147483647) * 2 - 1;
    };
    for (let i = 0; i < n; i++) {
      const z = rand();
      Z.push(z);
      X.push(1.2 * z + 0.3 * rand());
      Y.push(0.7 * z + 0.3 * rand());
    }
    const data = new Dataset({
      X: { kind: "continuous", values: X },
      Y: { kind: "continuous", values: Y },
      Z: { kind: "continuous", values: Z },
    });
    const fz = new FisherZ(data);
    const r = fz.runTest("X", "Y", ["Z"]);
    expect(r.dof).toBeNull();
    expect(r.pValue).toBeGreaterThan(0);
    const pc = new PearsonCorrelation(data).runTest("X", "Y", ["Z"]);
    // statistic = sqrt(n - 1 - 3) * atanh(r)
    expect(r.statistic).toBeCloseTo(
      Math.sqrt(n - 4) * Math.atanh(pc.statistic),
      9,
    );
  });
});

describe("strict config", () => {
  test("unknown config keys throw instead of being ignored", () => {
    expect(
      () => new ChiSquared(DISCRETE_COLS, { yate: false }), // typo of yates
    ).toThrow(/invalid config object/);
    expect(() => new FisherZ(new Dataset(DISCRETE_COLS), { foo: 1 })).toThrow(
      /invalid config object/,
    );
  });

  test("valid configs still work", () => {
    expect(
      new ChiSquared(DISCRETE_COLS, {}).runTest("A", "B", []).pValue,
    ).toBeGreaterThan(0);
    expect(
      new ChiSquared(DISCRETE_COLS, { yates: false }).runTest("A", "B", [])
        .pValue,
    ).toBeGreaterThan(0);
  });
});

describe("Dataset introspection", () => {
  test("nCols and indexOf resolve names to insertion-order indices", () => {
    const data = new Dataset(DISCRETE_COLS);
    expect(data.nCols()).toBe(2);
    expect(data.nRows()).toBe(12);
    // Object.entries preserves insertion order, so A -> 0, B -> 1.
    expect(data.indexOf("A")).toBe(0);
    expect(data.indexOf("B")).toBe(1);
    // A missing name resolves to `undefined` (Option::None), not an error.
    expect(data.indexOf("nope")).toBeUndefined();
  });
});

describe("meta", () => {
  test("a discrete test reports its name, data types, symmetry, and PValueGe rule", () => {
    const m = new ChiSquared(DISCRETE_COLS).meta();
    expect(m.name).toBe("chi_squared");
    expect(m.dataTypes).toEqual(["discrete"]);
    expect(m.symmetric).toBe(true);
    expect(m.rule).toBe("p_value_ge");
  });

  test("a continuous test reports a continuous data type and PValueGe rule", () => {
    const m = new PearsonCorrelation(DISCRETE_COLS).meta();
    expect(m.name).toBe("pearson_correlation");
    expect(m.dataTypes).toEqual(["continuous"]);
    expect(m.symmetric).toBe(true);
    expect(m.rule).toBe("p_value_ge");
  });

  test("the equivalence test advertises the inverted PValueLt rule", () => {
    const m = new PearsonEquivalence(DISCRETE_COLS).meta();
    expect(m.name).toBe("pearson_equivalence");
    expect(m.rule).toBe("p_value_lt");
  });
});

describe("effectSize result field", () => {
  test("ChiSquared populates effectSize (Cramér's V) for a 2x2 table", () => {
    const r = new ChiSquared(DISCRETE_COLS).runTest("A", "B", []);
    expect(typeof r.effectSize).toBe("number");
    expect(Number.isFinite(r.effectSize)).toBe(true);
    expect(r.effectSize).toBeGreaterThanOrEqual(0);
  });

  test("PearsonCorrelation populates effectSize (|r|) for continuous data", () => {
    const data = new Dataset({
      X: {
        kind: "continuous",
        values: [0.4, 1.3, -0.2, 0.9, 0.1, -0.7, 0.6, 1.1],
      },
      Y: {
        kind: "continuous",
        values: [1.0, -0.3, 0.8, 0.2, -0.9, 0.5, -0.1, 0.7],
      },
    });
    const r = new PearsonCorrelation(data).runTest("X", "Y", []);
    expect(typeof r.effectSize).toBe("number");
    expect(Number.isFinite(r.effectSize)).toBe(true);
    expect(r.effectSize).toBeGreaterThanOrEqual(0);
    expect(r.effectSize).toBeLessThanOrEqual(1);
  });

  test("effectSize is null when Cramér's V is undefined (a constant column)", () => {
    // A constant column has cardinality 1, so min(kx, ky) < 2 and Cramér's V is
    // undefined; the core returns None, which surfaces as JS `null`.
    const data = new Dataset({
      C: { kind: "discrete", values: [0, 0, 0, 0, 0, 0] },
      B: { kind: "discrete", values: [0, 1, 0, 1, 0, 1] },
    });
    const r = new ChiSquared(data).runTest("C", "B", []);
    expect(r.effectSize).toBeNull();
  });
});

describe("isIndependent", () => {
  // Exactly-zero correlation by construction: each x value pairs with +1 and -1
  // in y, so the covariance is 0 and the standard p-value is 1.
  function nearZeroCorrelation() {
    const X = [];
    const Y = [];
    for (let i = 0; i < 50; i++) {
      X.push(i);
      Y.push(1);
      X.push(i);
      Y.push(-1);
    }
    return new Dataset({
      x: { kind: "continuous", values: X },
      y: { kind: "continuous", values: Y },
    });
  }

  test("a standard test and PearsonEquivalence reach opposite decisions on the same data", () => {
    const data = nearZeroCorrelation();
    const pc = new PearsonCorrelation(data);
    // A tiny equivalence margin: r == 0 cannot be shown to lie *within* it, so
    // the TOST p-value stays large and the PValueLt rule refuses independence.
    const eq = new PearsonEquivalence(data, { deltaThreshold: 0.001 });

    const pcIndep = pc.isIndependent("x", "y", [], 0.05);
    const eqIndep = eq.isIndependent("x", "y", [], 0.05);

    // Standard (PValueGe): r == 0 gives p == 1 >= 0.05 -> independent.
    expect(pcIndep).toBe(true);
    // Equivalence (PValueLt): TOST p stays >= 0.05 -> NOT independent.
    expect(eqIndep).toBe(false);
    // The two rules therefore disagree on identical data.
    expect(pcIndep).not.toBe(eqIndep);

    // The decision must match each test's documented rule applied to its p-value.
    const alpha = 0.05;
    expect(pcIndep).toBe(pc.runTest("x", "y", []).pValue >= alpha);
    expect(eqIndep).toBe(eq.runTest("x", "y", []).pValue < alpha);
  });

  test("omitting significanceLevel defaults to 0.05 (not NaN)", () => {
    const chi = new ChiSquared(DISCRETE_COLS);
    // Omitting the level must behave like passing 0.05 explicitly, not coerce to
    // NaN (which would make every comparison false and silently return false).
    expect(chi.isIndependent("A", "B", [])).toBe(
      chi.isIndependent("A", "B", [], 0.05),
    );
  });

  test("a non-finite significanceLevel throws instead of silently returning false", () => {
    const chi = new ChiSquared(DISCRETE_COLS);
    expect(() => chi.isIndependent("A", "B", [], NaN)).toThrow(/finite/);
    expect(() => chi.isIndependent("A", "B", [], Infinity)).toThrow(/finite/);
    expect(() => chi.isIndependent("A", "B", [], -Infinity)).toThrow(/finite/);
  });
});
