// API-surface tests for the data-bound JS bindings: implicit construction
// (raw columns object vs shared Dataset), string categoricals, the strict
// NaN policy, query validation, and fisher_z.

import { describe, test, expect, beforeAll } from "vitest";
import pkg from "../pkg/ci_js.js";
const { Dataset, ChiSquared, FisherZ, PearsonCorrelation, init } = pkg;

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
      A: { kind: "discrete", values: ["yes", "no", "yes", "no", "maybe", "yes"] },
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
    const X = [], Y = [], Z = [];
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
    expect(r.statistic).toBeCloseTo(Math.sqrt(n - 4) * Math.atanh(pc.statistic), 9);
  });
});

describe("strict config", () => {
  test("unknown config keys throw instead of being ignored", () => {
    expect(
      () => new ChiSquared(DISCRETE_COLS, { yate: false }), // typo of yates
    ).toThrow(/invalid config object/);
    expect(
      () => new FisherZ(new Dataset(DISCRETE_COLS), { foo: 1 }),
    ).toThrow(/invalid config object/);
  });

  test("valid configs still work", () => {
    expect(new ChiSquared(DISCRETE_COLS, {}).runTest("A", "B", []).pValue).toBeGreaterThan(0);
    expect(
      new ChiSquared(DISCRETE_COLS, { yates: false }).runTest("A", "B", []).pValue,
    ).toBeGreaterThan(0);
  });
});
