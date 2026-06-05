import init, { pearson_equivalence_test } from "../pkg/ci_js.js";
import { beforeAll, describe, test, expect } from "vitest";

const wasm = await import("../pkg/ci_js.js");

let global_p = 0.05;

beforeAll(async () => {
  await wasm.default.init();
});

const toFloat64 = (...vals) => new Float64Array(vals);

describe("pearson_equivalence_tests", () => {
  test("independent_data_is_not_rejected", () => {
    const x = toFloat64(1, 1, 2, 2, 1, 1, 2, 2);
    const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
    const z = new Float64Array(0);

    const [p_value, coefficient] = pearson_equivalence_test(
      z,
      0,
      0,
      x,
      y,
      false,
      0.05,
      0.8,
    );

    expect(p_value).toBeLessThanOrEqual(0.05);
    expect(Math.abs(coefficient)).toBeLessThan(0.1);
  });

  test("dependent_data_is_rejected", () => {
    const x = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const y = toFloat64(3, 3, 3, 3, 6, 6, 6, 6);
    const z = new Float64Array(0);

    const [p_value, coefficient] = pearson_equivalence_test(
      z,
      0,
      0,
      x,
      y,
      false,
      0.05,
      0.1,
    );

    expect(p_value).toBeGreaterThanOrEqual(0.05);
    expect(Math.abs(coefficient)).toBeGreaterThan(0.9);
  });

  test("boolean mode returns true for independent data", () => {
    const x = toFloat64(1, 1, 2, 2, 1, 1, 2, 2);
    const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
    const z = new Float64Array(0);

    const result = pearson_equivalence_test(z, 0, 0, x, y, true, 0.05, 0.8);

    expect(result).toBe(true);
  });

  test("boolean mode returns false for dependent data", () => {
    const x = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const y = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const z = new Float64Array(0);

    const result = pearson_equivalence_test(z, 0, 0, x, y, true, 0.05, 0.1);

    expect(result).toBe(false);
  });
});
