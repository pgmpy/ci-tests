import init, { chi_squared_test } from "../pkg/ci_js.js";
import { beforeAll, describe, test, expect } from "vitest";

const wasm = await import("../pkg/ci_js.js");

let precision = 1e-9;
let global_p = 0.05;

beforeAll(async () => {
  await wasm.default.init();
});

const toFloat64 = (...vals) => new Float64Array(vals);

describe("chi_squared_test", () => {
  test("independent data is not rejected", () => {
    const x = toFloat64(1, 1, 2, 2, 1, 1, 2, 2);
    const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
    const z = new Float64Array(0);

    const [p_value, statistic, dof] = chi_squared_test(
      z,
      0,
      0,
      x,
      y,
      false,
      global_p,
    );

    expect(statistic).toBeLessThan(precision);
    expect(p_value).toBeGreaterThanOrEqual(0.99);
    expect(dof).toBe(1);
  });

  test("dependent data is rejected", () => {
    const x = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const y = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const z = new Float64Array(0);

    const [p_value, statistic, dof] = chi_squared_test(
      z,
      0,
      0,
      x,
      y,
      false,
      global_p,
    );

    expect(Math.abs(statistic - 8.0)).toBeLessThan(precision);
    expect(Math.abs(p_value - 0.004677734981047276)).toBeLessThan(1e-12);
    expect(dof).toBe(1);
  });

  test("boolean mode returns true for independent data", () => {
    const x = toFloat64(1, 1, 2, 2, 1, 1, 2, 2);
    const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
    const z = new Float64Array(0);

    const result = chi_squared_test(z, 0, 0, x, y, true, global_p);

    expect(result).toBe(true);
  });

  test("boolean mode returns false for dependent data", () => {
    const x = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const y = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const z = new Float64Array(0);

    const result = chi_squared_test(z, 0, 0, x, y, true, global_p);

    expect(result).toBe(false);
  });
});
