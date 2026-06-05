import init, { cressie_read_test } from "../pkg/ci_js.js";
import { beforeAll, describe, test, expect } from "vitest";

const wasm = await import("../pkg/ci_js.js");

let precision = 1e-9;
let global_p = 0.05;

beforeAll(async () => {
  await wasm.default.init();
});
const toFloat64 = (...vals) => new Float64Array(vals);

describe("cressie_read_test", () => {
  test("unconditional independent data is not rejected", () => {
    const x = toFloat64(1, 1, 2, 2, 1, 1, 2, 2);
    const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
    const z = new Float64Array(0);

    const [p_value, statistic, dof] = cressie_read_test(
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

  test("unconditional boolean accepts independent data", () => {
    const x = toFloat64(1, 1, 1, 1, 2, 2, 2, 2);
    const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
    const z = new Float64Array(0);

    const result = cressie_read_test(z, 0, 0, x, y, true, global_p);

    expect(result).toBe(true);
  });

  test("unconditional dependent data is rejected", () => {
    const x = new Float64Array([1, 1, 1, 1, 2, 2, 2, 2]);
    const y = new Float64Array([1, 1, 1, 1, 2, 2, 2, 2]);
    const z = new Float64Array(0);

    const [p_value, statistic, dof] = cressie_read_test(
      z,
      0,
      0,
      x,
      y,
      false,
      global_p,
    );

    expect(statistic).toBeGreaterThan(5.0);
    expect(p_value).toBeLessThan(global_p);
    expect(dof).toBe(1);
  });

  test("unconditional boolean rejects dependent data", () => {
    const x = new Float64Array([1, 1, 1, 1, 2, 2, 2, 2]);
    const y = new Float64Array([1, 1, 1, 1, 2, 2, 2, 2]);
    const z = new Float64Array(0);

    const result = cressie_read_test(z, 0, 0, x, y, true, global_p);

    expect(result).toBe(false);
  });

  test("conditional independent data is not rejected", () => {
    const x = new Float64Array([1, 1, 2, 2, 1, 1, 2, 2]);
    const y = new Float64Array([1, 2, 1, 2, 1, 2, 1, 2]);
    const z = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]); // 8×1, row-major

    const [p_value, statistic, dof] = cressie_read_test(
      z,
      8,
      1,
      x,
      y,
      false,
      global_p,
    );

    expect(statistic).toBeLessThan(precision);
    expect(p_value).toBeGreaterThan(0.99);
    expect(dof).toBe(2);
  });

  test("conditional boolean accepts conditionally independent data", () => {
    const x = new Float64Array([1, 1, 2, 2, 1, 1, 2, 2]);
    const y = new Float64Array([1, 2, 1, 2, 1, 2, 1, 2]);
    const z = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]);

    const result = cressie_read_test(z, 8, 1, x, y, true, global_p);

    expect(result).toBe(true);
  });

  test("conditional dependent data is rejected", () => {
    const x = new Float64Array([1, 1, 2, 2, 1, 1, 2, 2]);
    const y = new Float64Array([1, 1, 2, 2, 1, 1, 2, 2]);
    const z = new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]);

    const [p_value, statistic] = cressie_read_test(
      z,
      8,
      1,
      x,
      y,
      false,
      global_p,
    );

    expect(statistic).toBeGreaterThan(5.0);
    expect(p_value).toBeLessThan(global_p);
  });
});
