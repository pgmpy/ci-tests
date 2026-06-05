import init, {
  pearson_correlation_test,
  chi_squared_test,
  pearson_equivalence_test,
} from "../pkg/ci_js.js";
import { beforeAll, describe, test, expect } from "vitest";

const wasm = await import("../pkg/ci_js.js");

beforeAll(async () => {
  await wasm.default.init();
});

const toFloat64 = (...vals) => new Float64Array(vals);

describe("error handling", () => {
  test("throws when z_flat length does not match z_rows * z_cols", () => {
    const x = toFloat64(1, 2, 3, 4, 5);
    const y = toFloat64(1, 2, 3, 4, 5);
    // z_flat has 6 elements but z_rows * z_cols = 5 * 2 = 10
    const z_flat = toFloat64(1, 2, 3, 4, 5, 6);

    expect(() =>
      pearson_correlation_test(z_flat, 5, 2, x, y, false, 0.05),
    ).toThrow();
  });

  test("throws when x and y have different lengths", () => {
    const x = toFloat64(1, 2, 3, 4, 5);
    const y = toFloat64(1, 2, 3);
    const z = new Float64Array(0);

    expect(() =>
      pearson_correlation_test(z, 0, 0, x, y, false, 0.05),
    ).toThrow();
  });

  test("throws when x is empty", () => {
    const x = new Float64Array(0);
    const y = new Float64Array(0);
    const z = new Float64Array(0);

    expect(() =>
      pearson_correlation_test(z, 0, 0, x, y, false, 0.05),
    ).toThrow();
  });

  test("throws when z_flat length does not match for chi_squared", () => {
    const x = toFloat64(1, 1, 2, 2);
    const y = toFloat64(1, 2, 1, 2);
    const z_flat = toFloat64(1, 2, 3, 4);

    expect(() => chi_squared_test(z_flat, 4, 2, x, y, false, 0.05)).toThrow();
  });

  test("throws when z_flat length does not match for pearson_equivalence", () => {
    const x = toFloat64(1, 2, 3, 4, 5);
    const y = toFloat64(1, 2, 3, 4, 5);
    const z_flat = toFloat64(1, 2, 3);

    expect(() =>
      pearson_equivalence_test(z_flat, 5, 1, x, y, false, 0.05, 0.1),
    ).toThrow();
  });

  test("does not throw for valid conditional test", () => {
    const x = toFloat64(1, 2, 3, 4, 5);
    const y = toFloat64(1, 2, 3, 4, 5);
    const z_flat = toFloat64(1, 2, 3, 4, 6);

    expect(() =>
      pearson_correlation_test(z_flat, 5, 1, x, y, false, 0.05),
    ).not.toThrow();
  });

  test("does not throw for valid unconditional test", () => {
    const x = toFloat64(1, 2, 3, 4, 5);
    const y = toFloat64(1, 2, 3, 4, 5);
    const z = new Float64Array(0);

    expect(() =>
      pearson_correlation_test(z, 0, 0, x, y, false, 0.05),
    ).not.toThrow();
  });
});
