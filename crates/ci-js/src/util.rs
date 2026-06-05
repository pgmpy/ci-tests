use ci_core::strategy::TestResult;
use js_sys::{Array, Float64Array};
use ndarray::{Array1, Array2};
use wasm_bindgen::prelude::*;

/// Converts JavaScript `Float64Array` inputs into `ndarray` types for use in CI tests.
///
/// # Parameters
///
/// - `z_flat` - Row-major flattened conditioning matrix. Pass an empty `Float64Array`
///   for unconditional testing.
/// - `x` - First variable as a `Float64Array`.
/// - `y` - Second variable as a `Float64Array`.
/// - `z_rows` - Number of rows in the conditioning matrix. Ignored if `z_flat` is empty.
/// - `z_cols` - Number of columns in the conditioning matrix. Ignored if `z_flat` is empty.
///
/// # Returns
///
/// A tuple `(x, y, z)` where:
/// - `x` and `y` are `Array1<f64>` vectors
/// - `z` is an `Array2<f64>` matrix with shape `(n, 0)` for unconditional tests,
///   or `(z_rows, z_cols)` for conditional tests
///
/// # Errors
///
/// Returns a `JsValue` error if `z_flat.len() != z_rows * z_cols`.
#[allow(clippy::type_complexity)]
pub fn convert_to_ndarray(
    z_flat: &Float64Array,
    x: &Float64Array,
    y: &Float64Array,
    z_rows: usize,
    z_cols: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>), JsValue> {
    if u32::try_from(z_rows * z_cols).unwrap_or(u32::MAX) != z_flat.length() {
        return Err(JsValue::from_str(
            "z_rows*z_cols doesnt match the size of z_flat",
        ));
    }
    let x_vec: Vec<f64> = x.to_vec();
    let y_vec: Vec<f64> = y.to_vec();
    let z: Array2<f64> = if z_flat.length() == 0 {
        Array2::zeros((x_vec.len(), 0))
    } else {
        let z_vec: Vec<f64> = z_flat.to_vec();
        Array2::from_shape_vec((z_rows, z_cols), z_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?
    };
    let x: Array1<f64> = Array1::from_vec(x_vec);
    let y: Array1<f64> = Array1::from_vec(y_vec);

    Ok((x, y, z))
}

/// Converts a `TestResult` into a JavaScript value.
///
/// # Returns
///
/// The return type depends on the `TestResult` variant:
///
/// - `TestResult::Boolean` — returns a JS `boolean`
/// - `TestResult::PValue` — returns a JS `Array` of `[p_value, coefficient]`
/// - `TestResult::Statistic` — returns a JS `Array` of `[p_value, statistic, dof]`
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::unnecessary_wraps)]
pub fn convert_to_jsvalue(result: &TestResult) -> Result<JsValue, JsValue> {
    match result {
        TestResult::Boolean(b) => Ok(JsValue::from_bool(*b)),
        TestResult::PValue(p_value, coefficient) => {
            let array = Array::new();
            array.push(&JsValue::from_f64(*p_value));
            array.push(&JsValue::from_f64(*coefficient));
            Ok(array.into())
        }

        TestResult::Statistic(p_value, statistic, dof) => {
            let array = Array::new();
            array.push(&JsValue::from_f64(*p_value));
            array.push(&JsValue::from_f64(*statistic));
            array.push(&JsValue::from_f64(*dof as f64));
            Ok(array.into())
        }
    }
}
