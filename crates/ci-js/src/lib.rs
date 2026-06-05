use ci_core::ci_tests::{
    chi_squared::ChiSquared, cressie_read::CressieRead, freeman_tukey::FreemanTukey,
    log_likelihood::LogLikelihood, modified_likelihood::ModifiedLikelihood,
    pearson_correlation::PearsonCorrelation, pearson_equivalence::PearsonEquivalence,
};
use ci_core::strategy::CITest;
use js_sys::Float64Array;
use wasm_bindgen::prelude::*;
mod util;

#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

macro_rules! wasm_ci_test {
    ($fn_name:ident, $inner:ty) => {
        /// Runs the CI test and returns the result as a JS value.
        ///
        /// Pass an empty `Float64Array` for `z_flat` to run unconditionally. When `boolean` is `true`,
        /// returns an independence verdict instead of the raw p-value and correlation.
        ///
        /// # Returns
        ///
        /// - `boolean` if `boolean=true`
        /// - `[p_value, coefficient]` if `boolean=false`
        ///
        /// # Errors
        ///
        /// Returns a `JsValue` error if:
        /// - the input arrays have invalid dimensions,
        /// - the statistical computation fails,
        /// - serialization to JavaScript fails.
        #[wasm_bindgen]
        pub fn $fn_name(
            z_flat: &Float64Array,
            z_rows: usize,
            z_cols: usize,
            x: &Float64Array,
            y: &Float64Array,
            boolean: bool,
            significance_level: f64,
        ) -> Result<JsValue, JsValue> {
            let (x, y, z) = util::convert_to_ndarray(z_flat, x, y, z_rows, z_cols)?;
            let test = <$inner>::new(boolean, significance_level);
            let result = test
                .run_test(x, y, z)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            util::convert_to_jsvalue(&result)
        }
    };
}

wasm_ci_test!(chi_squared_test, ChiSquared);
wasm_ci_test!(log_likelihood_test, LogLikelihood);
wasm_ci_test!(cressie_read_test, CressieRead);
wasm_ci_test!(pearson_correlation_test, PearsonCorrelation);
wasm_ci_test!(freeman_tukey_test, FreemanTukey);
wasm_ci_test!(modified_likelihood_test, ModifiedLikelihood);

/// Pearson equivalence CI test (TOST): declares independence when the partial correlation
/// lies within `[-delta_threshold, delta_threshold]`.
///
/// Pass an empty `Float64Array` for `z_flat` to run unconditionally. When `boolean` is `true`,
/// returns an independence verdict instead of the raw p-value and correlation.
///
/// # Returns
///
/// - `boolean` if `boolean=true`
/// - `[p_value, coefficient]` if `boolean=false`
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - the input arrays have invalid dimensions,
/// - the statistical computation fails,
/// - serialization to JavaScript fails.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn pearson_equivalence_test(
    z_flat: &Float64Array,
    z_rows: usize,
    z_cols: usize,
    x: &Float64Array,
    y: &Float64Array,
    boolean: bool,
    significance_level: f64,
    delta_threshold: f64,
) -> Result<JsValue, JsValue> {
    let (x, y, z) = util::convert_to_ndarray(z_flat, x, y, z_rows, z_cols)?;
    let citest = PearsonEquivalence::new(boolean, significance_level, delta_threshold);

    let result = citest
        .run_test(x, y, z)
        .map_err(|e| JsError::new(&e.to_string()))?;

    util::convert_to_jsvalue(&result)
}
