//! R bindings for the conditional independence testing library.
//!
//! Exposes CI test functions to R via the `extendr` framework.
//! Each CI test accepts paired observation vectors and a conditioning matrix, returning
//! a named R list whose shape depends on whether the test runs in boolean or numeric mode.

use ci_core::ci_tests::{
    ChiSquared, CressieRead, FreemanTukey, LogLikelihood, ModifiedLikelihood, PearsonCorrelation,
    PearsonEquivalence,
};
use ci_core::strategy::CITest;
use extendr_api::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
mod util;

/// Generates an R-callable wrapper for a [`CITest`] implementation.
///
/// The generated function signature is:
/// ```text
/// fn $fn_name(x_values, y_values, z, boolean, significance_level) -> Robj
/// ```
/// - `x_values` / `y_values`: paired observation vectors.
/// - `z`: conditioning matrix; pass a 0-column matrix for unconditional tests.
/// - `boolean`: when `true`, returns only an independence verdict at `significance_level`
///   instead of the raw test statistic and p-value.
/// - `significance_level`: threshold used only when `boolean` is `true`.
macro_rules! r_ci_test {
    ($(#[$meta:meta])* $fn_name:ident, $inner:ty) => {
        $(#[$meta])*
        #[extendr]
        fn $fn_name(
            x_values: ArrayView1<f64>,
            y_values: ArrayView1<f64>,
            z: ArrayView2<f64>,
            boolean: bool,
            significance_level: f64,
        ) -> anyhow::Result<Robj> {
            let citest = <$inner>::new(boolean, significance_level);
            let result = citest.run_test(x_values.to_owned(), y_values.to_owned(), z.to_owned())?;
            Ok(util::test_result_to_robj(result))
        }
    };
}

r_ci_test!(
    /// Pearson chi-squared conditional independence test (λ = 1).
    ///
    /// Operates on discrete data only. Pass a 0-column matrix for `z` to run
    /// unconditionally. When `boolean` is `true`, returns an independence verdict
    /// instead of the raw test statistic and p-value.
    chi_squared_test, ChiSquared
);
r_ci_test!(
    /// Log-likelihood ratio (G-test) conditional independence test (λ = 0).
    ///
    /// Operates on discrete data only. Pass a 0-column matrix for `z` to run
    /// unconditionally. When `boolean` is `true`, returns an independence verdict
    /// instead of the raw test statistic and p-value.
    log_likelihood_test, LogLikelihood
);
r_ci_test!(
    /// Cressie–Read conditional independence test (λ = 2/3).
    ///
    /// Operates on discrete data only. Pass a 0-column matrix for `z` to run
    /// unconditionally. When `boolean` is `true`, returns an independence verdict
    /// instead of the raw test statistic and p-value.
    cressie_read_test, CressieRead
);
r_ci_test!(
    /// Pearson partial correlation conditional independence test.
    ///
    /// Should be used on continuous data only. When `z` is non-empty, uses linear
    /// regression residuals to compute the partial correlation. Pass a 0-column
    /// matrix for `z` to run unconditionally. When `boolean` is `true`, returns an
    /// independence verdict instead of the raw test statistic and p-value.
    pearson_correlation_test, PearsonCorrelation
);
r_ci_test!(
    /// Freeman–Tukey conditional independence test (λ = −1/2).
    ///
    /// Operates on discrete data only. Pass a 0-column matrix for `z` to run
    /// unconditionally. When `boolean` is `true`, returns an independence verdict
    /// instead of the raw test statistic and p-value.
    freeman_tukey_test, FreemanTukey
);
r_ci_test!(
    /// Modified log-likelihood ratio conditional independence test (λ = −1).
    ///
    /// Operates on discrete data only. Pass a 0-column matrix for `z` to run
    /// unconditionally. When `boolean` is `true`, returns an independence verdict
    /// instead of the raw test statistic and p-value.
    modified_likelihood_test, ModifiedLikelihood
);

/// Pearson equivalence CI test (TOST): declares independence when the partial correlation
/// lies within `[-delta_threshold, delta_threshold]`.
///
/// Pass a 0-column matrix for `z` to run unconditionally. When `boolean` is `true`,
/// returns an independence verdict instead of the raw p-value and correlation.
#[extendr]
fn pearson_equivalence_test(
    x_values: ArrayView1<f64>,
    y_values: ArrayView1<f64>,
    z: ArrayView2<f64>,
    boolean: bool,
    significance_level: f64,
    delta_threshold: f64,
) -> anyhow::Result<Robj> {
    let citest = PearsonEquivalence::new(boolean, significance_level, delta_threshold);
    let result = citest.run_test(x_values.to_owned(), y_values.to_owned(), z.to_owned())?;
    Ok(util::test_result_to_robj(result))
}

extendr_module! {
    mod cir;
    fn chi_squared_test;
    fn log_likelihood_test;
    fn cressie_read_test;
    fn pearson_correlation_test;
    fn freeman_tukey_test;
    fn modified_likelihood_test;
    fn pearson_equivalence_test;
}
