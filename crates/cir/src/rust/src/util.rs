//! Conversion utilities from core result types to R objects.

use ci_core::strategy::TestResult;
use extendr_api::prelude::*;

/// Converts a [`TestResult`] into a named R list.
///
/// The returned list always contains a `kind` field identifying the variant:
/// - `"pvalue"` — also contains `p_value` and `coefficient`.
/// - `"statistic"` — also contains `statistic`, `p_value`, and `df` (degrees of freedom).
/// - `"boolean"` — also contains `independent` (logical).
pub fn test_result_to_robj(r: TestResult) -> Robj {
    match r {
        TestResult::PValue(p, coef) => {
            list!(kind = "pvalue", p_value = p, coefficient = coef,).into()
        }
        TestResult::Statistic(p, stat, df) => list!(
            kind = "statistic",
            statistic = stat,
            p_value = p,
            df = df as i32,
        )
        .into(),
        TestResult::Boolean(b) => list!(kind = "boolean", independent = b,).into(),
    }
}
