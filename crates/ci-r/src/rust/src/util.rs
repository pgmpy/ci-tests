//! Conversion helpers from core types to R objects and errors.

use ci_core::error::CiError as CoreError;
use ci_core::strategy::CiResult as CoreResult;
use extendr_api::prelude::*;

/// Map a core [`CoreError`] onto an extendr [`Error`], which extendr raises as
/// an R error at the call site.
pub fn to_r_error(err: CoreError) -> Error {
    Error::Other(err.to_string())
}

/// Convert a core [`CoreResult`] into the uniform named R list
/// `list(statistic, p_value, dof, effect_size)`.
///
/// Absent optional fields (`statistic`, `dof`, `effect_size`) become R `NULL`;
/// `p_value` is always present. `dof` is returned as an integer.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "dof is a small degrees-of-freedom count, far below i32::MAX"
)]
pub fn result_to_list(result: &CoreResult) -> Robj {
    let statistic: Robj = match result.statistic {
        Some(s) => s.into(),
        None => r!(NULL),
    };
    let dof: Robj = match result.dof {
        Some(d) => (d as i32).into(),
        None => r!(NULL),
    };
    let effect_size: Robj = match result.effect_size {
        Some(e) => e.into(),
        None => r!(NULL),
    };
    list!(
        statistic = statistic,
        p_value = result.p_value,
        dof = dof,
        effect_size = effect_size,
    )
    .into()
}
