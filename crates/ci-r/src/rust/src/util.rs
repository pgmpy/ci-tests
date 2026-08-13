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
    list!(
        statistic = opt_or_null(result.statistic),
        p_value = result.p_value,
        dof = opt_or_null(result.dof.map(|d| d as i32)),
        effect_size = opt_or_null(result.effect_size),
    )
    .into()
}

/// Wrap `Some(v)` as its R scalar and `None` as R `NULL` (not `NA`).
fn opt_or_null<T: Into<Robj>>(v: Option<T>) -> Robj {
    v.map_or_else(|| r!(NULL), Into::into)
}
