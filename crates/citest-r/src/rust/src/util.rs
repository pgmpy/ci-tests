//! Conversion helpers from core types to R objects and errors.

use citest::error::CiError as CoreError;
use citest::strategy::{CiResult as CoreResult, DataType, IndependenceRule, TestMeta};
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

/// Convert a core [`TestMeta`] into the named R list
/// `list(name, data_types, symmetric, rule)`, matching the Python and
/// JavaScript bindings' `meta()`.
///
/// `rule` is the stable string `"p_value_ge"` or `"p_value_lt"`. Exposing it
/// means R code can ask a test how its p-value should be read instead of
/// hardcoding a list of which tests are inverted.
pub fn meta_to_list(meta: &TestMeta) -> Robj {
    let data_types: Vec<&str> = meta
        .data_types
        .iter()
        .map(|t| match t {
            DataType::Discrete => "discrete",
            DataType::Continuous => "continuous",
        })
        .collect();
    list!(
        name = meta.name,
        data_types = data_types,
        symmetric = meta.symmetric,
        rule = match meta.rule {
            IndependenceRule::PValueGe => "p_value_ge",
            IndependenceRule::PValueLt => "p_value_lt",
        },
    )
    .into()
}
