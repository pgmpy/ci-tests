//! Shared implementation for the power-divergence family of discrete tests.
//!
//! Every member ([`crate::ci_tests::ChiSquared`], [`LogLikelihood`], etc.) is a
//! struct carrying a single `yates: bool` flag and delegating to the
//! λ-parameterized discrete back-end with its fixed `λ`. This module factors out
//! the shared statistic → [`CiResult`] conversion (p-value from the chi-squared
//! survival function, Cramér's V effect size) and the [`crate::strategy::TestMeta`]
//! construction so each test file stays focused on its name and `λ`.
//!
//! [`LogLikelihood`]: crate::ci_tests::LogLikelihood

use statrs::distribution::{ChiSquared as ChiSquaredDist, ContinuousCDF};

use crate::dataset::Dataset;
use crate::discrete::{
    power_divergence_conditional, power_divergence_unconditional, DiscreteOutcome,
};
use crate::error::CiError;
use crate::strategy::{CiResult, DataType, IndependenceRule, TestMeta};

/// Cramér's V = sqrt(stat / (n * (min(kx, ky) - 1))). Returns `None` when the
/// smaller cardinality is < 2 (V undefined) or there are no rows.
///
/// The statistic is mathematically non-negative, but `powf` rounding in the
/// power-divergence back-end can make it a tiny *negative* value on tables where
/// observed == expected (e.g. a balanced independent table under Cressie-Read),
/// which would turn the `sqrt` into `NaN`. Clamp the radicand at 0 so the effect
/// size is a clean `0.0` there.
#[allow(clippy::cast_precision_loss)]
fn cramers_v(statistic: f64, n: usize, kx: usize, ky: usize) -> Option<f64> {
    let k = kx.min(ky);
    if k < 2 || n == 0 {
        return None;
    }
    Some((statistic.max(0.0) / (n as f64 * (k as f64 - 1.0))).sqrt())
}

/// Survival function `P(χ²_dof > statistic)`, with the degenerate and infinite
/// cases handled exactly: `dof == 0 ⇒ 1.0`, `statistic == +∞ ⇒ 0.0`.
fn chi_squared_sf(statistic: f64, dof: usize) -> Result<f64, CiError> {
    if dof == 0 {
        return Ok(1.0);
    }
    if statistic.is_infinite() {
        return Ok(0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let p = ChiSquaredDist::new(dof as f64)
        .map_err(|e| CiError::Numeric(format!("chi-squared distribution: {e}")))?
        .sf(statistic);
    Ok(p)
}

/// Run a power-divergence test with parameter `lambda` and Yates flag `yates`,
/// returning the full [`CiResult`] (statistic, p-value via the chi-squared
/// survival function, dof, Cramér's V effect size).
///
/// # Errors
///
/// Propagates [`CiError`] from reading the columns or constructing the
/// chi-squared distribution.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z are the standard conditional-independence variable names from the contract"
)]
pub(crate) fn run_power_divergence(
    data: &Dataset,
    x: usize,
    y: usize,
    z: &[usize],
    lambda: f64,
    yates: bool,
) -> Result<CiResult, CiError> {
    let (x_codes, kx) = data.discrete(x)?;
    let (y_codes, ky) = data.discrete(y)?;

    let DiscreteOutcome { statistic, dof } = if z.is_empty() {
        power_divergence_unconditional(x_codes, y_codes, kx, ky, lambda, yates)
    } else {
        let partition = data.strata_partition(z)?;
        power_divergence_conditional(x_codes, y_codes, kx, ky, &partition, lambda, yates)
    };

    let p_value = chi_squared_sf(statistic, dof)?;
    let effect_size = cramers_v(statistic, data.n_rows(), kx, ky);

    Ok(CiResult {
        statistic: Some(statistic),
        p_value,
        dof: Some(dof),
        effect_size,
    })
}

/// Build the [`TestMeta`] common to every discrete power-divergence test: the
/// given stable `name`, `[Discrete]`, symmetric, and the standard
/// null-of-independence [`IndependenceRule::PValueGe`].
pub(crate) fn discrete_meta(name: &'static str) -> TestMeta {
    TestMeta {
        name,
        data_types: &[DataType::Discrete],
        symmetric: true,
        rule: IndependenceRule::PValueGe,
    }
}
