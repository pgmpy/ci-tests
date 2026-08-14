//! Shared machinery for the Fisher z-transform continuous tests.
//!
//! [`super::FisherZ`] and [`super::PearsonEquivalence`] both reduce a query to
//! the same three quantities — the partial correlation, its z-transform, and
//! the `√(n − |Z| − 3)` scale — and differ only in what they do with them.
//! Deriving those once keeps the two tests from drifting apart, which matters
//! because they must agree on the clipping bound and on which correlation is
//! reported as the effect size.

use statrs::distribution::Normal;

use crate::ci_tests::pearson_correlation::partial_correlation;
use crate::dataset::Dataset;
use crate::error::CiError;

/// Clipping bound for `rho` before the Fisher z-transform: `[-1 + EPS, 1 - EPS]`.
/// Matches the reference's `np.clip(rho, -0.999999, 0.999999)`.
pub(crate) const RHO_CLIP_EPS: f64 = 1e-6;

/// The Fisher z-transform quantities shared by the two continuous tests.
pub(crate) struct FisherZInputs {
    /// The partial correlation as computed, **un-clipped**. This is what both
    /// tests report as `effect_size`; the clip exists only to keep `atanh`
    /// finite and must not leak into the reported effect.
    pub(crate) rho_raw: f64,
    /// `atanh` of the clipped correlation.
    pub(crate) z_rho: f64,
    /// `√(n − |Z| − 3)`, the Fisher-z scale factor.
    pub(crate) scale: f64,
}

/// Derive the shared Fisher z-transform inputs for `x ⊥ y | z`.
///
/// # Errors
///
/// Propagates every error from [`partial_correlation`].
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z are the standard conditional-independence variable names from the contract"
)]
pub(crate) fn fisher_z_inputs(
    data: &Dataset,
    x: usize,
    y: usize,
    z: &[usize],
) -> Result<FisherZInputs, CiError> {
    let rho_raw = partial_correlation(data, x, y, z)?.0;
    let rho = rho_raw.clamp(-1.0 + RHO_CLIP_EPS, 1.0 - RHO_CLIP_EPS);

    // `partial_correlation` guarantees n >= |Z| + 3, so the radicand is >= 0.
    // At n == |Z| + 3 the scale is 0, giving statistic 0 and p = 1, matching
    // pgmpy.
    #[allow(clippy::cast_precision_loss)]
    let scale = ((data.n_rows() - z.len() - 3) as f64).sqrt();

    Ok(FisherZInputs {
        rho_raw,
        z_rho: rho.atanh(),
        scale,
    })
}

/// The standard normal distribution.
///
/// # Errors
///
/// Returns [`CiError::Numeric`] if construction fails, which cannot happen for
/// these fixed parameters but is not worth an unwrap in library code.
pub(crate) fn standard_normal() -> Result<Normal, CiError> {
    Normal::new(0.0, 1.0).map_err(|e| CiError::Numeric(format!("standard normal: {e}")))
}
