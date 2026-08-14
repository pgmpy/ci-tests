//! Pearson / partial-correlation conditional-independence test (continuous).

use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::dataset::Dataset;
use crate::error::CiError;
use crate::gram::GramCache;
use crate::strategy::{CITest, CiResult, DataType, IndependenceRule, TestMeta};

/// Relative tolerance for declaring a post-conditioning residual vector
/// "constant": its deviation energy is negligible compared to the original
/// variable's. This catches the case where Z perfectly explains X or Y, leaving
/// residuals that are pure floating-point noise (~1e-15). It sits ~30 orders of
/// magnitude below the smallest genuine residual ratio in the golden fixture, so
/// it never affects a real partial correlation.
const VARIANCE_REL_EPS: f64 = 1e-12;

/// Pearson correlation test. With a non-empty conditioning set it reports the
/// partial correlation given Z.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PearsonCorrelation;

impl PearsonCorrelation {
    /// Construct the test (no configuration in this slice).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// The "input is constant; Pearson correlation is undefined" error for input
/// `which` (`"x"` or `"y"`).
fn constant_input_error(which: &str) -> CiError {
    CiError::DegenerateData(format!(
        "input `{which}` is constant; Pearson correlation is undefined"
    ))
}

/// Enforce the minimum row count for a (partial) correlation: 3 rows
/// unconditionally, or `|Z| + 3` when conditioning on `n_z` columns.
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] if there are too few rows.
fn check_row_count(n: usize, n_z: usize) -> Result<(), CiError> {
    if n_z == 0 {
        if n < 3 {
            return Err(CiError::DegenerateData(format!(
                "need at least 3 rows for correlation, got {n}"
            )));
        }
    } else if n < n_z + 3 {
        return Err(CiError::DegenerateData(format!(
            "need at least |Z| + 3 = {} rows, got {n}",
            n_z + 3
        )));
    }
    Ok(())
}

/// Reject a post-conditioning residual whose deviation energy is negligible
/// relative to the original variable's (Z explains essentially all variation).
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] if `residual_var <= VARIANCE_REL_EPS *
/// original_var`.
fn check_residual_variance(
    which: &str,
    residual_var: f64,
    original_var: f64,
) -> Result<(), CiError> {
    if residual_var <= VARIANCE_REL_EPS * original_var {
        return Err(CiError::DegenerateData(format!(
            "residual `{which}` is constant after conditioning on Z; \
             partial correlation is undefined"
        )));
    }
    Ok(())
}

/// Compute the (partial) correlation `r` and its degrees of freedom for
/// `x ⊥ y | z`.
///
/// With empty `z` this is the plain Pearson r with `dof = n - 2`. With `z` it
/// is the partial correlation given Z with `dof = n - |Z| - 2`. Computed in
/// O(|Z|³) from the dataset's Gaussian sufficient statistics (see
/// [`crate::gram`]), flat in the number of rows. Shared by the Pearson,
/// Fisher-z and equivalence tests.
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] on constant input / residuals, too few
/// rows, or a dataset with no continuous columns or more than
/// [`crate::gram::MAX_GRAM_COLS`] of them; [`CiError::WrongColumnKind`] for
/// discrete columns; and [`CiError::Numeric`] for a rank-deficient
/// conditioning set.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z are the standard conditional-independence variable names from the contract"
)]
pub(crate) fn partial_correlation(
    data: &Dataset,
    x: usize,
    y: usize,
    z: &[usize],
) -> Result<(f64, usize), CiError> {
    let Some(gram) = data.gram() else {
        // The cache is the only implementation. Refusing loudly is better than
        // silently switching to a second, differently-conditioned algorithm.
        return Err(CiError::DegenerateData(format!(
            "continuous tests need between 1 and {} continuous columns and at \
             least one row; this dataset has {} columns and {} rows",
            crate::gram::MAX_GRAM_COLS,
            data.n_cols(),
            data.n_rows(),
        )));
    };
    partial_correlation_gram(gram, data.n_rows(), x, y, z)
}

/// Fast path: partial correlation from the Gram cache's Schur complements.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z, a, b, c, r are standard CI and linear-algebra variable names"
)]
fn partial_correlation_gram(
    gram: &GramCache,
    n: usize,
    x: usize,
    y: usize,
    z: &[usize],
) -> Result<(f64, usize), CiError> {
    let n_z = z.len();
    check_row_count(n, n_z)?;

    let (s_xx, s_yy, a, b, c) = gram.schur_xy_given_z(x, y, z)?;

    if z.is_empty() {
        // Mirror `pearson_r`'s constant-input error.
        if s_xx == 0.0 || s_yy == 0.0 {
            return Err(constant_input_error(if s_xx == 0.0 { "x" } else { "y" }));
        }
    } else {
        // The relative eps also catches fp-negative Schur complements and the
        // constant-input case.
        check_residual_variance("x", a, s_xx)?;
        check_residual_variance("y", b, s_yy)?;
    }

    let r = c / (a * b).sqrt();
    Ok((r, n - n_z - 2))
}

/// Two-tailed p-value for H₀: ρ = 0 from `r` and `dof` via the t-distribution.
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] if `|r| == 1` (t undefined) and
/// [`CiError::Numeric`] if the t-distribution construction fails.
pub(crate) fn correlation_p_value(r: f64, dof: usize) -> Result<f64, CiError> {
    if (1.0 - r * r) <= 0.0 {
        return Err(CiError::DegenerateData(
            "correlation is ±1; t-statistic is undefined".to_string(),
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let dof_f = dof as f64;
    let t = r * (dof_f / (1.0 - r * r)).sqrt();
    let dist = StudentsT::new(0.0, 1.0, dof_f)
        .map_err(|e| CiError::Numeric(format!("Student's t distribution: {e}")))?;
    Ok(2.0 * dist.sf(t.abs()))
}

impl CITest for PearsonCorrelation {
    fn test_impl(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult, CiError> {
        let (r, dof) = partial_correlation(data, x, y, z)?;
        let p_value = correlation_p_value(r, dof)?;
        Ok(CiResult {
            statistic: Some(r),
            p_value,
            dof: Some(dof),
            effect_size: Some(r.abs()),
        })
    }

    fn meta(&self) -> TestMeta {
        TestMeta {
            name: "pearson_correlation",
            data_types: &[DataType::Continuous],
            symmetric: true,
            rule: IndependenceRule::PValueGe,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ColumnKind;

    fn ds(cols: Vec<(&str, Vec<f64>)>) -> Dataset {
        Dataset::from_columns(
            cols.into_iter()
                .map(|(n, v)| (n.to_string(), ColumnKind::Continuous, v))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn unconditional_strong_correlation() {
        // Strongly (but not perfectly) correlated, so 1 - r^2 > 0.
        let data = ds(vec![
            ("x", vec![1., 2., 3., 4., 5.]),
            ("y", vec![2., 4.1, 5.9, 8.2, 9.8]),
        ]);
        let r = PearsonCorrelation::new().test(&data, 0, 1, &[]).unwrap();
        assert!(r.statistic.unwrap() > 0.9);
        assert_eq!(r.dof, Some(3));
        assert!(r.p_value < 0.05);
        assert!((r.effect_size.unwrap() - r.statistic.unwrap().abs()).abs() < 1e-12);
    }

    #[test]
    fn perfect_correlation_is_degenerate() {
        // r == 1 exactly -> t-statistic undefined; report a clear error.
        let data = ds(vec![
            ("x", vec![1., 2., 3., 4., 5.]),
            ("y", vec![2., 4., 6., 8., 10.]),
        ]);
        assert!(matches!(
            PearsonCorrelation::new().test(&data, 0, 1, &[]),
            Err(CiError::DegenerateData(_))
        ));
    }

    #[test]
    fn constant_input_is_degenerate_not_panic() {
        let data = ds(vec![
            ("x", vec![1., 1., 1., 1., 1.]),
            ("y", vec![2., 4., 6., 8., 10.]),
        ]);
        assert!(matches!(
            PearsonCorrelation::new().test(&data, 0, 1, &[]),
            Err(CiError::DegenerateData(_))
        ));
    }

    #[test]
    fn conditional_uses_intercept() {
        // y = 2*z + 5, x = 3*z + 1: after regressing out [1, z] residuals are ~0.
        let z = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let x: Vec<f64> = z.iter().map(|v| 3.0 * v + 1.0).collect();
        let y: Vec<f64> = z.iter().map(|v| 2.0 * v + 5.0).collect();
        let data = ds(vec![("x", x), ("y", y), ("z", z)]);
        // residuals are ~0 -> constant -> degenerate (exact collinear case).
        let res = PearsonCorrelation::new().test(&data, 0, 1, &[2]);
        assert!(matches!(res, Err(CiError::DegenerateData(_))));
    }

    #[test]
    fn dof_accounts_for_conditioning_set() {
        let z1 = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 1.1];
        let z2 = vec![1.1, 0.2, 0.7, 0.3, 0.9, 0.4, 0.8, 0.1, 0.6, 0.5];
        let x = vec![0.5, 0.4, 0.6, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2];
        let y = vec![0.2, 0.7, 0.1, 0.8, 0.3, 0.9, 0.4, 0.6, 0.5, 0.7];
        let data = ds(vec![("x", x), ("y", y), ("z1", z1), ("z2", z2)]);
        let r = PearsonCorrelation::new()
            .test(&data, 0, 1, &[2, 3])
            .unwrap();
        // n - |Z| - 2 = 10 - 2 - 2 = 6.
        assert_eq!(r.dof, Some(6));
    }

    #[test]
    fn wrong_kind_errors() {
        let data = Dataset::from_columns(vec![
            ("x".into(), ColumnKind::Discrete, vec![1., 2., 3.]),
            ("y".into(), ColumnKind::Continuous, vec![1., 2., 3.]),
        ])
        .unwrap();
        assert!(matches!(
            PearsonCorrelation::new().test(&data, 0, 1, &[]),
            Err(CiError::WrongColumnKind(_))
        ));
    }

    /// Deterministic pseudo-random doubles in (-1, 1) without a rand dep.
    #[allow(
        clippy::unreadable_literal,
        clippy::cast_precision_loss,
        reason = "LCG constants must be exact; precision loss is intentional for the RNG output"
    )]
    #[test]
    fn gram_path_degenerate_errors_match() {
        // Constant x, unconditional -> "input is constant".
        let data = ds(vec![
            ("x", vec![1., 1., 1., 1., 1.]),
            ("y", vec![2., 4., 6., 8., 10.]),
        ]);
        assert!(matches!(
            partial_correlation(&data, 0, 1, &[]),
            Err(CiError::DegenerateData(_))
        ));

        // Z perfectly explains x -> "residual is constant".
        let z = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let x: Vec<f64> = z.iter().map(|v| 3.0 * v + 1.0).collect();
        let y = vec![0.3, -0.1, 0.9, 0.2, -0.5, 0.7, 0.1, -0.2];
        let data = ds(vec![("x", x), ("y", y), ("z", z)]);
        assert!(matches!(
            partial_correlation(&data, 0, 1, &[2]),
            Err(CiError::DegenerateData(_))
        ));

        // Collinear Z (z2 = 2*z1) -> rank-deficient.
        let z1 = vec![1., 2., 3., 4., 5., 6.];
        let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v).collect();
        let x = vec![0.4, 0.1, 0.8, 0.2, 0.9, 0.3];
        let y = vec![0.2, 0.7, 0.1, 0.8, 0.3, 0.9];
        let data = ds(vec![("x", x), ("y", y), ("z1", z1), ("z2", z2)]);
        assert!(matches!(
            partial_correlation(&data, 0, 1, &[2, 3]),
            Err(CiError::Numeric(_))
        ));
    }

    #[test]
    fn too_many_or_no_continuous_columns_error_explicitly() {
        // A dataset with no continuous columns has no Gaussian sufficient
        // statistics, so there is nothing to compute from. It must say so
        // rather than dispatch to a second algorithm.
        let data = Dataset::from_columns(vec![
            (
                "a".into(),
                crate::dataset::ColumnKind::Discrete,
                vec![1., 2., 1., 2.],
            ),
            (
                "b".into(),
                crate::dataset::ColumnKind::Discrete,
                vec![1., 1., 2., 2.],
            ),
        ])
        .unwrap();
        let err = partial_correlation(&data, 0, 1, &[]).unwrap_err();
        assert!(matches!(err, CiError::DegenerateData(_)), "{err}");
        let message = err.to_string();
        assert!(
            message.contains("continuous columns"),
            "message should explain the requirement: {message}"
        );
    }
}
