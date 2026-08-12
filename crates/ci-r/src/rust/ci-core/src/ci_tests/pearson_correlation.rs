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

/// Clipping bound for `rho` before the Fisher z-transform: `[-1 + EPS, 1 - EPS]`.
/// Matches the reference's `np.clip(rho, -0.999999, 0.999999)`. Shared by the
/// Fisher-z and equivalence tests.
pub(crate) const RHO_CLIP_EPS: f64 = 1e-6;

/// Pearson correlation test. With a non-empty conditioning set it computes the
/// partial correlation by regressing X and Y on `[1, Z]` (intercept included)
/// and correlating the residuals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PearsonCorrelation;

impl PearsonCorrelation {
    /// Construct the test (no configuration in this slice).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Sum of squared deviations from the mean, `Σ (v − v̄)²`.
fn sum_sq_deviations(v: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|&vi| (vi - mean) * (vi - mean)).sum()
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

/// Pearson correlation coefficient of two equal-length slices.
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] if either input has zero variance.
fn pearson_r(x: &[f64], y: &[f64]) -> Result<f64, CiError> {
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut covariance = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - x_mean;
        let dy = yi - y_mean;
        var_x += dx * dx;
        var_y += dy * dy;
        covariance += dx * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return Err(constant_input_error(if var_x == 0.0 { "x" } else { "y" }));
    }

    Ok(covariance / (var_x * var_y).sqrt())
}

/// Ordinary-least-squares residuals of `target` regressed on the
/// `rows × cols` design matrix `design` (stored row-major, `rows >= cols`),
/// computed via Householder QR.
///
/// Returns the length-`rows` residual vector `target − design · β̂`, where `β̂`
/// minimizes `‖design · β − target‖₂`. The design's first column is the
/// intercept in our use, so `cols ≤ |Z| + 1` is small and the QR is
/// well-conditioned.
///
/// # Errors
///
/// Returns [`CiError::Numeric`] if the design is rank-deficient (a zero pivot on
/// the diagonal of `R`), which would make the residuals ill-defined.
fn ols_residuals(
    design: &[f64],
    target: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<f64>, CiError> {
    let rank_deficient =
        || CiError::Numeric("rank-deficient design matrix in partial correlation".to_string());

    // Work on mutable copies: `mat` is triangularized in place, `rhs` is the
    // transformed right-hand side. Row-major index: mat[i * cols + j].
    let mut mat = design.to_vec();
    let mut rhs = target.to_vec();

    // Householder QR: zero out below the diagonal column by column.
    for k in 0..cols {
        // 2-norm of the sub-column mat[k..rows, k].
        let mut norm_sq = 0.0;
        for i in k..rows {
            let value = mat[i * cols + k];
            norm_sq += value * value;
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            return Err(rank_deficient());
        }
        // Householder reflector w = col − alpha·e1, alpha = −sign(pivot)·‖col‖,
        // where `pivot` is the on-diagonal entry of the column being reduced.
        let pivot = mat[k * cols + k];
        let alpha = if pivot >= 0.0 { -norm } else { norm };
        let mut reflector = vec![0.0; rows - k];
        reflector[0] = pivot - alpha;
        for i in (k + 1)..rows {
            reflector[i - k] = mat[i * cols + k];
        }
        let reflector_norm_sq: f64 = reflector.iter().map(|w| w * w).sum();
        if reflector_norm_sq == 0.0 {
            // Column already in upper-triangular form; nothing to reflect.
            continue;
        }

        // Apply H = I − 2·w·wᵀ / (wᵀw) to the trailing columns of `mat`.
        for j in k..cols {
            let mut dot = 0.0;
            for i in k..rows {
                dot += reflector[i - k] * mat[i * cols + j];
            }
            let factor = 2.0 * dot / reflector_norm_sq;
            for i in k..rows {
                mat[i * cols + j] -= factor * reflector[i - k];
            }
        }
        // Apply the same reflection to the right-hand side.
        let mut dot = 0.0;
        for i in k..rows {
            dot += reflector[i - k] * rhs[i];
        }
        let factor = 2.0 * dot / reflector_norm_sq;
        for i in k..rows {
            rhs[i] -= factor * reflector[i - k];
        }
    }

    // Back-substitute R·β = rhs[..cols] (R is the upper-left cols×cols of `mat`).
    let mut beta = vec![0.0; cols];
    for i in (0..cols).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..cols {
            acc -= mat[i * cols + j] * beta[j];
        }
        let diag = mat[i * cols + i];
        if diag == 0.0 {
            return Err(rank_deficient());
        }
        beta[i] = acc / diag;
    }

    // Residuals target − design·β using the original design matrix.
    let mut residual = target.to_vec();
    for i in 0..rows {
        let mut fitted = 0.0;
        for j in 0..cols {
            fitted += design[i * cols + j] * beta[j];
        }
        residual[i] -= fitted;
    }
    Ok(residual)
}

/// Compute the (partial) correlation `r` and its degrees of freedom for
/// `x ⊥ y | z`.
///
/// With empty `z` this is the plain Pearson r with `dof = n - 2`. With `z` it
/// is the partial correlation given Z with `dof = n - |Z| - 2`. Dispatches to
/// the O(|Z|³) sufficient-statistic fast path when the dataset's Gram cache is
/// available (see [`crate::gram`]), and to the O(n·|Z|²) residual-regression
/// path otherwise; both are algebraically identical for finite inputs (see
/// [`crate::gram`] for the non-finite caveat). Shared by the Pearson,
/// Fisher-z and equivalence tests.
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] on constant input / residuals or too
/// few rows, [`CiError::WrongColumnKind`] for discrete columns, and
/// [`CiError::Numeric`] for a rank-deficient conditioning set.
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
    match data.gram() {
        Some(gram) => partial_correlation_gram(gram, data.n_rows(), x, y, z),
        None => partial_correlation_residual(data, x, y, z),
    }
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
        // Mirror the residual-constant check (the relative eps also catches
        // fp-negative Schur complements and the constant-input case).
        check_residual_variance("x", a, s_xx)?;
        check_residual_variance("y", b, s_yy)?;
    }

    let r = c / (a * b).sqrt();
    Ok((r, n - n_z - 2))
}

/// O(n·|Z|²) fallback: compute partial correlation via Householder QR residual
/// regression on `[1, Z]`.
///
/// With empty `z` this is the plain Pearson r with `dof = n - 2`. With `z` it is
/// the correlation of the residuals after regressing X and Y on `[1, Z]`, with
/// `dof = n - |Z| - 2`. Shared with the equivalence test.
///
/// This is the O(n·|Z|²) fallback used when the dataset's Gram cache is
/// unavailable; the normal entry point is [`partial_correlation`].
///
/// # Errors
///
/// Returns [`CiError::DegenerateData`] on constant input or when there are too
/// few rows, and [`CiError::Numeric`] if the least-squares solve fails.
#[allow(
    clippy::many_single_char_names,
    reason = "x, y, z are the standard conditional-independence variable names from the contract"
)]
pub(crate) fn partial_correlation_residual(
    data: &Dataset,
    x: usize,
    y: usize,
    z: &[usize],
) -> Result<(f64, usize), CiError> {
    let x_vals = data.continuous(x)?;
    let y_vals = data.continuous(y)?;
    let n = x_vals.len();
    check_row_count(n, z.len())?;

    if z.is_empty() {
        let r = pearson_r(x_vals, y_vals)?;
        return Ok((r, n - 2));
    }

    let n_z = z.len();

    // Design matrix [1, Z] (n × (n_z + 1)), stored row-major.
    let n_cols = n_z + 1;
    let mut design = vec![0.0; n * n_cols];
    for row in 0..n {
        design[row * n_cols] = 1.0;
    }
    for (j, &zi) in z.iter().enumerate() {
        let z_vals = data.continuous(zi)?;
        for row in 0..n {
            design[row * n_cols + j + 1] = z_vals[row];
        }
    }

    // Residuals of X and Y after regressing each on [1, Z] (intercept included),
    // then correlate the residuals.
    let residual_x = ols_residuals(&design, x_vals, n, n_cols)?;
    let residual_y = ols_residuals(&design, y_vals, n, n_cols)?;

    // If Z explains essentially all of X's (or Y's) variation, the residuals are
    // pure floating-point noise and the partial correlation is undefined. Detect
    // this relative to the *original* variable's deviation energy (the SVD path
    // truncated such residuals to exactly zero; QR leaves ~1e-15 noise). The
    // threshold sits far below any genuine residual ratio in the fixture.
    check_residual_variance(
        "x",
        sum_sq_deviations(&residual_x),
        sum_sq_deviations(x_vals),
    )?;
    check_residual_variance(
        "y",
        sum_sq_deviations(&residual_y),
        sum_sq_deviations(y_vals),
    )?;

    let r = pearson_r(&residual_x, &residual_y)?;
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
    fn lcg_f64(seed: &mut u64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn gram_fast_path_matches_residual_path() {
        let mut seed = 0xFEED_u64;
        let n = 120;
        let z1 = lcg_f64(&mut seed, n);
        let z2 = lcg_f64(&mut seed, n);
        let noise_x = lcg_f64(&mut seed, n);
        let noise_y = lcg_f64(&mut seed, n);
        let x: Vec<f64> = (0..n)
            .map(|i| 1.3 * z1[i] - 0.7 * z2[i] + 0.5 * noise_x[i])
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| -0.4 * z1[i] + 0.9 * z2[i] + 0.5 * noise_y[i])
            .collect();
        let data = ds(vec![("x", x), ("y", y), ("z1", z1), ("z2", z2)]);

        for z in [vec![], vec![2], vec![2, 3]] {
            let (r_fast, dof_fast) = partial_correlation(&data, 0, 1, &z).unwrap();
            let (r_slow, dof_slow) = partial_correlation_residual(&data, 0, 1, &z).unwrap();
            assert_eq!(dof_fast, dof_slow);
            assert!(
                (r_fast - r_slow).abs() < 1e-12,
                "|Z|={}: fast {r_fast} vs slow {r_slow}",
                z.len()
            );
        }
    }

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
    fn residual_path_collinear_z_is_numeric() {
        // Directly exercise the non-gram residual path's rank-deficient outcome:
        // collinear Z (z2 = 2*z1) makes the OLS design rank-deficient.
        let z1 = vec![1., 2., 3., 4., 5., 6.];
        let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v).collect();
        let x = vec![0.4, 0.1, 0.8, 0.2, 0.9, 0.3];
        let y = vec![0.2, 0.7, 0.1, 0.8, 0.3, 0.9];
        let data = ds(vec![("x", x), ("y", y), ("z1", z1), ("z2", z2)]);
        assert!(matches!(
            partial_correlation_residual(&data, 0, 1, &[2, 3]),
            Err(CiError::Numeric(_))
        ));
    }
}
