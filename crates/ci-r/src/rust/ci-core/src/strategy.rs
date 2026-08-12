//! The data-bound conditional-independence test contract.
//!
//! Every test implements [`CITest`], operating on a shared [`Dataset`] and
//! referring to the variables under test (`x`, `y`) and the conditioning set
//! (`z`) by column index. Configuration lives on the test struct; the data
//! lives on the [`Dataset`]. The provided [`CITest::test`] validates the query
//! via [`validate_query`] before delegating to the required
//! [`CITest::test_impl`]. The default [`CITest::is_independent`] turns a
//! numeric [`CiResult`] into a boolean using the test's [`IndependenceRule`].

use crate::dataset::Dataset;
use crate::error::CiError;

/// The numeric outcome of a conditional-independence test.
#[derive(Debug, Clone, PartialEq)]
pub struct CiResult {
    /// The test statistic (e.g. chi-squared statistic, Pearson r, Fisher-z),
    /// when the test defines one.
    pub statistic: Option<f64>,
    /// The p-value. Interpretation depends on the test's [`IndependenceRule`].
    pub p_value: f64,
    /// Degrees of freedom, when applicable.
    pub dof: Option<usize>,
    /// An effect-size summary (e.g. Cramér's V, |r|, |rho|), when applicable.
    pub effect_size: Option<f64>,
}

/// How a p-value is turned into an independence decision at level `alpha`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndependenceRule {
    /// Standard null-of-independence tests: independent when `p >= alpha`.
    PValueGe,
    /// Equivalence / TOST tests: independent when `p < alpha`.
    PValueLt,
}

impl IndependenceRule {
    /// Whether independence holds for the given `p` at significance `alpha`.
    #[must_use]
    pub fn holds(self, p: f64, alpha: f64) -> bool {
        match self {
            IndependenceRule::PValueGe => p >= alpha,
            IndependenceRule::PValueLt => p < alpha,
        }
    }
}

/// The kind of data a test consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Categorical data.
    Discrete,
    /// Numeric data.
    Continuous,
}

/// Static description of a test: its name, supported data types, whether it is
/// symmetric in `x`/`y`, and its independence rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestMeta {
    /// Stable identifier (e.g. `"chi_squared"`).
    pub name: &'static str,
    /// Data types the test supports.
    pub data_types: &'static [DataType],
    /// Whether swapping `x` and `y` leaves the result unchanged.
    pub symmetric: bool,
    /// How the p-value maps to an independence decision.
    pub rule: IndependenceRule,
}

/// Validate a `(x, y, z)` query against `data` before running a test.
///
/// Checks, in order: all column indices are in range
/// ([`CiError::UnknownColumn`]); `x != y`; `x`/`y` do not appear in `z`; `z`
/// has no duplicates (all [`CiError::InvalidQuery`]). Called by the provided
/// [`CITest::test`] so every test is validated uniformly.
///
/// # Errors
///
/// Returns the first violated rule as described above.
pub fn validate_query(data: &Dataset, x: usize, y: usize, z: &[usize]) -> Result<(), CiError> {
    let n_cols = data.n_cols();
    for idx in [x, y].into_iter().chain(z.iter().copied()) {
        if idx >= n_cols {
            return Err(CiError::UnknownColumn(format!("column index {idx}")));
        }
    }
    if x == y {
        return Err(CiError::InvalidQuery(format!(
            "x and y must be different columns (both are column index {x})"
        )));
    }
    for (label, idx) in [("x", x), ("y", y)] {
        if z.contains(&idx) {
            return Err(CiError::InvalidQuery(format!(
                "{label} (column index {idx}) must not appear in the conditioning set z"
            )));
        }
    }
    for i in 0..z.len() {
        if z[i + 1..].contains(&z[i]) {
            return Err(CiError::InvalidQuery(format!(
                "conditioning set z contains column index {} more than once",
                z[i]
            )));
        }
    }
    Ok(())
}

/// A conditional-independence test bound to a [`Dataset`].
pub trait CITest: Send + Sync {
    /// Test-specific computation for `x ⊥ y | z`. Implementations may assume
    /// the query has already been validated by [`CITest::test`]; call sites
    /// should use [`CITest::test`], not this method.
    ///
    /// # Errors
    ///
    /// Returns a [`CiError`] if the data is unsuitable (wrong column kind,
    /// degenerate input, dimension mismatch) or a numerical routine fails.
    fn test_impl(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult, CiError>;

    /// Run the test for `x ⊥ y | z` after validating the query
    /// (see [`validate_query`]).
    ///
    /// Implementors should not override this method; implement
    /// [`CITest::test_impl`] instead, so the uniform validation is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::UnknownColumn`] / [`CiError::InvalidQuery`] for a
    /// malformed query, or any error from [`CITest::test_impl`].
    fn test(&self, data: &Dataset, x: usize, y: usize, z: &[usize]) -> Result<CiResult, CiError> {
        validate_query(data, x, y, z)?;
        self.test_impl(data, x, y, z)
    }

    /// Static metadata describing this test.
    fn meta(&self) -> TestMeta;

    /// Convenience wrapper returning an independence decision at level `alpha`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`CITest::test`].
    fn is_independent(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
        alpha: f64,
    ) -> Result<bool, CiError> {
        if !alpha.is_finite() {
            return Err(CiError::InvalidQuery(format!(
                "significance level must be finite, got {alpha}"
            )));
        }
        Ok(self
            .meta()
            .rule
            .holds(self.test(data, x, y, z)?.p_value, alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_ge() {
        assert!(IndependenceRule::PValueGe.holds(0.5, 0.05));
        assert!(IndependenceRule::PValueGe.holds(0.05, 0.05));
        assert!(!IndependenceRule::PValueGe.holds(0.01, 0.05));
    }

    #[test]
    fn rule_lt() {
        assert!(IndependenceRule::PValueLt.holds(0.01, 0.05));
        assert!(!IndependenceRule::PValueLt.holds(0.05, 0.05));
        assert!(!IndependenceRule::PValueLt.holds(0.5, 0.05));
    }

    use crate::ci_tests::ChiSquared;
    use crate::dataset::{ColumnKind, Dataset};

    fn two_col_data() -> Dataset {
        Dataset::from_columns(vec![
            ("a".into(), ColumnKind::Discrete, vec![1., 2., 1., 2.]),
            ("b".into(), ColumnKind::Discrete, vec![1., 1., 2., 2.]),
            ("c".into(), ColumnKind::Discrete, vec![1., 2., 2., 1.]),
        ])
        .unwrap()
    }

    #[test]
    fn rejects_x_equals_y() {
        let data = two_col_data();
        let err = ChiSquared::new().test(&data, 0, 0, &[]).unwrap_err();
        assert!(
            matches!(err, crate::error::CiError::InvalidQuery(_)),
            "{err}"
        );
    }

    #[test]
    fn rejects_x_or_y_in_z() {
        let data = two_col_data();
        assert!(matches!(
            ChiSquared::new().test(&data, 0, 1, &[0]),
            Err(crate::error::CiError::InvalidQuery(_))
        ));
        assert!(matches!(
            ChiSquared::new().test(&data, 0, 1, &[1]),
            Err(crate::error::CiError::InvalidQuery(_))
        ));
    }

    #[test]
    fn rejects_duplicate_z() {
        let data = two_col_data();
        assert!(matches!(
            ChiSquared::new().test(&data, 0, 1, &[2, 2]),
            Err(crate::error::CiError::InvalidQuery(_))
        ));
    }

    #[test]
    fn rejects_out_of_range_indices() {
        let data = two_col_data();
        assert!(matches!(
            ChiSquared::new().test(&data, 9, 1, &[]),
            Err(crate::error::CiError::UnknownColumn(_))
        ));
        assert!(matches!(
            ChiSquared::new().test(&data, 0, 1, &[9]),
            Err(crate::error::CiError::UnknownColumn(_))
        ));
        // is_independent goes through the same validation.
        assert!(matches!(
            ChiSquared::new().is_independent(&data, 0, 0, &[], 0.05),
            Err(crate::error::CiError::InvalidQuery(_))
        ));
    }

    #[test]
    fn rejects_non_finite_significance_levels() {
        let data = two_col_data();
        let test = ChiSquared::new();

        for alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = test.is_independent(&data, 0, 1, &[], alpha).unwrap_err();
            assert!(
                matches!(
                    err,
                    crate::error::CiError::InvalidQuery(ref message)
                        if message.contains("significance level must be finite")
                ),
                "alpha={alpha}: {err}"
            );
        }
    }
}
