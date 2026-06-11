//! The data-bound conditional-independence test contract.
//!
//! Every test implements [`CITest`], operating on a shared [`Dataset`] and
//! referring to the variables under test (`x`, `y`) and the conditioning set
//! (`z`) by column index. Configuration lives on the test struct; the data
//! lives on the [`Dataset`]. The default [`CITest::is_independent`] turns a
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

/// A conditional-independence test bound to a [`Dataset`].
pub trait CITest: Send + Sync {
    /// Run the test for `x ⊥ y | z`, where `x`, `y` and the entries of `z` are
    /// column indices into `data`.
    ///
    /// # Errors
    ///
    /// Returns a [`CiError`] if the data is unsuitable (wrong column kind,
    /// degenerate input, dimension mismatch) or a numerical routine fails.
    fn test(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult, CiError>;

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
}
