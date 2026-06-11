//! Pearson chi-squared conditional-independence test (power divergence, λ = 1).

use crate::ci_tests::discrete_common::{discrete_meta, run_power_divergence};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, TestMeta};

/// The power-divergence parameter for the Pearson chi-squared statistic.
const LAMBDA: f64 = 1.0;

/// Pearson chi-squared test for discrete data (power divergence with λ = 1).
///
/// When `yates` is set, Yates' continuity correction is applied on 2×2
/// (sub-)tables, matching `scipy.stats.chi2_contingency(correction=True)` and
/// pgmpy. Cramér's V is reported as the effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChiSquared {
    /// Whether to apply Yates' continuity correction on 2×2 (sub-)tables.
    pub yates: bool,
}

impl ChiSquared {
    /// Construct the test with Yates' continuity correction enabled (the
    /// scipy/pgmpy default).
    #[must_use]
    pub fn new() -> Self {
        Self { yates: true }
    }
}

impl Default for ChiSquared {
    fn default() -> Self {
        Self::new()
    }
}

impl CITest for ChiSquared {
    fn test(&self, data: &Dataset, x: usize, y: usize, z: &[usize]) -> Result<CiResult, CiError> {
        run_power_divergence(data, x, y, z, LAMBDA, self.yates)
    }

    fn meta(&self) -> TestMeta {
        discrete_meta("chi_squared")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ColumnKind;
    use crate::strategy::{DataType, IndependenceRule};

    fn ds(cols: Vec<(&str, Vec<f64>)>) -> Dataset {
        Dataset::from_columns(
            cols.into_iter()
                .map(|(n, v)| (n.to_string(), ColumnKind::Discrete, v))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn unconditional_independent() {
        let data = ds(vec![
            ("x", vec![1., 1., 2., 2., 1., 1., 2., 2.]),
            ("y", vec![1., 2., 1., 2., 1., 2., 1., 2.]),
        ]);
        let r = ChiSquared::new().test(&data, 0, 1, &[]).unwrap();
        assert!(r.statistic.unwrap().abs() < 1e-9);
        assert_eq!(r.dof, Some(1));
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn unconditional_dependent_with_yates() {
        // [[4,0],[0,4]]: Yates makes stat = 4 * (1.5^2 / 2) = 4.5.
        let data = ds(vec![
            ("x", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
            ("y", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
        ]);
        let r = ChiSquared::new().test(&data, 0, 1, &[]).unwrap();
        assert!(
            (r.statistic.unwrap() - 4.5).abs() < 1e-9,
            "got {:?}",
            r.statistic
        );
        assert_eq!(r.dof, Some(1));
    }

    #[test]
    fn yates_off_changes_statistic() {
        // Same [[4,0],[0,4]] without Yates -> full chi-square = 8.
        let data = ds(vec![
            ("x", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
            ("y", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
        ]);
        let r = ChiSquared { yates: false }.test(&data, 0, 1, &[]).unwrap();
        assert!((r.statistic.unwrap() - 8.0).abs() < 1e-9, "got {:?}", r.statistic);
    }

    #[test]
    fn conditional_independent() {
        let data = ds(vec![
            ("x", vec![1., 1., 2., 2., 1., 1., 2., 2.]),
            ("y", vec![1., 2., 1., 2., 1., 2., 1., 2.]),
            ("z", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
        ]);
        let r = ChiSquared::new().test(&data, 0, 1, &[2]).unwrap();
        assert!(r.statistic.unwrap().abs() < 1e-9);
        assert_eq!(r.dof, Some(2));
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn wrong_column_kind_errors() {
        let data = Dataset::from_columns(vec![
            ("x".into(), ColumnKind::Continuous, vec![1., 2., 3.]),
            ("y".into(), ColumnKind::Discrete, vec![1., 2., 3.]),
        ])
        .unwrap();
        assert!(matches!(
            ChiSquared::new().test(&data, 0, 1, &[]),
            Err(CiError::WrongColumnKind(_))
        ));
    }

    #[test]
    fn meta_is_correct() {
        let m = ChiSquared::new().meta();
        assert_eq!(m.name, "chi_squared");
        assert_eq!(m.data_types, &[DataType::Discrete]);
        assert!(m.symmetric);
        assert_eq!(m.rule, IndependenceRule::PValueGe);
    }
}
