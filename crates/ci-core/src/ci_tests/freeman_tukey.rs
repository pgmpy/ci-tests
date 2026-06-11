//! Freeman-Tukey conditional-independence test (power divergence, λ = −1/2).

use crate::ci_tests::discrete_common::{discrete_meta, run_power_divergence};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, TestMeta};

/// The power-divergence parameter for the Freeman-Tukey statistic.
const LAMBDA: f64 = -0.5;

/// Freeman-Tukey power-divergence test for discrete data (λ = −1/2).
///
/// When `yates` is set, Yates' continuity correction is applied on 2×2
/// (sub-)tables. Cramér's V is reported as the effect size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreemanTukey {
    /// Whether to apply Yates' continuity correction on 2×2 (sub-)tables.
    pub yates: bool,
}

impl FreemanTukey {
    /// Construct the test with Yates' continuity correction enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { yates: true }
    }
}

impl Default for FreemanTukey {
    fn default() -> Self {
        Self::new()
    }
}

impl CITest for FreemanTukey {
    fn test_impl(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult, CiError> {
        run_power_divergence(data, x, y, z, LAMBDA, self.yates)
    }

    fn meta(&self) -> TestMeta {
        discrete_meta("freeman_tukey")
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
    fn unconditional_independent_near_zero() {
        let data = ds(vec![
            ("x", vec![1., 1., 2., 2., 1., 1., 2., 2.]),
            ("y", vec![1., 2., 1., 2., 1., 2., 1., 2.]),
        ]);
        let r = FreemanTukey::new().test(&data, 0, 1, &[]).unwrap();
        assert!(r.statistic.unwrap().abs() < 1e-9);
        assert_eq!(r.dof, Some(1));
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn meta_is_correct() {
        let m = FreemanTukey::new().meta();
        assert_eq!(m.name, "freeman_tukey");
        assert_eq!(m.data_types, &[DataType::Discrete]);
        assert!(m.symmetric);
        assert_eq!(m.rule, IndependenceRule::PValueGe);
    }
}
