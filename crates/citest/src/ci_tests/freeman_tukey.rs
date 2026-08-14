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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    use crate::ci_tests::discrete_common::discrete_dataset as ds;

    #[test]
    fn dependent_table_pins_lambda_and_effect_size() {
        // 2x3 dependent table (n=27, dof=2, Yates inert) pinning the Freeman-Tukey
        // lambda wiring and Cramér's V effect size against scipy references.
        let x = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1.,
        ];
        let y = vec![
            0., 0., 0., 0., 0., 0., 1., 1., 2., 2., 2., 2., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
            2., 2., 2., 2., 2.,
        ];
        let data = ds(vec![("x", x), ("y", y)]);
        let r = FreemanTukey::new().test(&data, 0, 1, &[]).unwrap();
        assert_eq!(r.dof, Some(2));
        assert!((r.statistic.unwrap() - 3.868_242_083_0).abs() < 1e-6);
        assert!((r.effect_size.unwrap() - 0.378_507_893_3).abs() < 1e-6);
    }
}
