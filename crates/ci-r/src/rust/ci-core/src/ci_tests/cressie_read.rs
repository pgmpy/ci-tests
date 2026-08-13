//! Cressie-Read conditional-independence test (power divergence, λ = 2/3).

use crate::ci_tests::discrete_common::{discrete_meta, run_power_divergence};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, TestMeta};

/// The power-divergence parameter for the Cressie-Read statistic.
const LAMBDA: f64 = 2.0 / 3.0;

/// Cressie-Read power-divergence test for discrete data (λ = 2/3), the
/// recommended compromise between Pearson and the likelihood-ratio statistics.
///
/// When `yates` is set, Yates' continuity correction is applied on 2×2
/// (sub-)tables. Cramér's V is reported as the effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CressieRead {
    /// Whether to apply Yates' continuity correction on 2×2 (sub-)tables.
    pub yates: bool,
}

impl CressieRead {
    /// Construct the test with Yates' continuity correction enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { yates: true }
    }
}

impl Default for CressieRead {
    fn default() -> Self {
        Self::new()
    }
}

impl CITest for CressieRead {
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
        discrete_meta("cressie_read")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_tests::discrete_common::discrete_dataset as ds;

    #[test]
    fn balanced_independent_table_has_finite_effect_size() {
        // Cressie-Read on a balanced independent 2x2 ([[4,4],[4,4]]): O == E, so
        // the power-divergence statistic rounds to a tiny *negative* value. The
        // Cramér's V effect size must still be a finite 0.0, not NaN (regression
        // test for the cramers_v radicand clamp).
        let data = ds(vec![
            (
                "x",
                vec![
                    0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
                ],
            ),
            (
                "y",
                vec![
                    0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.,
                ],
            ),
        ]);
        let r = CressieRead::new().test(&data, 0, 1, &[]).unwrap();
        let effect = r.effect_size.expect("effect size should be present");
        assert!(
            effect.is_finite(),
            "effect size must be finite, got {effect}"
        );
        assert!(
            effect < 1e-9,
            "balanced independent table -> ~0 effect, got {effect}"
        );
    }

    #[test]
    fn dependent_table_pins_lambda_and_effect_size() {
        // 2x3 dependent table (n=27, dof=2, Yates inert) pinning the Cressie-Read
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
        let r = CressieRead::new().test(&data, 0, 1, &[]).unwrap();
        assert_eq!(r.dof, Some(2));
        assert!((r.statistic.unwrap() - 3.629_764_546_5).abs() < 1e-6);
        assert!((r.effect_size.unwrap() - 0.366_654_774_9).abs() < 1e-6);
    }
}
