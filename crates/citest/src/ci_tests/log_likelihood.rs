//! Log-likelihood (G-test) conditional-independence test (power divergence, λ = 0).

use crate::ci_tests::discrete_common::{discrete_meta, run_power_divergence};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, TestMeta};

/// The power-divergence parameter for the log-likelihood (G-test) statistic.
const LAMBDA: f64 = 0.0;

/// Log-likelihood ratio (G-test) for discrete data (power divergence with λ = 0).
///
/// The per-cell statistic is `2 · Σ O·ln(O/E)`. When `yates` is set, Yates'
/// continuity correction is applied on 2×2 (sub-)tables. Cramér's V is reported
/// as the effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogLikelihood {
    /// Whether to apply Yates' continuity correction on 2×2 (sub-)tables.
    pub yates: bool,
}

impl LogLikelihood {
    /// Construct the test with Yates' continuity correction enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { yates: true }
    }
}

impl Default for LogLikelihood {
    fn default() -> Self {
        Self::new()
    }
}

impl CITest for LogLikelihood {
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
        discrete_meta("log_likelihood")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_tests::discrete_common::discrete_dataset as ds;

    #[test]
    fn dependent_table_pins_lambda_and_effect_size() {
        // 2x3 dependent table (n=27, dof=2, Yates inert) pinning the log-likelihood
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
        let r = LogLikelihood::new().test(&data, 0, 1, &[]).unwrap();
        assert_eq!(r.dof, Some(2));
        assert!((r.statistic.unwrap() - 3.738_650_145_2).abs() < 1e-6);
        assert!((r.effect_size.unwrap() - 0.372_113_590_0).abs() < 1e-6);
    }
}
