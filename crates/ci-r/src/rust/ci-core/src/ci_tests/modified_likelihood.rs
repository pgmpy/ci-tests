//! Modified log-likelihood (Neyman) conditional-independence test
//! (power divergence, λ = −1).

use crate::ci_tests::discrete_common::{discrete_meta, run_power_divergence};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, TestMeta};

/// The power-divergence parameter for the modified log-likelihood statistic.
const LAMBDA: f64 = -1.0;

/// Modified log-likelihood (Neyman) power-divergence test for discrete data
/// (λ = −1). The per-cell statistic is `2 · Σ E·ln(E/O)`; a structural zero
/// (`O = 0` with `E > 0`) drives the statistic to `+∞` and hence the p-value to
/// `0`.
///
/// When `yates` is set, Yates' continuity correction is applied on 2×2
/// (sub-)tables. Cramér's V is reported as the effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModifiedLikelihood {
    /// Whether to apply Yates' continuity correction on 2×2 (sub-)tables.
    pub yates: bool,
}

impl ModifiedLikelihood {
    /// Construct the test with Yates' continuity correction enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { yates: true }
    }
}

impl Default for ModifiedLikelihood {
    fn default() -> Self {
        Self::new()
    }
}

impl CITest for ModifiedLikelihood {
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
        discrete_meta("modified_likelihood")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_tests::discrete_common::discrete_dataset as ds;

    #[test]
    fn structural_zero_gives_p_zero() {
        // A 3×3 table (dof != 1, so no Yates) with a structural zero in an
        // active row/column -> statistic +∞ -> p = 0.
        let data = ds(vec![
            ("x", vec![0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]),
            ("y", vec![0., 0., 1., 1., 0., 0., 1., 1., 2., 2., 2., 2.]),
        ]);
        let r = ModifiedLikelihood::new().test(&data, 0, 1, &[]).unwrap();
        assert!(r.statistic.unwrap().is_infinite());
        // p is exactly 0 for an infinite statistic.
        assert!(r.p_value.total_cmp(&0.0).is_eq());
    }
}
