use crate::strategy::{CITest, CITestDataType, TestResult};
use crate::utils::power_divergence::power_divergence;

use ndarray::{Array1, Array2};

const CHI_SQUARED_LAMBDA: f64 = 1.0;

/// Pearson chi-squared conditional independence test (λ = 1).
///
/// Operates on discrete data only. Delegates to the power-divergence family
/// with λ = 1, which is the classical chi-squared statistic.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquared {
    pub boolean: bool,
    pub significance_level: f64,
}

impl ChiSquared {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64) -> Self {
        Self {
            boolean,
            significance_level,
        }
    }
}

impl CITest for ChiSquared {
    fn run_test(
        &self,
        x_values: Array1<f64>,
        y_values: Array1<f64>,
        z: Array2<f64>,
    ) -> anyhow::Result<TestResult> {
        power_divergence(
            &x_values,
            &y_values,
            &z,
            self.boolean,
            self.significance_level,
            CHI_SQUARED_LAMBDA,
        )
    }

    fn data_types(&self) -> &'static [CITestDataType] {
        &[CITestDataType::Discrete]
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::utils::EPS;
    use ndarray::{array, Array2};

    fn unwrap_correlated(r: &TestResult) -> (f64, f64, usize) {
        match r {
            TestResult::Statistic(a, b, c) => (*a, *b, *c),
            _ => panic!("expected Correlated2"),
        }
    }

    #[test]
    fn uncond_independent_data_accepted() {
        let t = ChiSquared {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!(stat.abs() < EPS, "stat should be ~0, got {stat}");
        assert!(p > 0.99);
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_independent_data_accepted() {
        let t = ChiSquared {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(stat.abs() < EPS, "stat should be ~0, got {stat}");
        assert!(p > 0.99);
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_dependent_data_rejected() {
        let t = ChiSquared {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 1., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 2., 2., 2., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!((stat - 8.0).abs() < EPS, "got {stat}");
        assert!((p - 0.004_677_734_981_047_276).abs() < EPS, "got {p}");
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_dependent_data_rejected() {
        let t = ChiSquared {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!((stat - 8.0).abs() < EPS, "stat {stat} should be larger");
        assert!(
            (p - 0.018_315_638_888_734_193).abs() < EPS,
            "rejected p value {p}"
        );
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_boolean_mode() {
        let t = ChiSquared {
            boolean: true,
            significance_level: 0.05,
        };
        let empty = Array2::<f64>::zeros((0, 0));
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let r = t.run_test(x, y, empty.clone()).unwrap();
        assert!(matches!(r, TestResult::Boolean(true)));

        let x = array![1., 1., 1., 1., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 2., 2., 2., 2.];
        let r = t.run_test(x, y, empty).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }

    #[test]
    fn cond_boolean_mode() {
        let t = ChiSquared {
            boolean: true,
            significance_level: 0.05,
        };
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(true)));

        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];
        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }
}
