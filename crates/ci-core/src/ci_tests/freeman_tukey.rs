use crate::strategy::{CITest, CITestDataType, TestResult};
use crate::utils::power_divergence::power_divergence;
use ndarray::{Array1, Array2};

const FREEMAN_TUKEY_LAMBDA: f64 = -1.0 / 2.0;

/// Freeman–Tukey conditional independence test (λ = −1/2).
///
/// Operates on discrete data only. Delegates to the power-divergence family
/// with λ = −1/2, the Freeman–Tukey statistic.
#[derive(Debug, Clone, PartialEq)]
pub struct FreemanTukey {
    pub boolean: bool,
    pub significance_level: f64,
}

impl FreemanTukey {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64) -> Self {
        Self {
            boolean,
            significance_level,
        }
    }
}

impl CITest for FreemanTukey {
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
            FREEMAN_TUKEY_LAMBDA,
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
    fn uncond_independent_data_not_rejected() {
        let t = FreemanTukey {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!(stat.abs() < EPS);
        assert!(p > 0.99);
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_independent_not_rejected() {
        let t = FreemanTukey {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.],];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(stat.abs() < EPS);
        assert!(p > 0.99);
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_dependent_rejected() {
        let t = FreemanTukey {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 1., 2., 1., 2., 2., 2., 2., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!((stat - 6.319_453_539_579_289).abs() < EPS, "got {stat}");
        assert!((p - 0.011_942_042_564_347_121).abs() < EPS, "got {p}");
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_dependent_rejected() {
        let t = FreemanTukey {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 2., 1., 1., 2., 2., 1., 2.];
        let y = array![1., 2., 1., 2., 2., 1., 1., 2., 1., 2., 2., 1.];
        let z = array![
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.]
        ];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(
            (stat - 1.382_538_273_265_069_5).abs() < EPS,
            "got stat {stat}"
        );
        assert!((p - 0.500_939_904_278_208_8).abs() < EPS, "got p value {p}");
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_boolean_accepts_independent() {
        let t = FreemanTukey {
            boolean: true,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let empty = Array2::<f64>::zeros((0, 0));
        let r = t.run_test(x, y, empty).unwrap();
        assert!(matches!(r, TestResult::Boolean(true)));
    }

    #[test]
    fn cond_boolean_rejects_dependent() {
        let t = FreemanTukey {
            boolean: true,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2.];
        let y = array![1., 1., 2., 2., 2., 2., 1., 1., 2., 2., 2., 2.];
        let z = array![
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.],
            [2.]
        ];

        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }
}
