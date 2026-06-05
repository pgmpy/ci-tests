use crate::strategy::{CITest, CITestDataType, TestResult};
use crate::utils::power_divergence::power_divergence;
use ndarray::{Array1, Array2};

const LOG_LIKELIHOOD_LAMBDA: f64 = 0.0;

/// Log-likelihood ratio (G-test) conditional independence test (λ = 0).
///
/// Operates on discrete data only. Delegates to the power-divergence family
/// with λ = 0, which corresponds to the G-test / log-likelihood ratio statistic.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLikelihood {
    pub boolean: bool,
    pub significance_level: f64,
}

impl LogLikelihood {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64) -> Self {
        Self {
            boolean,
            significance_level,
        }
    }
}

impl CITest for LogLikelihood {
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
            LOG_LIKELIHOOD_LAMBDA,
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
        let t = LogLikelihood {
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
    fn cond_independent_data_accepted() {
        let t = LogLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!((stat).abs() < EPS, " got {stat}");
        assert!(p > 0.99);
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_dependent_data_rejected() {
        let t = LogLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 1., 2., 1., 2., 2., 2., 2., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!((stat - 5.822_063_320_647_374).abs() < EPS, "got {stat}");
        assert!((p - 0.015_826_368_796_540_195).abs() < EPS, "got {p}");
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_dependent_data_rejected() {
        let t = LogLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(
            (stat - 11.090_354_888_959_125).abs() < EPS,
            "for stat got {stat}"
        );
        assert!((p - 0.003_906_249_999_999_994).abs() < EPS, "for p got {p}");
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_bool_rejects_dependent() {
        let t = LogLikelihood {
            boolean: true,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 1., 2., 1., 2., 2., 2., 2., 2.];
        let empty = Array2::<f64>::zeros((0, 0));
        let r = t.run_test(x, y, empty).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }

    #[test]
    fn cond_bool_rejects_dependent() {
        let t = LogLikelihood {
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
