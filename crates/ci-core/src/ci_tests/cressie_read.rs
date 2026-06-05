use crate::strategy::{CITest, CITestDataType, TestResult};
use crate::utils::power_divergence::power_divergence;
use ndarray::{Array1, Array2};

const CRESSIE_READ_LAMBDA: f64 = 2.0 / 3.0;

/// Cressie–Read conditional independence test (λ = 2/3).
///
/// Operates on discrete data only. Delegates to the power-divergence family
/// with λ = 2/3, the Cressie–Read statistic.
#[derive(Debug, Clone, PartialEq)]
pub struct CressieRead {
    pub boolean: bool,
    pub significance_level: f64,
}

impl CressieRead {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64) -> Self {
        Self {
            boolean,
            significance_level,
        }
    }
}

impl CITest for CressieRead {
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
            CRESSIE_READ_LAMBDA,
        )
    }
    fn data_types(&self) -> &'static [CITestDataType] {
        &[CITestDataType::Discrete]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::EPS;
    use ndarray::array;

    fn unwrap_correlated(result: &TestResult) -> (f64, f64, usize) {
        match result {
            TestResult::Statistic(a, b, c) => (*a, *b, *c),
            _ => panic!("expected Correlated2"),
        }
    }

    fn unwrap_boolean(result: &TestResult) -> bool {
        match result {
            TestResult::Boolean(b) => *b,
            _ => panic!("expected Boolean"),
        }
    }

    #[test]
    fn unconditional_independent_data_is_not_rejected() {
        let test = CressieRead {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let empty_z = Array2::<f64>::zeros((0, 0));

        let (p_value, statistic, dof) = unwrap_correlated(
            &test
                .run_test(x.clone(), y.clone(), empty_z.clone())
                .unwrap(),
        );
        assert!(
            statistic.abs() < EPS,
            "expected statistic ~0, got {statistic}"
        );
        assert!(p_value > 0.99, "expected p ~1, got {p_value}");
        assert_eq!(dof, 1);

        let independent = unwrap_boolean(
            &CressieRead {
                boolean: true,
                significance_level: 0.05,
            }
            .run_test(x, y, empty_z)
            .unwrap(),
        );
        assert!(independent, "expected fail-to-reject (independent=true)");
    }

    #[test]
    fn unconditional_dependent_data_is_rejected() {
        let test = CressieRead {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 1., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 2., 2., 2., 2.];
        let empty_z = Array2::<f64>::zeros((0, 0));

        let (p_value, statistic, _dof) = unwrap_correlated(
            &test
                .run_test(x.clone(), y.clone(), empty_z.clone())
                .unwrap(),
        );
        assert!(statistic > 5.0, "expected large statistic, got {statistic}");
        assert!(
            p_value < test.significance_level,
            "expected p < {}, got {p_value}",
            test.significance_level
        );

        let independent = unwrap_boolean(
            &CressieRead {
                boolean: true,
                significance_level: 0.05,
            }
            .run_test(x, y, empty_z)
            .unwrap(),
        );
        assert!(!independent, "expected reject (independent=false)");
    }

    #[test]
    fn conditional_independent_per_group() {
        let test = CressieRead {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let z = Array2::from_shape_vec((8, 1), vec![0., 0., 0., 0., 1., 1., 1., 1.]).unwrap();

        let (p_value, statistic, dof) =
            unwrap_correlated(&test.run_test(x.clone(), y.clone(), z.clone()).unwrap());
        assert!(
            statistic.abs() < EPS,
            "expected statistic ~0, got {statistic}"
        );
        assert!(p_value > 0.99, "expected p ~1, got {p_value}");
        assert_eq!(dof, 2);

        let independent = unwrap_boolean(
            &CressieRead {
                boolean: true,
                significance_level: 0.05,
            }
            .run_test(x, y, z)
            .unwrap(),
        );
        assert!(independent);
    }

    #[test]
    fn conditional_dependent_per_group() {
        let test = CressieRead {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let z = Array2::from_shape_vec((8, 1), vec![0., 0., 0., 0., 1., 1., 1., 1.]).unwrap();

        let (p_value, statistic, _dof) = unwrap_correlated(&test.run_test(x, y, z).unwrap());
        assert!(statistic > 5.0, "expected large statistic, got {statistic}");
        assert!(
            p_value < test.significance_level,
            "expected p < {}, got {p_value}",
            test.significance_level
        );
    }
}
