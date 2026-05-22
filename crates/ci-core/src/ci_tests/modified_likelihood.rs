use crate::strategy::{CITest, CITestDataType, TestResult};
use crate::utils::power_divergence::power_divergence;
use ndarray::{Array1, Array2};

const MODIFIED_LIKELIHOOD_LAMBDA: f64 = -1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ModifiedLikelihood {
    pub boolean: bool,
    pub significance_level: f64,
}

impl ModifiedLikelihood {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64) -> Self {
        Self {
            boolean,
            significance_level,
        }
    }
}

impl CITest for ModifiedLikelihood {
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
            MODIFIED_LIKELIHOOD_LAMBDA,
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
    use ndarray::{array, Array2};

    fn unwrap_correlated(r: &TestResult) -> (f64, f64, usize) {
        match r {
            TestResult::Statistic(a, b, c) => (*a, *b, *c),
            _ => panic!("expected Correlated2"),
        }
    }

    #[test]
    fn uncond_independent_data_accepted() {
        let t = ModifiedLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!(stat.abs() < 1e-9);
        assert!(p > 0.99);
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_independent_data_accepted() {
        let t = ModifiedLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, z).unwrap());

        // Even with lambda = -1, perfectly independent data results in 0
        assert!(stat.abs() < 1e-9, " got stat {stat}");
        assert!(p > 0.99, " got p {p}");
        assert_eq!(dof, 2);
    }

    // scipy: power_divergence([[5,1],[1,5]], lambda_=-1) -> stat=7.053439978825427
    #[test]
    fn uncond_dependent_data_rejected() {
        let t = ModifiedLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.];
        let y = array![1., 1., 1., 1., 1., 2., 1., 2., 2., 2., 2., 2.];
        let empty = Array2::<f64>::zeros((0, 0));

        let (p, stat, dof) = unwrap_correlated(&t.run_test(x, y, empty).unwrap());
        assert!((stat - 7.053_439_978_825_427).abs() < 1e-9, "got {stat}");
        assert!((p - 0.007_911_317_670_556_329).abs() < 1e-12, "got {p}");
        assert_eq!(dof, 1);
    }

    #[test]
    fn cond_dependent_data_rejected() {
        let t = ModifiedLikelihood {
            boolean: false,
            significance_level: 0.05,
        };
        let x = array![1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2.];
        let y = array![1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2., 1.];
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
            (stat - 1.413_396_427_876_601_6).abs() < 1e-9,
            "got stat {stat}"
        );
        assert!((p - 0.493_270_184_272_571_97).abs() < 1e-12, "got p {p}");
        assert_eq!(dof, 2);
    }

    #[test]
    fn uncond_bool_rejects_dependent() {
        let t = ModifiedLikelihood {
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
    fn cond_bool_rejects_independent() {
        let t = ModifiedLikelihood {
            boolean: true,
            significance_level: 0.05,
        };
        let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let y = array![1., 1., 2., 2., 1., 1., 2., 2.];
        let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];
        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(true)));
    }
}
