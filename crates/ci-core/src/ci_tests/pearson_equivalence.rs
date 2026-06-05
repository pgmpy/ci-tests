use crate::ci_tests::PearsonCorrelation;
use crate::strategy::{CITest, CITestDataType, TestResult};
use crate::utils::EPS;

const FISHER_Z_DOF_OFFSET: usize = 3;
use anyhow::bail;
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};

/// Pearson equivalence (TOST) conditional independence test.
///
/// Uses the Two One-Sided Tests (TOST) framework with Fisher's z-transformation
/// to test whether the partial correlation is small enough to declare independence.
///
/// **Note**: the p-value convention is inverted relative to the other tests. A *low*
/// p-value (below `significance_level`) means the correlation is within `delta_threshold`
/// of zero and the null of dependence is rejected — i.e. the variables are declared
/// independent.
#[derive(Debug, Clone, PartialEq)]
pub struct PearsonEquivalence {
    pub boolean: bool,
    pub significance_level: f64,
    pub delta_threshold: f64,
}

impl PearsonEquivalence {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64, delta_threshold: f64) -> Self {
        Self {
            boolean,
            significance_level,
            delta_threshold,
        }
    }
}

impl CITest for PearsonEquivalence {
    fn run_test(
        &self,
        x_values: Array1<f64>,
        y_values: Array1<f64>,
        z: Array2<f64>,
    ) -> anyhow::Result<TestResult> {
        let n = x_values.len();
        let s = z.axis_iter(Axis(1)).len();

        let pearsonr = PearsonCorrelation {
            boolean: false,
            significance_level: self.significance_level,
        }
        .run_test(x_values, y_values, z);
        let statistic = match pearsonr {
            Ok(TestResult::PValue(_, statistic)) => statistic,
            Ok(_) => 0.0,
            Err(e) => return Err(e),
        };
        let rho = if statistic <= -1.0 {
            -1.0 + EPS
        } else if statistic >= 1.0 {
            1.0 - EPS
        } else {
            statistic
        };

        let coefficient = rho.atanh();
        let z_delta = self.delta_threshold.atanh();

        #[allow(
            clippy::cast_precision_loss,
            reason = "array length and number of variables most likely won't exceed 2^53"
        )]
        let argument = (n - s - FISHER_Z_DOF_OFFSET) as f64;
        let std_error_factor = if argument >= 0.0 {
            argument.sqrt()
        } else {
            bail!("The length of the data should be at least 3 greater than the number of conditional variables");
        };

        let normal = Normal::new(0.0, 1.0).unwrap();

        let z_score_lower = std_error_factor * (coefficient + z_delta);
        let z_score_upper = std_error_factor * (coefficient - z_delta);

        let p_value_lower = 1.0 - normal.cdf(z_score_lower);
        let p_value_upper = normal.cdf(z_score_upper);

        let p_value = if p_value_lower > p_value_upper {
            p_value_lower
        } else {
            p_value_upper
        };

        Ok(wrap_result(
            self.boolean,
            p_value,
            coefficient,
            self.significance_level,
        ))
    }

    fn data_types(&self) -> &'static [CITestDataType] {
        &[CITestDataType::Continuous]
    }
}

#[must_use]
pub fn wrap_result(
    boolean: bool,
    p_value: f64,
    coefficient: f64,
    significance_level: f64,
) -> TestResult {
    if boolean {
        return TestResult::Boolean(p_value < significance_level);
    }
    TestResult::PValue(p_value, coefficient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, stack, Array1, Array2, Axis};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    const SIGNIFICANCE_LEVEL: f64 = 0.05;
    const DELTA_THRESHOLD: f64 = 0.1;
    // Specific tests imported from pgmpy fail with default epsilon
    const PGMPY_EPS: f64 = 1e-8;

    const N: usize = 1000;

    #[test]
    fn basic_test() {
        let x_vals = array![1.0, 2.0, 3.0, 4.0];
        let y_vals = array![1.0, 1.0, 2.0, 2.0];
        let empty_z = array![[]];

        let test = PearsonEquivalence {
            boolean: false,
            significance_level: 0.05,
            delta_threshold: DELTA_THRESHOLD,
        };
        let result = test.run_test(x_vals, y_vals, empty_z);

        let (p_value, statistic) = match result {
            Ok(TestResult::PValue(a, b)) => (a, b),
            _ => (0.0, 0.0),
        };

        // values taken from pgmpy
        assert!((p_value - 0.910_412_594_569_001_1).abs() < PGMPY_EPS);
        assert!((statistic - 1.443_635_475_178_810_7).abs() < PGMPY_EPS);
    }

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(40)
    }

    fn gen_normal(n: usize, mean: f64, std_dev: f64, rng: &mut SmallRng) -> Array1<f64> {
        let dist = Normal::new(mean, std_dev).unwrap();
        Array1::from_vec((0..n).map(|_| dist.sample(rng)).collect())
    }

    fn empty_array() -> Array2<f64> {
        Array2::zeros((0, 0))
    }

    fn pearson() -> PearsonEquivalence {
        PearsonEquivalence {
            boolean: false,
            significance_level: 0.05,
            delta_threshold: DELTA_THRESHOLD,
        }
    }

    fn pearson_boolean() -> PearsonEquivalence {
        PearsonEquivalence {
            boolean: true,
            significance_level: 0.05,
            delta_threshold: DELTA_THRESHOLD,
        }
    }

    #[test]
    fn unconditional_independent_data_is_not_rejected() {
        let mut rng = seeded_rng();
        let x = gen_normal(N, 0.0, 1.0, &mut rng);
        let y = gen_normal(N, 0.0, 1.0, &mut rng);

        let result = pearson().run_test(x, y, empty_array()).unwrap();
        match result {
            TestResult::PValue(p_value, coefficient) => {
                assert!(
                    p_value <= SIGNIFICANCE_LEVEL,
                    "p_value {p_value} should be <= 0.05 for independent data"
                );
                assert!(
                    coefficient.abs() < DELTA_THRESHOLD,
                    "coefficient {coefficient} should be near 0 for independent data"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }

    #[test]
    fn unconditional_boolean_accepts_independent() {
        let mut rng = seeded_rng();
        let x = gen_normal(N, 0.0, 1.0, &mut rng);
        let y = gen_normal(N, 0.0, 1.0, &mut rng);

        let result = pearson_boolean().run_test(x, y, empty_array()).unwrap();
        match result {
            TestResult::Boolean(independent) => {
                assert!(independent, "Independent data should return true");
            }
            _ => panic!("Expected TestResult::Boolean"),
        }
    }

    #[test]
    fn unconditional_dependent_data_is_rejected() {
        let mut rng = seeded_rng();
        let x = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise = gen_normal(N, 0.0, 0.1, &mut rng);
        let y = &x * 3.0 + &noise;

        let result = pearson().run_test(x, y, empty_array()).unwrap();
        match result {
            TestResult::PValue(p_value, coefficient) => {
                assert!(
                    p_value >= SIGNIFICANCE_LEVEL,
                    "p_value {p_value} should be >= 0.05 for correlated data"
                );
                assert!(
                    coefficient.abs() > 0.9,
                    "coefficient {coefficient} should be high for correlated data"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }

    #[test]
    fn unconditional_boolean_rejects_dependent() {
        let mut rng = seeded_rng();
        let x = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise = gen_normal(N, 0.0, 0.1, &mut rng);
        let y = &x * 3.0 + &noise;

        let result = pearson_boolean().run_test(x, y, empty_array()).unwrap();
        match result {
            TestResult::Boolean(independent) => {
                assert!(!independent, "Correlated data should return false");
            }
            _ => panic!("Expected TestResult::Boolean"),
        }
    }

    // Z is a confounder: X = 3*Z + noise, Y = 2*Z + noise. After conditioning, residuals are independent.
    #[test]
    fn conditional_independent_data_is_not_rejected() {
        let mut rng = seeded_rng();
        let z = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise_x = gen_normal(N, 0.0, 0.1, &mut rng);
        let noise_y = gen_normal(N, 0.0, 0.1, &mut rng);
        let x = &z * 3.0 + &noise_x;
        let y = &z * 2.0 + &noise_y;
        let array = z.insert_axis(Axis(1));

        let result = pearson().run_test(x, y, array).unwrap();
        match result {
            TestResult::PValue(p_value, coefficient) => {
                assert!(
                    p_value <= SIGNIFICANCE_LEVEL,
                    "p_value {p_value} should be <= 0.05 after conditioning"
                );
                assert!(
                    coefficient.abs() < DELTA_THRESHOLD,
                    "coefficient {coefficient} should be near 0 after conditioning"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }

    #[test]
    fn conditional_boolean_accepts_independent() {
        let mut rng = seeded_rng();
        let z = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise_x = gen_normal(N, 0.0, 0.1, &mut rng);
        let noise_y = gen_normal(N, 0.0, 0.1, &mut rng);
        let x = &z * 3.0 + &noise_x;
        let y = &z * 2.0 + &noise_y;
        let array = z.insert_axis(Axis(1));

        let result = pearson_boolean().run_test(x, y, array).unwrap();
        match result {
            TestResult::Boolean(independent) => {
                assert!(
                    independent,
                    "Conditionally independent data should return true"
                );
            }
            _ => panic!("Expected TestResult::Boolean"),
        }
    }

    // Z = 2*X + 2*Y + noise is a collider; conditioning on it induces dependence between X and Y.
    #[test]
    fn conditional_dependent_data_is_rejected() {
        let mut rng = seeded_rng();
        let x = gen_normal(N, 0.0, 1.0, &mut rng);
        let y = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise = gen_normal(N, 0.0, 0.1, &mut rng);
        let z = &x * 2.0 + &y * 2.0 + &noise;
        let array = z.insert_axis(Axis(1));

        let result = pearson().run_test(x, y, array).unwrap();
        match result {
            TestResult::PValue(p_value, coefficient) => {
                assert!(
                    p_value >= SIGNIFICANCE_LEVEL,
                    "p_value {p_value} should be >= 0.05 for v-structure"
                );
                assert!(
                    coefficient.abs() > 0.9,
                    "coefficient {coefficient} should be high for v-structure"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }

    #[test]
    fn conditional_boolean_rejects_dependent() {
        let mut rng = seeded_rng();
        let x = gen_normal(N, 0.0, 1.0, &mut rng);
        let y = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise = gen_normal(N, 0.0, 0.1, &mut rng);
        let z = &x * 2.0 + &y * 2.0 + &noise;
        let array = z.insert_axis(Axis(1));

        let result = pearson_boolean().run_test(x, y, array).unwrap();
        match result {
            TestResult::Boolean(independent) => {
                assert!(
                    !independent,
                    "V-structure conditioned on collider should return false"
                );
            }
            _ => panic!("Expected TestResult::Boolean"),
        }
    }

    #[test]
    fn conditional_multiple_vars_independent_is_not_rejected() {
        let mut rng = seeded_rng();
        let z_1 = gen_normal(N, 0.0, 1.0, &mut rng);
        let z_2 = gen_normal(N, 0.0, 1.0, &mut rng);
        let z_3 = gen_normal(N, 0.0, 1.0, &mut rng);
        let noise_x = gen_normal(N, 0.0, 0.1, &mut rng);
        let noise_y = gen_normal(N, 0.0, 0.1, &mut rng);
        let x = 0.5 * &z_1 + 0.5 * &z_2 + 0.5 * &z_3 + &noise_x;
        let y = 0.5 * &z_1 + 0.5 * &z_2 + 0.5 * &z_3 + &noise_y;

        let array = stack(Axis(1), &[z_1.view(), z_2.view(), z_3.view()]).unwrap();

        let result = pearson().run_test(x, y, array).unwrap();
        match result {
            TestResult::PValue(p_value, coefficient) => {
                assert!(
                    p_value < SIGNIFICANCE_LEVEL,
                    "p_value {p_value} should be < 0.05 after conditioning on all confounders"
                );
                assert!(
                    coefficient.abs() <= DELTA_THRESHOLD,
                    "coefficient {coefficient} should be near 0 after conditioning on all confounders"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }
}
