use crate::ci_tests::PearsonCorrelation;
use crate::strategy::{CITest, CITestDataType, TestResult};
use anyhow::bail;
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};

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
            -1.0 + 1e-12
        } else if statistic >= 1.0 {
            1.0 - 1e-12
        } else {
            statistic
        };

        let coefficient = rho.atanh();
        let z_delta = self.delta_threshold.atanh();

        #[allow(
            clippy::cast_precision_loss,
            reason = "array length and number of variables most likely won't exceed 2^53"
        )]
        let argument = (n - s - 3) as f64;
        let std_error_factor = if argument >= 0.0 {
            argument.sqrt()
        } else {
            bail!("The length of the data should be at least 3 greater than the number of conditional variables");
        };

        let z_score_lower = std_error_factor * (coefficient + z_delta);
        let p_value_lower = 1.0 - Normal::new(0.0, 1.0).unwrap().cdf(z_score_lower);

        let z_score_upper = std_error_factor * (coefficient - z_delta);
        let p_value_upper = Normal::new(0.0, 1.0).unwrap().cdf(z_score_upper);

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

    const N: usize = 1000;

    #[test]
    fn basic_test() {
        let x_vals = array![1.0, 2.0, 3.0, 4.0];
        let y_vals = array![1.0, 1.0, 2.0, 2.0];
        let empty_z = array![[]];

        let test = PearsonEquivalence {
            boolean: false,
            significance_level: 0.05,
            delta_threshold: 0.1,
        };
        let result = test.run_test(x_vals, y_vals, empty_z);

        let (p_value, statistic) = match result {
            Ok(TestResult::PValue(a, b)) => (a, b),
            _ => (0.0, 0.0),
        };

        // values taken from pgmpy
        assert!((p_value - 0.910_412_594_569_001_1).abs() < 1e-8);
        assert!((statistic - 1.443_635_475_178_810_7).abs() < 1e-8);
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
            delta_threshold: 0.1,
        }
    }

    fn pearson_boolean() -> PearsonEquivalence {
        PearsonEquivalence {
            boolean: true,
            significance_level: 0.05,
            delta_threshold: 0.1,
        }
    }

    // --- 1. Empty array + independent X, Y + boolean=false ---
    // X and Y are independently generated, no conditioning variables.
    // Expected: low p_value (<= 0.05), low |coefficient| (< 0.1)
    // Note: p_value is opposite to other tests
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
                    coefficient.abs() < 0.1,
                    "coefficient {coefficient} should be near 0 for independent data"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }

    // --- 2. Empty array + independent X, Y + boolean=true ---
    // Expected: true (variables are independent)
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

    // --- 3. Empty array + correlated X, Y + boolean=false ---
    // Y = 3*X + small noise, so they are strongly correlated.
    // Expected: high p_value (>= 0.05), high |coefficient| (> 0.9)
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

    // --- 4. Empty array + correlated X, Y + boolean=true ---
    // Expected: false (variables are NOT independent)
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

    // --- 5. Non-empty array + conditionally independent + boolean=false ---
    // Z is a confounder: X = 3*Z + noise, Y = 2*Z + noise.
    // After conditioning on Z, residuals should be independent.
    // Expected: low p_value (< 0.05), low |coefficient| (< 0.1)
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
                    coefficient.abs() < 0.1,
                    "coefficient {coefficient} should be near 0 after conditioning"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }

    // --- 6. Non-empty array + conditionally independent + boolean=true ---
    // Expected: true (conditionally independent given Z)
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

    // --- 7. Non-empty array + conditionally dependent (v-structure) + boolean=false ---
    // X and Y are independent, but Z = 2*X + 2*Y + noise (collider).
    // Conditioning on Z makes X and Y dependent.
    // Expected: high p_value (>= 0.05), high |coefficient|
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

    // --- 8. Non-empty array + conditionally dependent (v-structure) + boolean=true ---
    // Expected: false (NOT independent after conditioning on collider)
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

    // --- 9. Multiple conditioning variables + conditionally independent + boolean=false ---
    // Z1, Z2, Z3 are confounders: X and Y both depend on them.
    // After conditioning on all three, residuals should be independent.
    // Expected: low p_value, low |coefficient|
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
                    coefficient.abs() <= 0.1,
                    "coefficient {coefficient} should be near 0 after conditioning on all confounders"
                );
            }
            _ => panic!("Expected TestResult::PValue"),
        }
    }
}
