use crate::{
    strategy::{CITest, CITestDataType, TestResult},
    utils::EPS,
};
use anyhow::ensure;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView1};
use statrs::distribution::{ContinuousCDF, StudentsT};

const SVD_TOLERANCE: f64 = 1e-10;
const MIN_SAMPLE_SIZE: usize = 3;

/// Pearson correlation conditional independence test.
///
/// Should be used only on continuous data. When the conditioning set is non-empty,
/// uses linear regression to compute residuals and tests the Pearson correlation
/// on those residuals (partial correlation).
///
/// # References
///
/// - [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
/// - [Partial correlation using linear regression](https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression)
#[derive(Debug, Clone, PartialEq)]
pub struct PearsonCorrelation {
    pub boolean: bool,
    pub significance_level: f64,
}

impl PearsonCorrelation {
    #[must_use]
    pub fn new(boolean: bool, significance_level: f64) -> Self {
        Self {
            boolean,
            significance_level,
        }
    }
}

impl CITest for PearsonCorrelation {
    fn run_test(
        &self,
        x_values: Array1<f64>,
        y_values: Array1<f64>,
        z: Array2<f64>,
    ) -> anyhow::Result<TestResult> {
        if z.is_empty() {
            let (coefficient, p_value) = pearsonr(&x_values.view(), &y_values.view())?;
            Ok(wrap_result(
                self.boolean,
                p_value,
                coefficient,
                self.significance_level,
            ))
        } else {
            let z_na = DMatrix::from_row_iterator(z.nrows(), z.ncols(), z.iter().copied());
            let x_na = DVector::from_iterator(x_values.len(), x_values.iter().copied());
            let y_na = DVector::from_iterator(y_values.len(), y_values.iter().copied());

            let svd = z_na.svd(true, true);
            let x_coefficient = svd
                .solve(&x_na, SVD_TOLERANCE)
                .map_err(|e| anyhow::anyhow!("least squares failed for x: {e}"))?;
            let y_coefficient = svd
                .solve(&y_na, SVD_TOLERANCE)
                .map_err(|e| anyhow::anyhow!("least squares failed for y: {e}"))?;

            let x_coef_nd = Array1::from_vec(x_coefficient.iter().copied().collect());
            let y_coef_nd = Array1::from_vec(y_coefficient.iter().copied().collect());

            let residual_x = x_values - z.dot(&x_coef_nd);
            let residual_y = y_values - z.dot(&y_coef_nd);

            let (coefficient, p_value) = pearsonr(&residual_x.view(), &residual_y.view())?;
            Ok(wrap_result(
                self.boolean,
                p_value,
                coefficient,
                self.significance_level,
            ))
        }
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
        return TestResult::Boolean(p_value >= significance_level);
    }
    TestResult::PValue(p_value, coefficient)
}

/// Compute the Pearson correlation coefficient and its two-tailed p-value.
///
/// Tests H₀: ρ = 0 using the t-distribution with n − 2 degrees of freedom.
/// Returns `(coefficient, p_value)`.
///
/// # Errors
///
/// Returns an error if the input has fewer than 3 elements (degrees of freedom < 1).
fn pearsonr(x_values: &ArrayView1<f64>, y_values: &ArrayView1<f64>) -> anyhow::Result<(f64, f64)> {
    let n = x_values.len();
    ensure!(
        x_values.len() == y_values.len() && x_values.len() >= MIN_SAMPLE_SIZE,
        "pearsonr requires equal-length inputs with n >= 3"
    );

    #[allow(
        clippy::cast_precision_loss,
        reason = "array length most likely won't exceed 2^53"
    )]
    let number_of_elements = n as f64;

    let x_mean = x_values.sum() / number_of_elements;
    let y_mean = y_values.sum() / number_of_elements;

    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    let mut sum_coproduct = 0.0;

    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        let dx = x - x_mean;
        let dy = y - y_mean;

        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
        sum_coproduct += dx * dy;
    }

    // If one of the datasets is constant, pearson coefficient is undefined.
    if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
        let array_name = if sum_sq_x == 0.0 { "x" } else { "y" };
        panic!("Array {array_name} is constant, so the pearson coëfficient is undefined.");
    }

    // Calculate correlation directly
    let mut coefficient = sum_coproduct / (sum_sq_x * sum_sq_y).sqrt();

    // Floating-point math can sometimes drift slightly outside (-1.0, 1.0), so then we clamp.
    // By adding/subtracting EPS we prevent divide by 0 errors.
    coefficient = coefficient.clamp(-1.0 + EPS, 1.0 - EPS);

    let t_statistic =
        coefficient * (number_of_elements - 2.0).sqrt() / (1.0 - coefficient.powi(2)).sqrt();

    let t_distribution = StudentsT::new(0.0, 1.0, number_of_elements - 2.0)?;
    let p_value = 2.0 * t_distribution.sf(t_statistic.abs());

    Ok((coefficient, p_value))
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::utils::EPS;
    use ndarray::{array, Array1, Array2};

    const SIGNIFICANCE_LEVEL: f64 = 0.05;

    fn unwrap_correlated(r: &TestResult) -> (f64, f64) {
        match r {
            TestResult::PValue(p, coef) => (*p, *coef),
            _ => panic!("expected TestResult::PValue"),
        }
    }

    #[test]
    fn uncond_independent_data_accepted() {
        let t = PearsonCorrelation {
            boolean: false,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![2., 4., 1., 5., 3., 8., 6., 7., 9., 10.];
        let y = array![5., 3., 7., 2., 8., 1., 9., 4., 6., 10.];

        let (p, coef) = unwrap_correlated(&t.run_test(x, y, Array2::zeros((0, 0))).unwrap());
        assert!(p > SIGNIFICANCE_LEVEL, "got {p}");
        assert!(
            coef.abs() < 0.1,
            "coef={coef} should be near 0 for uncorrelated data"
        );
    }

    #[test]
    fn uncond_boolean_mode() {
        let t = PearsonCorrelation {
            boolean: true,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        // independent -> true
        let x = array![2., 4., 1., 5., 3., 8., 6., 7., 9., 10.];
        let y = array![5., 3., 7., 2., 8., 1., 9., 4., 6., 10.];
        let r = t.run_test(x, y, Array2::zeros((0, 0))).unwrap();
        assert!(matches!(r, TestResult::Boolean(true)));

        // dependent -> false
        let x = array![1., 2., 3., 4., 5.];
        let y = array![2., 4., 6., 8., 10.];
        let r = t.run_test(x, y, Array2::zeros((0, 0))).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }

    #[test]
    fn uncond_dependent_data_rejected() {
        let t = PearsonCorrelation {
            boolean: false,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![1., 2., 3., 4., 5.];
        let y = array![2., 4., 6., 8., 10.];

        let (p, coef) = unwrap_correlated(&t.run_test(x, y, Array2::zeros((0, 0))).unwrap());
        assert!(p < SIGNIFICANCE_LEVEL, "got {p}");
        assert!(
            coef.abs() > 0.9,
            "coef={coef} should be high for perfectly correlated data"
        );
    }

    // Z is a confounder: X = 3*Z + noise, Y = 2*Z + noise. After conditioning, residuals are independent.
    #[test]
    fn cond_independent_data_accepted() {
        let t = PearsonCorrelation {
            boolean: false,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![
            2.019_608, 1.039_216, 4.058_824, 3.078_431, 6.098_039, 5.117_647, 8.137_255, 7.156_863
        ];
        let y = array![
            2.059_406, 3.059_406, 2.138_614, 3.138_614, 6.217_822, 7.217_822, 6.297_030, 7.297_030
        ];
        let z = array![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.]];
        let (p, coef) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(p > SIGNIFICANCE_LEVEL, "got {p}");
        assert!(
            coef.abs() < 0.1,
            "coef={coef} should be near 0 after conditioning"
        );
    }

    #[test]
    fn cond_boolean_mode() {
        // accepted
        let t = PearsonCorrelation {
            boolean: true,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![
            2.019_608, 1.039_216, 4.058_824, 3.078_431, 6.098_039, 5.117_647, 8.137_255, 7.156_863
        ];
        let y = array![
            2.059_406, 3.059_406, 2.138_614, 3.138_614, 6.217_822, 7.217_822, 6.297_030, 7.297_030
        ];
        let z = array![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.]];
        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(true)));

        // rejected
        let x = array![1., 2., 3., 4., 5., 6., 7., 8.];
        let y = array![8., 7., 6., 5., 4., 3., 2., 1.];
        let z = array![[4.5], [4.5], [4.5], [4.5], [4.5], [4.5], [4.5], [4.5]];
        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }

    // Z = 2*X + 2*Y + noise is a collider; conditioning on it induces dependence between X and Y.
    #[test]
    fn cond_dependent_data_rejected() {
        let t = PearsonCorrelation {
            boolean: false,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![1., 2., 3., 4., 5., 6., 7., 8.];
        let y = array![8., 7., 6., 5., 4., 3., 2., 1.];
        let z = array![[9.], [9.], [9.], [9.], [9.], [9.], [9.], [9.]];

        let (p, coef) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(p < SIGNIFICANCE_LEVEL, "got {p}");
        assert!(
            coef.abs() > 0.9,
            "coef={coef} should be high for collider structure"
        );
    }

    #[test]
    fn cond_bool_rejects_dependent() {
        let t = PearsonCorrelation {
            boolean: true,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![1., 2., 3., 4., 5., 6., 7., 8.];
        let y = array![8., 7., 6., 5., 4., 3., 2., 1.];
        let z = array![[9.], [9.], [9.], [9.], [9.], [9.], [9.], [9.]];

        let r = t.run_test(x, y, z).unwrap();
        assert!(matches!(r, TestResult::Boolean(false)));
    }
    #[test]
    fn cond_multiple_vars_independent_accepted() {
        let t = PearsonCorrelation {
            boolean: false,
            significance_level: SIGNIFICANCE_LEVEL,
        };
        let x = array![2.5, 2.5, 2.5, 4.0, 4.0, 4.0, 4.0, 5.5];
        let y = array![2.4, 2.4, 0.8, 2.8, 5.2, 5.2, 3.6, 5.6];
        let z = array![
            [1., 0., 1.],
            [1., 1., 0.],
            [2., 0., 0.],
            [2., 1., 1.],
            [3., 0., 1.],
            [3., 1., 0.],
            [4., 0., 0.],
            [4., 1., 1.],
        ];

        let (p, coef) = unwrap_correlated(&t.run_test(x, y, z).unwrap());
        assert!(p > SIGNIFICANCE_LEVEL, "got {p}");
        assert!(
            coef.abs() < 0.1,
            "coef={coef} should be near 0 after conditioning on all confounders"
        );
    }

    #[test]
    fn pearsonr_errors_on_empty_input() {
        let x: Array1<f64> = Array1::zeros(0);
        let y: Array1<f64> = Array1::zeros(0);
        assert!(pearsonr(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn pearsonr_errors_on_too_few_elements() {
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![3.0, 4.0]);
        assert!(pearsonr(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn pearsonr_errors_on_mismatched_lengths() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);
        assert!(pearsonr(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn pearsonr_succeeds_with_minimum_input() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let (coef, p_value) = pearsonr(&x.view(), &y.view()).unwrap();
        assert!((coef - 1.0).abs() < EPS, "perfect positive correlation");
        assert!(p_value < SIGNIFICANCE_LEVEL, "got {p_value}");
    }
}
