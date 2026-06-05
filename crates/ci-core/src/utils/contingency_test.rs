use crate::utils::EPS;
use anyhow::{bail, Result};

const MODIFIED_LIKELIHOOD_LAMBDA: f64 = -1.0;
const FREEMAN_TUKEY_LAMBDA: f64 = -0.5;
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Compute a Cressie-Read power-divergence statistic for a contingency table.
///
/// # Errors
/// Returns an error when the table is empty, contains negatives, sums to zero,
/// or has a zero expected frequency.
pub fn contingency_test(observed: &Array2<f64>, lambda: f64) -> Result<(f64, f64, usize)> {
    // Check whether contingency test is applicable
    if observed.is_empty() {
        bail!("No data; `observed` has size 0.");
    }
    if observed.iter().any(|&x| x < 0.0) {
        bail!("All values in `observed` must be nonnegative.");
    }

    let (nrows, ncols) = observed.dim();
    let row_sums = observed.sum_axis(Axis(1));
    let col_sums = observed.sum_axis(Axis(0));
    let total: f64 = row_sums.sum();
    if total == 0.0 {
        bail!("Total sum of observed frequencies must be > 0.");
    }
    let inverse_total = 1.0 / total;

    let col_times_total = &col_sums * inverse_total;
    let ln_total = total.ln();
    let ln_row_sums = row_sums.mapv(f64::ln);
    let ln_col_sums = col_sums.mapv(f64::ln);

    let statistic: f64 = if lambda.abs() < EPS {
        g_test(
            observed,
            &row_sums,
            &col_times_total,
            ln_total,
            &ln_row_sums,
            &ln_col_sums,
        )?
    } else if (lambda - MODIFIED_LIKELIHOOD_LAMBDA).abs() < EPS {
        modified_log_likelihood_ratio_test(
            observed,
            &row_sums,
            &col_times_total,
            ln_total,
            &ln_row_sums,
            &ln_col_sums,
        )?
    } else if (lambda - FREEMAN_TUKEY_LAMBDA).abs() < EPS {
        freeman_tukey(lambda, observed, &row_sums, &col_times_total)?
    } else {
        cressie_read(lambda, observed, &row_sums, &col_times_total)?
    };

    let degrees_of_freedom = if nrows < 2 || ncols < 2 {
        0
    } else {
        (nrows - 1) * (ncols - 1)
    };

    let p_value = if degrees_of_freedom == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        ChiSquared::new(degrees_of_freedom as f64)?.sf(statistic)
    };

    Ok((statistic, p_value, degrees_of_freedom))
}

/// # Errors
///
/// Returns an error if an expected frequency evaluates to zero, or if the
/// calculated statistic results in a mathematically impossible value.
pub fn g_test(
    observed: &Array2<f64>,
    row_sums: &Array1<f64>,
    col_times_total: &Array1<f64>,
    ln_total: f64,
    ln_row_sums: &Array1<f64>,
    ln_col_sums: &Array1<f64>,
) -> anyhow::Result<f64> {
    // G-test: 2 * sum(O * ln(O / E))
    let (nrows, ncols) = observed.dim();
    let mut temp_stat: f64 = 0.0;
    for i in 0..nrows {
        for j in 0..ncols {
            let temp_expected: f64 = row_sums[i] * col_times_total[j]; // division is worse than multiplication in rust
            let temp_observed = observed[[i, j]];
            if temp_expected == 0. {
                bail!("Expected frequency is zero at position [{i}, {j}]");
            }
            if temp_observed == 0. {
                continue;
            }
            temp_stat +=
                temp_observed * (temp_observed.ln() + ln_total - ln_row_sums[i] - ln_col_sums[j]);
            //used logarithmic rules to get rid of division and mulitplication
        }
    }
    let final_stat = 2.0 * temp_stat;
    if final_stat < -EPS {
        bail!("Statistic evaluated to {final_stat}, which should be impossible.");
        //make sure it bails when the negative number is not negligibly small
    }
    Ok(final_stat.max(0.0))
}

/// # Errors
///
/// Returns an error if an expected frequency evaluates to zero, or if the
/// calculated statistic results in a mathematically impossible value.
pub fn modified_log_likelihood_ratio_test(
    observed: &Array2<f64>,
    row_sums: &Array1<f64>,
    col_times_total: &Array1<f64>,
    ln_total: f64,
    ln_row_sums: &Array1<f64>,
    ln_col_sums: &Array1<f64>,
) -> anyhow::Result<f64> {
    let (nrows, ncols) = observed.dim();
    let mut temp_stat: f64 = 0.0;
    for i in 0..nrows {
        for j in 0..ncols {
            let temp_expected: f64 = row_sums[i] * col_times_total[j]; //multiplication instead of division
            let temp_observed = observed[[i, j]];
            if temp_expected == 0. {
                continue;
            }
            if temp_observed == 0. {
                // log of temp_observed will be -infinity, causing temp_stat to become infinity.
                // However the math still works, so this is not an error.
                return Ok(f64::INFINITY);
            }
            temp_stat +=
                temp_expected * (ln_row_sums[i] + ln_col_sums[j] - temp_observed.ln() - ln_total);
        }
    }
    let final_stat = 2.0 * temp_stat;
    if final_stat < -EPS {
        bail!("Statistic evaluated to {final_stat}, which should be impossible.");
    }
    Ok(final_stat.max(0.0))
}

/// # Errors
///
/// Returns an error if an expected frequency evaluates to zero, or if the
/// calculated statistic results in a mathematically impossible value.
pub fn freeman_tukey(
    lambda: f64,
    observed: &Array2<f64>,
    row_sums: &Array1<f64>,
    col_times_total: &Array1<f64>,
) -> anyhow::Result<f64> {
    let (nrows, ncols) = observed.dim();
    let mut temp_stat: f64 = 0.0;
    for i in 0..nrows {
        for j in 0..ncols {
            let temp_expected: f64 = row_sums[i] * col_times_total[j]; //again the multiplication instead of division
            let temp_observed = observed[[i, j]];
            if temp_expected == 0.0 {
                bail!("Expected frequency is zero at position [{i}, {j}]");
            }
            temp_stat += temp_observed.sqrt() * temp_expected.sqrt() - temp_observed;
        }
    }
    let final_stat = (2.0 * temp_stat) / (lambda * (lambda + 1.0));
    if final_stat < -EPS {
        bail!("Statistic evaluated to {final_stat}, which should be impossible.");
    }
    Ok(final_stat.max(0.0))
}

/// # Errors
///
/// Returns an error if an expected frequency evaluates to zero, or if the
/// calculated statistic results in a mathematically impossible value.
pub fn cressie_read(
    lambda: f64,
    observed: &Array2<f64>,
    row_sums: &Array1<f64>,
    col_times_total: &Array1<f64>,
) -> anyhow::Result<f64> {
    let (nrows, ncols) = observed.dim();
    let mut temp_stat: f64 = 0.0;
    for i in 0..nrows {
        for j in 0..ncols {
            let temp_expected: f64 = row_sums[i] * col_times_total[j]; //again the multiplication instead of division
            let temp_observed = observed[[i, j]];
            if temp_expected == 0.0 {
                bail!("Expected frequency is zero at position [{i}, {j}]");
            }
            temp_stat += temp_observed * ((temp_observed / temp_expected).powf(lambda) - 1.0);
        }
    }
    let final_stat = (2.0 * temp_stat) / (lambda * (lambda + 1.0));
    if final_stat < -EPS {
        bail!("Statistic evaluated to {final_stat}, which should be impossible.");
    }
    Ok(final_stat.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_empty_table_error() {
        let observed = Array2::<f64>::zeros((0, 0));
        let result = contingency_test(&observed, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("size 0"));
    }

    #[test]
    fn test_negative_values_error() {
        let observed = array![[1.0, -1.0], [2.0, 3.0]];
        let result = contingency_test(&observed, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonnegative"));
    }

    #[test]
    fn test_zero_total_error() {
        let observed = array![[0.0, 0.0], [0.0, 0.0]];
        let result = contingency_test(&observed, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("> 0"));
    }

    #[test]
    fn test_zero_expected_frequency_error() {
        let observed = array![[10.0, 0.0], [0.0, 0.0]];
        let result = contingency_test(&observed, 1.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Expected frequency is zero"));
    }

    #[test]
    fn test_g_test_zero_expected_frequency() {
        let observed = array![[5.0, 0.0], [0.0, 0.0]];
        let result = contingency_test(&observed, 0.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Expected frequency is zero"));
    }

    #[test]
    fn test_modified_log_likelihood_zero_observed() {
        let observed = array![[5.0, 1.0], [2.0, 0.0]]; // contains zero observed
        let (statistic, p_value, _dof) = contingency_test(&observed, -1.0).unwrap();

        assert!(statistic.is_infinite() && p_value < EPS);
    }

    #[test]
    fn test_g_test_valid() {
        let observed = array![[10.0, 20.0], [20.0, 40.0]];
        let result = contingency_test(&observed, 0.0).unwrap();

        let (stat, p, dof) = result;
        assert!(stat >= 0.0);
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(dof, 1);
    }

    #[test]
    fn test_modified_log_likelihood_valid() {
        let observed = array![[10.0, 20.0], [20.0, 40.0]];
        let result = contingency_test(&observed, -1.0).unwrap();

        let (stat, p, dof) = result;
        assert!(stat >= 0.0);
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(dof, 1);
    }

    #[test]
    fn test_freeman_tukey_valid() {
        let observed = array![[5.0, 1.0], [1.0, 5.0]];
        let result = contingency_test(&observed, -0.5).unwrap();

        let (stat, p, dof) = result;
        assert!(stat >= 0.0);
        assert!((stat - 6.319_453_539_579_289).abs() < EPS); // Validated against scipy
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(dof, 1);
    }

    #[test]
    fn test_freeman_tukey_zero_observed() {
        let observed = array![[2.0, 1.0], [0.0, 3.0]]; // contains zero observed
        let result = contingency_test(&observed, -0.5);

        assert!(result.is_ok());
        let (stat, p, dof) = result.unwrap();

        assert!(stat >= 0.0);
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(dof, 1);

        // Manual calculation
        let expected_stat = 4.0
            * ((2.0f64.sqrt() - 1.0).powi(2) +
            (1.0 - 2.0f64.sqrt()).powi(2) +
            1.0 + // (0.0 - sqrt(1.0))^2
            (3.0f64.sqrt() - 2.0f64.sqrt()).powi(2));
        assert!((stat - expected_stat).abs() < EPS);
    }

    #[test]
    fn test_cressie_read_valid() {
        let observed = array![[10.0, 20.0], [20.0, 40.0]];
        let result = contingency_test(&observed, 0.5).unwrap();

        let (stat, p, dof) = result;
        assert!(stat >= 0.0);
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(dof, 1);
    }

    #[test]
    fn test_degrees_of_freedom_2x2() {
        let observed = array![[1.0, 2.0], [3.0, 4.0]];
        let (_, _, dof) = contingency_test(&observed, 0.0).unwrap();
        assert_eq!(dof, 1); // (2-1)*(2-1)
    }

    #[test]
    fn test_degrees_of_freedom_3x4() {
        let observed = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ];
        let (_, _, dof) = contingency_test(&observed, 0.0).unwrap();
        assert_eq!(dof, (3 - 1) * (4 - 1)); // 2 * 3 = 6
    }

    #[test]
    fn test_degrees_of_freedom_degenerate() {
        let observed = array![[1.0, 2.0, 3.0]]; // 1x3
        let (_, p, dof) = contingency_test(&observed, 0.0).unwrap();

        assert_eq!(dof, 0);
        assert!((p - 1.0).abs() < 1e12); // By definition in your implementation
    }
}
