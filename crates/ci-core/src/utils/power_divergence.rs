use crate::strategy::TestResult;
use crate::utils::contingency_table::{
    build_global_category_map, contingency_table, contingency_table_from_indices,
};
use crate::utils::contingency_test::contingency_test;
use crate::utils::partition_indices::partition_indices;
use anyhow::ensure;
use ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Run a power-divergence based conditional independence test.
///
/// # Errors
/// Returns an error if the inputs are invalid or if the underlying contingency
/// test or chi-squared distribution construction fails.
pub fn power_divergence(
    x_values: &Array1<f64>,
    y_values: &Array1<f64>,
    z: &Array2<f64>,
    boolean: bool,
    significance_level: f64,
    lambda: f64,
) -> anyhow::Result<TestResult> {
    ensure!(
        x_values.len() == y_values.len(),
        "x and y must have the same length, got {} and {}",
        x_values.len(),
        y_values.len(),
    );
    ensure!(
        z.ncols() == 0 || z.nrows() == x_values.len(),
        "z must have the same number of rows as x and y ({}), got {}",
        x_values.len(),
        z.nrows(),
    );
    if z.ncols() == 0 {
        let table = contingency_table(x_values, y_values);
        let (statistic, p_value, degrees_of_freedom) = contingency_test(&table, lambda)?;
        return Ok(wrap_result(
            boolean,
            p_value,
            statistic,
            degrees_of_freedom,
            significance_level,
        ));
    }

    let x_categories = build_global_category_map(x_values);
    let y_categories = build_global_category_map(y_values);

    let mut statistic = 0.0;
    let mut degrees_of_freedom = 0;

    for indices in partition_indices(z) {
        let table = contingency_table_from_indices(
            &indices,
            x_values,
            y_values,
            &x_categories,
            &y_categories,
        );
        let Ok((stat, _p, dof)) = contingency_test(&table, lambda) else {
            continue;
        };

        if dof == 0 {
            continue;
        }
        statistic += stat;
        degrees_of_freedom += dof;
    }
    let p_value = if degrees_of_freedom == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        ChiSquared::new(degrees_of_freedom as f64)?.sf(statistic)
    };
    Ok(wrap_result(
        boolean,
        p_value,
        statistic,
        degrees_of_freedom,
        significance_level,
    ))
}

fn wrap_result(
    boolean: bool,
    p_value: f64,
    coefficient: f64,
    degrees_of_freedom: usize,
    significance_level: f64,
) -> TestResult {
    if boolean {
        return TestResult::Boolean(p_value >= significance_level);
    }
    TestResult::Statistic(p_value, coefficient, degrees_of_freedom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::EPS;
    use ndarray::{array, Array2};

    const LAMBDA: f64 = 1.0;
    const SIGNIFICANCE_LEVEL: f64 = 0.05;

    fn empty_z() -> Array2<f64> {
        Array2::zeros((0, 0))
    }

    fn unwrap_statistic(r: &TestResult) -> (f64, f64, usize) {
        match r {
            TestResult::Statistic(a, b, c) => (*a, *b, *c),
            _ => panic!("expected Statistic"),
        }
    }

    #[test]
    fn unconditional_mismatched_lengths_returns_error() {
        let x = array![1., 2., 3.];
        let y = array![1., 2.];
        let result = power_divergence(&x, &y, &empty_z(), false, SIGNIFICANCE_LEVEL, LAMBDA);
        assert!(result.is_err());
    }

    // A single observation produces a 1x1 table with dof=0.
    #[test]
    fn unconditional_single_element_has_zero_dof() {
        let x = array![1.];
        let y = array![1.];
        let result =
            power_divergence(&x, &y, &empty_z(), false, SIGNIFICANCE_LEVEL, LAMBDA).unwrap();
        let (p, stat, dof) = unwrap_statistic(&result);
        assert_eq!(dof, 0);
        assert!((p - 1.0).abs() < EPS);
        assert!(stat.abs() < EPS);
    }

    // When X has only one distinct value the table has one row → dof=0.
    #[test]
    fn unconditional_single_category_in_x_has_zero_dof() {
        let x = array![1., 1., 1., 1.];
        let y = array![1., 2., 1., 2.];
        let result =
            power_divergence(&x, &y, &empty_z(), false, SIGNIFICANCE_LEVEL, LAMBDA).unwrap();
        let (p, _stat, dof) = unwrap_statistic(&result);
        assert_eq!(dof, 0);
        assert!((p - 1.0).abs() < EPS);
    }

    // When Y has only one distinct value the table has one column → dof=0.
    #[test]
    fn unconditional_single_category_in_y_has_zero_dof() {
        let x = array![1., 2., 1., 2.];
        let y = array![1., 1., 1., 1.];
        let result =
            power_divergence(&x, &y, &empty_z(), false, SIGNIFICANCE_LEVEL, LAMBDA).unwrap();
        let (p, _stat, dof) = unwrap_statistic(&result);
        assert_eq!(dof, 0);
        assert!((p - 1.0).abs() < EPS);
    }

    // NaN in x_values becomes its own category via OrderedFloat.
    // The result is a valid (though likely meaningless) statistic, not a panic.
    #[test]
    fn unconditional_nan_in_x_does_not_panic() {
        let x = array![1., f64::NAN, 2., 1.];
        let y = array![1., 2., 1., 2.];
        let result = power_divergence(&x, &y, &empty_z(), false, SIGNIFICANCE_LEVEL, LAMBDA);
        assert!(result.is_ok());
    }

    // NaN in the conditioning set creates its own partition group.
    // Global category maps include both X categories, so a singleton
    // group gets a table with a zero-expected cell. That group is
    // skipped, so the overall result is valid (p=1 when all groups are skipped).
    #[test]
    fn conditional_nan_in_z_skips_bad_groups() {
        let x = array![1., 1., 2., 2.];
        let y = array![1., 2., 1., 2.];
        let z = Array2::from_shape_vec((4, 1), vec![0., f64::NAN, 0., 1.]).unwrap();
        let result = power_divergence(&x, &y, &z, false, SIGNIFICANCE_LEVEL, LAMBDA).unwrap();
        let (p, _stat, dof) = unwrap_statistic(&result);
        assert_eq!(dof, 0);
        assert!((p - 1.0).abs() < EPS);
    }

    // Each Z-group has only one X value. Global category maps force a 2x2
    // table per group with a zero row → zero expected frequency. All
    // groups are skipped, yielding dof=0 and p=1.
    #[test]
    fn conditional_all_groups_single_category_skips_all() {
        let x = array![1., 1., 2., 2.];
        let y = array![1., 2., 1., 2.];
        let z = Array2::from_shape_vec((4, 1), vec![0., 0., 1., 1.]).unwrap();
        let result = power_divergence(&x, &y, &z, false, SIGNIFICANCE_LEVEL, LAMBDA).unwrap();
        let (p, _stat, dof) = unwrap_statistic(&result);
        assert_eq!(dof, 0);
        assert!((p - 1.0).abs() < EPS);
    }
}
