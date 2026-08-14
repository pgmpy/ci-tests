//! Pearson equivalence (TOST) conditional-independence test (continuous).

use statrs::distribution::ContinuousCDF;

use crate::ci_tests::continuous_common::{fisher_z_inputs, standard_normal};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, DataType, IndependenceRule, TestMeta};

/// Two One-Sided Tests (TOST) equivalence test on the partial correlation,
/// using Fisher's z-transform. A *low* p-value (below the significance level)
/// declares independence, so the rule is [`IndependenceRule::PValueLt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PearsonEquivalence {
    /// Equivalence margin on the correlation scale (the "negligible" effect).
    /// Private so that it cannot be set to a value [`PearsonEquivalence::new`]
    /// would reject; read it back with
    /// [`PearsonEquivalence::delta_threshold`].
    delta_threshold: f64,
}

impl PearsonEquivalence {
    /// Construct the test with the given equivalence margin.
    ///
    /// Configuration is validated here rather than on every query, so an
    /// out-of-range margin fails once, at the point the caller made the
    /// mistake.
    ///
    /// # Errors
    ///
    /// Returns [`CiError::InvalidConfig`] unless `delta_threshold` lies
    /// strictly within `(0, 1)`. A margin of 0 admits no equivalence region,
    /// and 1 or more admits every correlation.
    pub fn new(delta_threshold: f64) -> Result<Self, CiError> {
        if !(delta_threshold > 0.0 && delta_threshold < 1.0) {
            return Err(CiError::InvalidConfig(format!(
                "delta_threshold must be in (0, 1), got {delta_threshold}"
            )));
        }
        Ok(Self { delta_threshold })
    }

    /// The configured equivalence margin.
    #[must_use]
    pub fn delta_threshold(&self) -> f64 {
        self.delta_threshold
    }
}

impl CITest for PearsonEquivalence {
    #[allow(
        clippy::many_single_char_names,
        reason = "x, y, z are the contract variable names; c is the Fisher-z scale factor"
    )]
    fn test_impl(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult, CiError> {
        let inputs = fisher_z_inputs(data, x, y, z)?;
        let z_delta = self.delta_threshold.atanh();
        let normal = standard_normal()?;
        let c = inputs.scale;
        let z_rho = inputs.z_rho;

        let p_lower = 1.0 - normal.cdf(c * (z_rho + z_delta));
        let p_upper = normal.cdf(c * (z_rho - z_delta));
        let p_value = p_lower.max(p_upper);

        Ok(CiResult {
            statistic: Some(z_rho),
            p_value,
            dof: None,
            // effect_size reports the observed (un-clipped) partial correlation,
            // matching Fisher-Z; the clip only guards the atanh statistic above.
            effect_size: Some(inputs.rho_raw.abs()),
        })
    }

    fn meta(&self) -> TestMeta {
        TestMeta {
            name: "pearson_equivalence",
            data_types: &[DataType::Continuous],
            symmetric: true,
            rule: IndependenceRule::PValueLt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ColumnKind;

    fn ds(cols: Vec<(&str, Vec<f64>)>) -> Dataset {
        Dataset::from_columns(
            cols.into_iter()
                .map(|(n, v)| (n.to_string(), ColumnKind::Continuous, v))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn meta_uses_pvalue_lt() {
        let m = PearsonEquivalence::new(0.1).unwrap().meta();
        assert_eq!(m.name, "pearson_equivalence");
        assert_eq!(m.rule, IndependenceRule::PValueLt);
        assert_eq!(m.data_types, &[DataType::Continuous]);
        assert!(m.symmetric);
    }

    #[test]
    fn reports_fisher_z_statistic_and_no_dof() {
        let data = ds(vec![
            ("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("y", vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
        ]);
        let r = PearsonEquivalence::new(0.1)
            .unwrap()
            .test(&data, 0, 1, &[])
            .unwrap();
        assert!(r.dof.is_none());
        // statistic is atanh(rho); effect_size is |rho|.
        let rho = r.statistic.unwrap().tanh();
        assert!((r.effect_size.unwrap() - rho.abs()).abs() < 1e-9);
    }

    #[test]
    fn out_of_range_delta_is_rejected_at_construction() {
        // The margin is configuration, not data, so it fails once when the
        // caller supplies it rather than on every query -- and reports
        // InvalidConfig, not DegenerateData, which would blame the data.
        for delta in [1.5, 1.0, 0.0, -0.1, f64::NAN, f64::INFINITY] {
            let err = PearsonEquivalence::new(delta).unwrap_err();
            assert!(
                matches!(err, CiError::InvalidConfig(_)),
                "delta {delta}: {err}"
            );
            assert!(err.to_string().contains("delta_threshold"));
        }
    }

    #[test]
    fn valid_delta_round_trips() {
        let test = PearsonEquivalence::new(0.25).unwrap();
        assert!((test.delta_threshold() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn near_zero_correlation_declares_independence() {
        // Exactly-zero correlation by construction: each x value pairs with +1
        // and -1 in y, so the covariance is 0. With delta 0.2 and n = 100 the
        // TOST p-value falls below 0.05, so PValueLt declares independence.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..50 {
            x.push(f64::from(i));
            y.push(1.0);
            x.push(f64::from(i));
            y.push(-1.0);
        }
        let data = ds(vec![("x", x), ("y", y)]);
        assert!(PearsonEquivalence::new(0.2)
            .unwrap()
            .is_independent(&data, 0, 1, &[], 0.05)
            .unwrap());
    }

    #[test]
    fn strong_correlation_not_independent_under_small_delta() {
        // A near-perfect correlation with a tiny equivalence margin: the TOST
        // p-value is large, so PValueLt does not declare independence.
        let data = ds(vec![
            ("x", vec![1., 2., 3., 4., 5.]),
            ("y", vec![2., 4.1, 5.9, 8.2, 9.8]),
        ]);
        assert!(!PearsonEquivalence::new(0.05)
            .unwrap()
            .is_independent(&data, 0, 1, &[], 0.05)
            .unwrap());
    }

    #[test]
    fn too_few_rows_is_degenerate() {
        // n=3, |Z|=1 -> n - |Z| - 3 = -1 < 0.
        let data = ds(vec![
            ("x", vec![1.0, 2.0, 3.0]),
            ("y", vec![3.0, 2.0, 1.0]),
            ("z", vec![1.0, 2.0, 3.0]),
        ]);
        assert!(matches!(
            PearsonEquivalence::new(0.1)
                .unwrap()
                .test(&data, 0, 1, &[2]),
            Err(CiError::DegenerateData(_))
        ));
    }
}
