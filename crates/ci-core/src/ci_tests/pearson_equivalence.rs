//! Pearson equivalence (TOST) conditional-independence test (continuous).

use statrs::distribution::{ContinuousCDF, Normal};

use crate::ci_tests::pearson_correlation::{partial_correlation, RHO_CLIP_EPS};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, DataType, IndependenceRule, TestMeta};

/// Two One-Sided Tests (TOST) equivalence test on the partial correlation,
/// using Fisher's z-transform. A *low* p-value (below the significance level)
/// declares independence, so the rule is [`IndependenceRule::PValueLt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PearsonEquivalence {
    /// Equivalence margin on the correlation scale (the "negligible" effect).
    pub delta_threshold: f64,
}

impl PearsonEquivalence {
    /// Construct the test with the given equivalence margin.
    #[must_use]
    pub fn new(delta_threshold: f64) -> Self {
        Self { delta_threshold }
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
        if !(self.delta_threshold > 0.0 && self.delta_threshold < 1.0) {
            return Err(CiError::DegenerateData(format!(
                "delta_threshold must be in (0, 1), got {}",
                self.delta_threshold
            )));
        }

        let n = data.continuous(x)?.len();
        let n_z = z.len();

        let (rho_raw, _dof) = partial_correlation(data, x, y, z)?;
        let rho = rho_raw.clamp(-1.0 + RHO_CLIP_EPS, 1.0 - RHO_CLIP_EPS);

        let z_rho = rho.atanh();
        let z_delta = self.delta_threshold.atanh();

        // partial_correlation guarantees n >= |Z| + 3, so the radicand is >= 0
        // (matching FisherZ, which relies on the same invariant).
        #[allow(clippy::cast_precision_loss)]
        let c = ((n - n_z - 3) as f64).sqrt();

        let normal =
            Normal::new(0.0, 1.0).map_err(|e| CiError::Numeric(format!("standard normal: {e}")))?;

        let p_lower = 1.0 - normal.cdf(c * (z_rho + z_delta));
        let p_upper = normal.cdf(c * (z_rho - z_delta));
        let p_value = p_lower.max(p_upper);

        Ok(CiResult {
            statistic: Some(z_rho),
            p_value,
            dof: None,
            // effect_size reports the observed (un-clipped) partial correlation,
            // matching Fisher-Z; the clip only guards the atanh statistic above.
            effect_size: Some(rho_raw.abs()),
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
        let m = PearsonEquivalence::new(0.1).meta();
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
        let r = PearsonEquivalence::new(0.1).test(&data, 0, 1, &[]).unwrap();
        assert!(r.dof.is_none());
        // statistic is atanh(rho); effect_size is |rho|.
        let rho = r.statistic.unwrap().tanh();
        assert!((r.effect_size.unwrap() - rho.abs()).abs() < 1e-9);
    }

    #[test]
    fn out_of_range_delta_errors() {
        // delta_threshold must lie in (0, 1); 1.5 is rejected before any compute.
        let data = ds(vec![
            ("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("y", vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
        ]);
        assert!(matches!(
            PearsonEquivalence::new(1.5).test(&data, 0, 1, &[]),
            Err(CiError::DegenerateData(_))
        ));
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
            PearsonEquivalence::new(0.1).test(&data, 0, 1, &[2]),
            Err(CiError::DegenerateData(_))
        ));
    }
}
