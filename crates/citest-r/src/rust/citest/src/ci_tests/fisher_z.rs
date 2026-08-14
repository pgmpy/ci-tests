//! Fisher-z conditional-independence test (continuous).

use statrs::distribution::ContinuousCDF;

use crate::ci_tests::continuous_common::{fisher_z_inputs, standard_normal};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, DataType, IndependenceRule, TestMeta};

/// Fisher-z test on the (partial) correlation: `z = √(n − |Z| − 3) · atanh(ρ)`
/// is approximately standard normal under `X ⊥ Y | Z`. The de-facto standard
/// continuous CI test in causal discovery (pcalg's `gaussCItest`,
/// causal-learn's `fisherz`, pgmpy's `FisherZ`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FisherZ;

impl FisherZ {
    /// Construct the test (no configuration).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl CITest for FisherZ {
    #[allow(
        clippy::many_single_char_names,
        reason = "x, y, z are the standard conditional-independence variable names from the contract"
    )]
    fn test_impl(
        &self,
        data: &Dataset,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> Result<CiResult, CiError> {
        let inputs = fisher_z_inputs(data, x, y, z)?;
        let statistic = inputs.scale * inputs.z_rho;
        let p_value = 2.0 * standard_normal()?.sf(statistic.abs());

        Ok(CiResult {
            statistic: Some(statistic),
            p_value,
            dof: None,
            effect_size: Some(inputs.rho_raw.abs()),
        })
    }

    fn meta(&self) -> TestMeta {
        TestMeta {
            name: "fisher_z",
            data_types: &[DataType::Continuous],
            symmetric: true,
            rule: IndependenceRule::PValueGe,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_tests::PearsonCorrelation;
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
    fn meta_is_correct() {
        let m = FisherZ::new().meta();
        assert_eq!(m.name, "fisher_z");
        assert_eq!(m.data_types, &[DataType::Continuous]);
        assert!(m.symmetric);
        assert_eq!(m.rule, IndependenceRule::PValueGe);
    }

    #[test]
    fn statistic_is_scaled_atanh_of_r() {
        let data = ds(vec![
            ("x", vec![0.2, 1.1, -0.4, 0.9, 1.8, -0.7, 0.3, 1.2]),
            ("y", vec![1.0, 0.1, 0.8, -0.2, 0.4, 1.1, -0.3, 0.6]),
        ]);
        let fz = FisherZ::new().test(&data, 0, 1, &[]).unwrap();
        let pc = PearsonCorrelation::new().test(&data, 0, 1, &[]).unwrap();
        let r = pc.statistic.unwrap();
        let expected = (8.0_f64 - 3.0).sqrt() * r.atanh();
        assert!((fz.statistic.unwrap() - expected).abs() < 1e-12);
        assert!(fz.dof.is_none());
        assert!((fz.effect_size.unwrap() - r.abs()).abs() < 1e-12);
        assert!(fz.p_value > 0.0 && fz.p_value <= 1.0);
    }

    #[test]
    fn conditional_uses_n_minus_z_minus_3() {
        let z: Vec<f64> = (0..20u8).map(|i| f64::from(i) * 0.37 % 1.9).collect();
        let x: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.31 * f64::from(u8::try_from(i).unwrap() % 5))
            .collect();
        let y: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, v)| v - 0.27 * f64::from(u8::try_from(i).unwrap() % 7))
            .collect();
        let data = ds(vec![("x", x), ("y", y), ("z", z)]);
        let fz = FisherZ::new().test(&data, 0, 1, &[2]).unwrap();
        let pc = PearsonCorrelation::new().test(&data, 0, 1, &[2]).unwrap();
        let r = pc.statistic.unwrap();
        let expected = (20.0_f64 - 1.0 - 3.0).sqrt() * r.atanh();
        assert!((fz.statistic.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn perfect_correlation_is_clamped_finite() {
        // y = 2x: r == 1 exactly. PearsonCorrelation errors here (t undefined);
        // FisherZ instead clamps rho and reports a large finite statistic.
        let x: Vec<f64> = (0..30u8).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let data = ds(vec![("x", x), ("y", y)]);
        let res = FisherZ::new().test(&data, 0, 1, &[]).unwrap();
        let stat = res.statistic.unwrap();
        assert!(stat.is_finite() && stat > 0.0);
        assert!(res.p_value < 1e-12);
        // effect_size reports the raw (unclipped) |r| — pgmpy convention.
        assert!((res.effect_size.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn is_independent_applies_p_value_ge_rule() {
        let x: Vec<f64> = (0..24u8).map(|i| f64::from(i % 4)).collect();
        let y: Vec<f64> = (0..24u8).map(|i| f64::from((i / 4) % 3)).collect();
        let data = ds(vec![("x", x), ("y", y)]);
        let fz = FisherZ::new();
        let res = fz.test(&data, 0, 1, &[]).unwrap();
        // The decision must equal the PValueGe rule applied to the p-value.
        assert_eq!(
            fz.is_independent(&data, 0, 1, &[], 0.05).unwrap(),
            res.p_value >= 0.05
        );
    }
}
