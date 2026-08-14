//! The power-divergence family of discrete conditional-independence tests.
//!
//! The five members differ only in the parameter `λ` selecting a member of the
//! family; everything else — the `yates` flag, the contract, the statistic, the
//! p-value and the Cramér's V effect size — is identical, and lives in
//! [`super::discrete_common`]. Declaring them with a macro keeps that fact
//! visible: a member *is* a name plus a `λ`, and adding a sixth is one line
//! rather than a sixth near-identical file.
//!
//! Each type keeps its own documentation, so the generated rustdoc is the same
//! as five hand-written definitions would produce.

use crate::ci_tests::discrete_common::{discrete_meta, run_power_divergence};
use crate::dataset::Dataset;
use crate::error::CiError;
use crate::strategy::{CITest, CiResult, TestMeta};

/// Declare one member of the power-divergence family.
///
/// Takes the type name, its rustdoc, the stable [`TestMeta::name`], and the
/// `λ` that selects the family member.
macro_rules! power_divergence_test {
    ($(#[doc = $doc:expr])+ $name:ident, $stable_name:literal, $lambda:expr) => {
        $(#[doc = $doc])+
        ///
        /// When `yates` is set, Yates' continuity correction is applied on 2×2
        /// (sub-)tables, matching `scipy.stats.chi2_contingency` and pgmpy.
        /// Cramér's V is reported as the effect size.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            /// Whether to apply Yates' continuity correction on 2×2 (sub-)tables.
            pub yates: bool,
        }

        impl $name {
            /// Construct the test with Yates' continuity correction enabled
            /// (the scipy/pgmpy default).
            #[must_use]
            pub fn new() -> Self {
                Self { yates: true }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl CITest for $name {
            fn test_impl(
                &self,
                data: &Dataset,
                x: usize,
                y: usize,
                z: &[usize],
            ) -> Result<CiResult, CiError> {
                run_power_divergence(data, x, y, z, $lambda, self.yates)
            }

            fn meta(&self) -> TestMeta {
                discrete_meta($stable_name)
            }
        }
    };
}

power_divergence_test!(
    /// Pearson chi-squared test for discrete data (power divergence with
    /// `λ = 1`), the default choice for contingency-table independence.
    ChiSquared,
    "chi_squared",
    1.0
);

power_divergence_test!(
    /// Log-likelihood ratio (G-test) for discrete data (`λ = 0`). The per-cell
    /// statistic is `2 · Σ O·ln(O/E)`.
    LogLikelihood,
    "log_likelihood",
    0.0
);

power_divergence_test!(
    /// Cressie-Read power-divergence test for discrete data (`λ = 2/3`), the
    /// recommended compromise between the Pearson and likelihood-ratio
    /// statistics.
    CressieRead,
    "cressie_read",
    2.0 / 3.0
);

power_divergence_test!(
    /// Freeman-Tukey power-divergence test for discrete data (`λ = −1/2`).
    FreemanTukey,
    "freeman_tukey",
    -0.5
);

power_divergence_test!(
    /// Modified log-likelihood (Neyman) power-divergence test for discrete data
    /// (`λ = −1`). The per-cell statistic is `2 · Σ E·ln(E/O)`; a structural
    /// zero (`O = 0` with `E > 0`) drives the statistic to `+∞` and hence the
    /// p-value to `0`.
    ModifiedLikelihood,
    "modified_likelihood",
    -1.0
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_tests::discrete_common::discrete_dataset as ds;
    use crate::dataset::ColumnKind;

    /// `[[4,0],[0,4]]`, a perfectly dependent 2×2 table.
    fn dependent_2x2() -> Dataset {
        ds(vec![
            ("x", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
            ("y", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
        ])
    }

    #[test]
    fn chi_squared_applies_yates() {
        // Yates makes stat = 4 * (1.5^2 / 2) = 4.5.
        let r = ChiSquared::new().test(&dependent_2x2(), 0, 1, &[]).unwrap();
        assert!(
            (r.statistic.unwrap() - 4.5).abs() < 1e-9,
            "got {:?}",
            r.statistic
        );
        assert_eq!(r.dof, Some(1));
    }

    #[test]
    fn chi_squared_yates_off_changes_statistic() {
        // The same table without Yates -> full chi-square = 8.
        let r = ChiSquared { yates: false }
            .test(&dependent_2x2(), 0, 1, &[])
            .unwrap();
        assert!(
            (r.statistic.unwrap() - 8.0).abs() < 1e-9,
            "got {:?}",
            r.statistic
        );
    }

    #[test]
    fn conditional_independence_sums_over_strata() {
        let data = ds(vec![
            ("x", vec![1., 1., 2., 2., 1., 1., 2., 2.]),
            ("y", vec![1., 2., 1., 2., 1., 2., 1., 2.]),
            ("z", vec![1., 1., 1., 1., 2., 2., 2., 2.]),
        ]);
        let r = ChiSquared::new().test(&data, 0, 1, &[2]).unwrap();
        assert!(r.statistic.unwrap().abs() < 1e-9);
        assert_eq!(r.dof, Some(2), "one dof per stratum");
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn continuous_column_errors_for_every_member() {
        let data = Dataset::from_columns(vec![
            ("x".into(), ColumnKind::Continuous, vec![1., 2., 3.]),
            ("y".into(), ColumnKind::Discrete, vec![1., 2., 3.]),
        ])
        .unwrap();
        for (label, test) in family() {
            assert!(
                matches!(
                    test.test(&data, 0, 1, &[]),
                    Err(CiError::WrongColumnKind(_))
                ),
                "{label} accepted a continuous column"
            );
        }
    }

    /// Every member, boxed, for contract assertions that must hold family-wide.
    fn family() -> Vec<(&'static str, Box<dyn CITest>)> {
        vec![
            ("chi_squared", Box::new(ChiSquared::new())),
            ("log_likelihood", Box::new(LogLikelihood::new())),
            ("cressie_read", Box::new(CressieRead::new())),
            ("freeman_tukey", Box::new(FreemanTukey::new())),
            ("modified_likelihood", Box::new(ModifiedLikelihood::new())),
        ]
    }

    /// Each member must pin its own λ: run the same dependent table through all
    /// five and require five distinct statistics. A copy-paste error in the
    /// macro invocations (a wrong λ, or a duplicated one) shows up here.
    #[test]
    fn each_member_pins_a_distinct_lambda() {
        let data = dependent_2x2();
        let mut seen: Vec<(&str, f64)> = Vec::new();
        for (name, test) in family() {
            let r = test.test(&data, 0, 1, &[]).unwrap();
            let statistic = r.statistic.expect("discrete tests define a statistic");
            assert_eq!(test.meta().name, name, "stable name must match its type");
            for (other, value) in &seen {
                assert!(
                    (statistic - value).abs() > 1e-9,
                    "{name} and {other} produced the same statistic {statistic}; \
                     one of the λ values is wrong"
                );
            }
            seen.push((name, statistic));
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn effect_size_is_finite_on_an_independent_table() {
        let data = ds(vec![
            ("x", vec![1., 1., 2., 2.]),
            ("y", vec![1., 2., 1., 2.]),
        ]);
        for (name, test) in family() {
            let r = test.test(&data, 0, 1, &[]).unwrap();
            let v = r.effect_size.expect("Cramér's V is always reported");
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{name}: V = {v}");
        }
    }

    #[test]
    fn modified_likelihood_structural_zero_gives_p_zero() {
        // λ = −1 is the only member for which O = 0 with E > 0 is +∞.
        let data = ds(vec![
            ("x", vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]),
            ("y", vec![1., 2., 2., 1., 2., 3., 1., 2., 3.]),
        ]);
        let r = ModifiedLikelihood::new().test(&data, 0, 1, &[]).unwrap();
        assert!(
            r.statistic.unwrap().is_infinite(),
            "expected +inf, got {:?}",
            r.statistic
        );
        assert_eq!(r.p_value, 0.0);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(ChiSquared::default(), ChiSquared::new());
        assert!(ChiSquared::default().yates, "Yates on by default");
    }
}
