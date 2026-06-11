//! Registry of the built-in closed-form conditional-independence tests.
//!
//! A single source of truth listing the seven tests that ship with this crate,
//! so callers (and the language bindings) can enumerate the available tests and
//! construct a default-configured one by its stable [`TestMeta::name`] without
//! depending on every concrete struct.
//!
//! The default configuration for each test matches the scipy/pgmpy convention:
//! the discrete power-divergence tests enable Yates' continuity correction, and
//! [`PearsonEquivalence`] uses an equivalence margin of `0.1`.

use crate::ci_tests::{
    ChiSquared, CressieRead, FreemanTukey, LogLikelihood, ModifiedLikelihood, PearsonCorrelation,
    PearsonEquivalence,
};
use crate::strategy::{CITest, TestMeta};

/// Default equivalence margin for [`PearsonEquivalence`] in [`make_default`].
const DEFAULT_DELTA_THRESHOLD: f64 = 0.1;

/// Construct every built-in test in its default configuration as a boxed
/// [`CITest`]. The single source of truth that [`all_metas`] and
/// [`make_default`] are derived from.
fn default_tests() -> Vec<Box<dyn CITest>> {
    vec![
        Box::new(ChiSquared::new()),
        Box::new(LogLikelihood::new()),
        Box::new(CressieRead::new()),
        Box::new(FreemanTukey::new()),
        Box::new(ModifiedLikelihood::new()),
        Box::new(PearsonCorrelation::new()),
        Box::new(PearsonEquivalence::new(DEFAULT_DELTA_THRESHOLD)),
    ]
}

/// Metadata for every built-in closed-form test, in registry order.
#[must_use]
pub fn all_metas() -> Vec<TestMeta> {
    default_tests().iter().map(|t| t.meta()).collect()
}

/// Construct a default-configured boxed test by its [`TestMeta::name`].
///
/// Returns `None` if `name` does not match any built-in test. Default
/// configuration: `chi_squared`/`log_likelihood`/`cressie_read`/
/// `freeman_tukey`/`modified_likelihood` have Yates enabled, and
/// `pearson_equivalence` uses `delta_threshold = 0.1`.
#[must_use]
pub fn make_default(name: &str) -> Option<Box<dyn CITest>> {
    default_tests().into_iter().find(|t| t.meta().name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The seven stable names the registry must expose.
    const EXPECTED_NAMES: [&str; 7] = [
        "chi_squared",
        "log_likelihood",
        "cressie_read",
        "freeman_tukey",
        "modified_likelihood",
        "pearson_correlation",
        "pearson_equivalence",
    ];

    #[test]
    fn exposes_seven_unique_metas() {
        let metas = all_metas();
        assert_eq!(metas.len(), 7, "expected exactly 7 registered tests");
        let names: BTreeSet<&str> = metas.iter().map(|m| m.name).collect();
        assert_eq!(names.len(), 7, "test names must be unique");
        for expected in EXPECTED_NAMES {
            assert!(names.contains(expected), "missing test `{expected}`");
        }
    }

    #[test]
    fn make_default_resolves_all_names() {
        for name in EXPECTED_NAMES {
            let test = make_default(name).unwrap_or_else(|| panic!("`{name}` did not resolve"));
            assert_eq!(test.meta().name, name);
        }
    }

    #[test]
    fn make_default_unknown_is_none() {
        assert!(make_default("not_a_real_test").is_none());
    }
}
