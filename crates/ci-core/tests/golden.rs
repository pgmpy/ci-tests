//! Fixture-driven acceptance gate.
//!
//! Parses the language-agnostic reference fixture (`tests/fixtures/golden.json`
//! at the repository root) and checks **every** case against the scipy-derived
//! expectations:
//!
//! - The five discrete power-divergence tests (`chi_squared`, `log_likelihood`,
//!   `cressie_read`, `freeman_tukey`, `modified_likelihood`) are run with the
//!   case's own `params.yates`, so both Yates-on and Yates-off rows are covered.
//! - All `pearson_correlation`, `pearson_equivalence`, and `fisher_z` rows.
//!
//! Every non-null `statistic`/`p_value`/`dof`/`effect_size` field is asserted to
//! within `1e-7`, with the `+∞` statistic / `p = 0` degenerate case handled
//! exactly. The test asserts it exercised all 80 fixture cases.

use std::collections::BTreeMap;

use ci_core::ci_tests::{
    ChiSquared, CressieRead, FisherZ, FreemanTukey, LogLikelihood, ModifiedLikelihood,
    PearsonCorrelation, PearsonEquivalence,
};
use ci_core::dataset::{ColumnKind, Dataset};
use ci_core::strategy::{CITest, CiResult};
use serde::Deserialize;

const TOL: f64 = 1e-7;
const EXPECTED_CASE_COUNT: usize = 80;

#[derive(Debug, Default, Deserialize)]
struct Params {
    #[serde(default)]
    delta_threshold: Option<f64>,
    #[serde(default)]
    yates: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ColumnSpec {
    kind: String,
    values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Expected {
    statistic: Option<f64>,
    p_value: Option<f64>,
    dof: Option<usize>,
    effect_size: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Case {
    test: String,
    #[serde(default)]
    params: Params,
    columns: BTreeMap<String, ColumnSpec>,
    x: String,
    y: String,
    z: Vec<String>,
    expected: Expected,
}

fn load_cases() -> Vec<Case> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/golden.json"
    );
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read fixture at {path}: {e}"));
    serde_json::from_str(&raw).expect("failed to parse golden.json")
}

/// Build a `Dataset` plus a name->index map from a case's columns, preserving a
/// deterministic ordering so X/Y/Z names resolve correctly.
fn build_dataset(case: &Case) -> (Dataset, BTreeMap<String, usize>) {
    let mut cols = Vec::new();
    let mut index = BTreeMap::new();
    for (i, (name, spec)) in case.columns.iter().enumerate() {
        let kind = match spec.kind.as_str() {
            "discrete" => ColumnKind::Discrete,
            "continuous" => ColumnKind::Continuous,
            other => panic!("unknown column kind {other}"),
        };
        cols.push((name.clone(), kind, spec.values.clone()));
        index.insert(name.clone(), i);
    }
    (Dataset::from_columns(cols).unwrap(), index)
}

/// The `yates` flag for a discrete case (defaults to `true` if unspecified).
fn case_yates(case: &Case) -> bool {
    case.params.yates.unwrap_or(true)
}

fn run_case(case: &Case) -> CiResult {
    let (data, index) = build_dataset(case);
    let x = index[&case.x];
    let y = index[&case.y];
    let z: Vec<usize> = case.z.iter().map(|n| index[n]).collect();

    // Construct each discrete test with the case's own Yates flag so both the
    // Yates-on and Yates-off fixture rows are exercised.
    let test: Box<dyn CITest> = match case.test.as_str() {
        "chi_squared" => Box::new(ChiSquared {
            yates: case_yates(case),
        }),
        "log_likelihood" => Box::new(LogLikelihood {
            yates: case_yates(case),
        }),
        "cressie_read" => Box::new(CressieRead {
            yates: case_yates(case),
        }),
        "freeman_tukey" => Box::new(FreemanTukey {
            yates: case_yates(case),
        }),
        "modified_likelihood" => Box::new(ModifiedLikelihood {
            yates: case_yates(case),
        }),
        "pearson_correlation" => Box::new(PearsonCorrelation::new()),
        "pearson_equivalence" => {
            let delta = case
                .params
                .delta_threshold
                .expect("equivalence case missing delta_threshold");
            Box::new(PearsonEquivalence::new(delta))
        }
        "fisher_z" => Box::new(FisherZ::new()),
        other => panic!("run_case called for unsupported test {other}"),
    };

    test.test(&data, x, y, &z).unwrap()
}

fn assert_field(name: &str, case: &Case, expected: Option<f64>, actual: Option<f64>) {
    let Some(exp) = expected else { return };
    let act = actual
        .unwrap_or_else(|| panic!("[{}] expected {name}={exp} but result had None", case.test));

    // Handle the degenerate `+∞` statistic / `p = 0` case exactly: an infinite
    // expectation requires a matching infinite actual of the same sign (and
    // vice versa). `total_cmp` gives an exact ordering without a float `==`.
    if exp.is_infinite() || act.is_infinite() {
        assert!(
            exp.total_cmp(&act).is_eq(),
            "[{}] {name} mismatch: expected {exp}, got {act} (x={}, y={}, z={:?})",
            case.test,
            case.x,
            case.y,
            case.z,
        );
        return;
    }

    assert!(
        (act - exp).abs() < TOL,
        "[{}] {name} mismatch: expected {exp}, got {act} (x={}, y={}, z={:?})",
        case.test,
        case.x,
        case.y,
        case.z,
    );
}

#[test]
fn golden_fixture_matches() {
    let cases = load_cases();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();

    for case in &cases {
        let result = run_case(case);

        assert_field("statistic", case, case.expected.statistic, result.statistic);
        assert_field("p_value", case, case.expected.p_value, Some(result.p_value));
        #[allow(clippy::cast_precision_loss)]
        let dof_expected = case.expected.dof.map(|d| d as f64);
        #[allow(clippy::cast_precision_loss)]
        let dof_actual = result.dof.map(|d| d as f64);
        assert_field("dof", case, dof_expected, dof_actual);
        assert_field(
            "effect_size",
            case,
            case.expected.effect_size,
            result.effect_size,
        );

        *counts.entry(case.test.as_str()).or_default() += 1;
    }

    eprintln!("golden fixture: asserted cases per test: {counts:?}");

    // Every test in the fixture must be covered, and every single case asserted.
    let total: usize = counts.values().sum();
    assert_eq!(
        total, EXPECTED_CASE_COUNT,
        "expected to assert all {EXPECTED_CASE_COUNT} fixture cases, asserted {total}: {counts:?}"
    );
    for test in [
        "chi_squared",
        "log_likelihood",
        "cressie_read",
        "freeman_tukey",
        "modified_likelihood",
        "pearson_correlation",
        "pearson_equivalence",
        "fisher_z",
    ] {
        assert!(
            counts.get(test).copied().unwrap_or(0) > 0,
            "no cases asserted for {test}"
        );
    }
}
