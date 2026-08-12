from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


CHECKER_PATH = Path(__file__).with_name("check_pgmpy_parity.py")


def load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_pgmpy_parity", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yates_disabled_discrete_case_is_an_intentional_skip() -> None:
    checker = load_checker()
    case = {"test": "chi_squared", "params": {"yates": False}}
    assert checker.should_skip(case) == "pgmpy has no Yates-disable option"


def test_continuous_case_is_not_skipped() -> None:
    checker = load_checker()
    case = {"test": "fisher_z", "params": {}}
    assert checker.should_skip(case) is None


def test_compare_fields_reports_a_named_mismatch() -> None:
    checker = load_checker()
    errors = checker.compare_fields(
        case_id="pearson-positive",
        expected={
            "statistic": 0.8,
            "p_value": 0.01,
            "dof": 14,
            "effect_size": 0.8,
        },
        actual={
            "statistic": 0.7,
            "p_value": 0.01,
            "dof": 14,
            "effect_size": 0.8,
        },
        tolerance=1e-7,
    )
    assert errors == ["pearson-positive: statistic expected 0.8, got 0.7"]


def test_missing_pgmpy_dof_is_not_compared() -> None:
    checker = load_checker()
    assert (
        checker.compare_fields(
            case_id="pearson-unconditional",
            expected={
                "statistic": 0.0,
                "p_value": 1.0,
                "dof": 14,
                "effect_size": 0.0,
            },
            actual={
                "statistic": 0.0,
                "p_value": 1.0,
                "dof": None,
                "effect_size": 0.0,
            },
            tolerance=1e-7,
        )
        == []
    )


def test_pgmpy_numeric_dof_is_compared_when_fixture_expects_null() -> None:
    checker = load_checker()
    errors = checker.compare_fields(
        case_id="pearson-equivalence-inside-conditioned",
        expected={
            "statistic": 0.0499791900693,
            "p_value": 0.363006943902,
            "dof": None,
            "effect_size": 0.0499376169439,
        },
        actual={
            "statistic": 0.0499791900693,
            "p_value": 0.363006943902,
            "dof": 13,
            "effect_size": 0.0499376169439,
        },
        tolerance=1e-7,
    )
    assert errors == [
        "pearson-equivalence-inside-conditioned: dof expected None, got 13"
    ]


def test_conditioned_equivalence_numeric_dof_is_an_intentional_divergence() -> None:
    checker = load_checker()
    case = {
        "id": "pearson-equivalence-inside-conditioned",
        "test": "pearson_equivalence",
        "z": ["Z1"],
        "expected": {"dof": None},
    }

    assert checker.classify_intentional_divergence(case, "dof", 13) == (
        "pearson-equivalence-inside-conditioned: dof expected None, got 13 "
        "(current pgmpy propagates conditioned Pearsonr dof)"
    )


def test_unconditional_equivalence_numeric_dof_is_not_a_divergence() -> None:
    checker = load_checker()
    case = {
        "id": "pearson-equivalence-inside-unconditional",
        "test": "pearson_equivalence",
        "z": [],
        "expected": {"dof": None},
    }

    assert checker.classify_intentional_divergence(case, "dof", 14) is None


def test_other_numeric_dof_mismatch_is_not_a_divergence() -> None:
    checker = load_checker()
    case = {
        "id": "pearson-correlation-conditioned",
        "test": "pearson_correlation",
        "z": ["Z1"],
        "expected": {"dof": 13},
    }

    assert checker.classify_intentional_divergence(case, "dof", 12) is None


def test_declared_dof_divergence_does_not_hide_other_field_mismatch() -> None:
    checker = load_checker()
    case = {
        "id": "pearson-equivalence-inside-conditioned",
        "test": "pearson_equivalence",
        "z": ["Z1"],
        "expected": {
            "statistic": 0.8,
            "p_value": 0.01,
            "dof": None,
            "effect_size": 0.8,
        },
    }
    actual = {
        "statistic": 0.7,
        "p_value": 0.01,
        "dof": 13,
        "effect_size": 0.8,
    }

    errors, divergences = checker.compare_actual_fields(case, actual, 1e-7)

    assert errors == [
        "pearson-equivalence-inside-conditioned: statistic expected 0.8, got 0.7"
    ]
    assert divergences == [
        "pearson-equivalence-inside-conditioned: dof expected None, got 13 "
        "(current pgmpy propagates conditioned Pearsonr dof)"
    ]


def test_non_declared_dof_mismatch_remains_a_failure() -> None:
    checker = load_checker()
    case = {
        "id": "pearson-equivalence-unconditional",
        "test": "pearson_equivalence",
        "z": [],
        "expected": {
            "statistic": 0.8,
            "p_value": 0.01,
            "dof": None,
            "effect_size": 0.8,
        },
    }
    actual = {
        "statistic": 0.8,
        "p_value": 0.01,
        "dof": 14,
        "effect_size": 0.8,
    }

    errors, divergences = checker.compare_actual_fields(case, actual, 1e-7)

    assert errors == ["pearson-equivalence-unconditional: dof expected None, got 14"]
    assert divergences == []
