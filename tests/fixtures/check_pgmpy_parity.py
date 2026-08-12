#!/usr/bin/env python3
"""Optionally compare the independent golden fixture with current pgmpy."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import numbers
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_GOLDEN = Path(__file__).with_name("golden.json")
DEFAULT_TOLERANCE = 1e-7
DISCRETE_TESTS = {
    "chi_squared",
    "log_likelihood",
    "cressie_read",
    "freeman_tukey",
    "modified_likelihood",
}


Constructor = Callable[[ModuleType, Any, dict[str, Any]], Any]
CONSTRUCTORS: dict[str, Constructor] = {
    "chi_squared": lambda m, df, p: m.ChiSquare(data=df),
    "log_likelihood": lambda m, df, p: m.LogLikelihood(data=df),
    "cressie_read": lambda m, df, p: m.PowerDivergence(data=df, lambda_="cressie-read"),
    "freeman_tukey": lambda m, df, p: m.PowerDivergence(
        data=df, lambda_="freeman-tuckey"
    ),
    "modified_likelihood": lambda m, df, p: m.ModifiedLogLikelihood(data=df),
    "pearson_correlation": lambda m, df, p: m.Pearsonr(data=df),
    "pearson_equivalence": lambda m, df, p: m.PearsonrEquivalence(
        data=df, delta_threshold=p["delta_threshold"]
    ),
    "fisher_z": lambda m, df, p: m.FisherZ(data=df),
}


def should_skip(case: dict[str, Any]) -> str | None:
    """Return the single declared pgmpy parity exception, if applicable."""
    if case["test"] in DISCRETE_TESTS and case["params"].get("yates") is False:
        return "pgmpy has no Yates-disable option"
    return None


def build_dataframe(case: dict[str, Any], pandas: ModuleType) -> Any:
    """Convert language-neutral fixture columns to a pandas DataFrame."""
    return pandas.DataFrame(
        {name: spec["values"] for name, spec in case["columns"].items()}
    )


def _fields_match(expected: Any, actual: Any, tolerance: float) -> bool:
    if (
        isinstance(expected, numbers.Real)
        and isinstance(actual, numbers.Real)
        and math.isfinite(float(expected))
        and math.isfinite(float(actual))
    ):
        return math.isclose(
            float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance
        )
    return bool(expected == actual)


def compare_fields(
    case_id: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    tolerance: float,
) -> list[str]:
    """Compare all result fields, except a dof unavailable from pgmpy."""
    errors: list[str] = []
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if field == "dof" and actual_value is None:
            continue
        if not _fields_match(expected_value, actual_value, tolerance):
            errors.append(
                f"{case_id}: {field} expected {expected_value}, got {actual_value}"
            )
    return errors


def classify_intentional_divergence(
    case: dict[str, Any], field: str, actual_value: Any
) -> str | None:
    """Classify current pgmpy's conditioned-equivalence dof propagation."""
    if (
        field == "dof"
        and case["test"] == "pearson_equivalence"
        and bool(case["z"])
        and case["expected"]["dof"] is None
        and isinstance(actual_value, numbers.Real)
        and not isinstance(actual_value, bool)
    ):
        return (
            f"{case['id']}: dof expected None, got {actual_value} "
            "(current pgmpy propagates conditioned Pearsonr dof)"
        )
    return None


def compare_actual_fields(
    case: dict[str, Any], actual: dict[str, Any], tolerance: float
) -> tuple[list[str], list[str]]:
    """Separate strict field mismatches from declared pgmpy divergences."""
    errors: list[str] = []
    divergences: list[str] = []
    for field, expected_value in case["expected"].items():
        field_errors = compare_fields(
            case["id"],
            {field: expected_value},
            {field: actual[field]},
            tolerance,
        )
        if not field_errors:
            continue
        divergence = classify_intentional_divergence(case, field, actual[field])
        if divergence is None:
            errors.extend(field_errors)
        else:
            divergences.append(divergence)
    return errors, divergences


def _compare_case_result(
    case: dict[str, Any], pgmpy_module: ModuleType, tolerance: float
) -> tuple[list[str], list[str]]:
    data = build_dataframe(case, importlib.import_module("pandas"))
    test = CONSTRUCTORS[case["test"]](pgmpy_module, data, case["params"])
    test.run_test(X=case["x"], Y=case["y"], Z=case["z"])
    actual = {
        "statistic": test.statistic_,
        "p_value": test.p_value_,
        "dof": getattr(test, "dof_", None),
        "effect_size": test.effect_size_,
    }
    return compare_actual_fields(case, actual, tolerance)


def compare_case(
    case: dict[str, Any], pgmpy_module: ModuleType, tolerance: float
) -> list[str]:
    """Run one fixture case through pgmpy and compare every available field."""
    errors, _ = _compare_case_result(case, pgmpy_module, tolerance)
    return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgmpy-source",
        type=Path,
        help="repository root to import pgmpy from instead of the environment",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.pgmpy_source is not None:
        sys.path.insert(0, str(args.pgmpy_source.resolve()))

    pgmpy_module = importlib.import_module("pgmpy.ci_tests")
    with DEFAULT_GOLDEN.open(encoding="utf-8") as fixture:
        cases = json.load(fixture)

    compared = 0
    skipped_reasons: Counter[str] = Counter()
    errors: list[str] = []
    divergences: list[str] = []
    for case in cases:
        reason = should_skip(case)
        if reason is not None:
            skipped_reasons[reason] += 1
            continue
        compared += 1
        case_errors, case_divergences = _compare_case_result(
            case, pgmpy_module, args.tolerance
        )
        errors.extend(case_errors)
        divergences.extend(case_divergences)

    for reason, count in skipped_reasons.items():
        print(f"skipped {count}: {reason}")
    for error in errors:
        print(error)
    for divergence in divergences:
        print(f"intentional divergence: {divergence}")
    print(
        f"compared={compared} skipped={sum(skipped_reasons.values())} "
        f"divergences={len(divergences)} failed={len(errors)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
