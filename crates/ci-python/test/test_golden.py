"""Golden parity test for the data-bound Python bindings.

Loads the shared cross-language fixture ``tests/fixtures/golden.json`` and, for
each of the seven tests, builds a :class:`Dataset` from the case's columns,
constructs the test with the case's parameters, runs it, and asserts that
``statistic`` / ``p_value`` / ``dof`` match the recorded ``expected`` values.
This is the binding's numeric parity gate against the scipy/pgmpy reference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ci_python import (
    ChiSquared,
    CressieRead,
    Dataset,
    FreemanTukey,
    LogLikelihood,
    ModifiedLikelihood,
    PearsonCorrelation,
    PearsonEquivalence,
)

# Locate the repo-root fixture relative to this file:
# crates/ci-python/test/test_golden.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden.json"

TOL = 1e-7
EXPECTED_CASE_COUNT = 73

# Map the fixture's stable test name to its binding class.
TEST_CLASSES = {
    "chi_squared": ChiSquared,
    "log_likelihood": LogLikelihood,
    "cressie_read": CressieRead,
    "freeman_tukey": FreemanTukey,
    "modified_likelihood": ModifiedLikelihood,
    "pearson_correlation": PearsonCorrelation,
    "pearson_equivalence": PearsonEquivalence,
}


def _load_cases() -> list[dict[str, Any]]:
    with GOLDEN_PATH.open() as fh:
        return json.load(fh)


def _build_dataset(columns: dict[str, dict[str, Any]]) -> Dataset:
    spec = {
        name: (col["kind"], np.asarray(col["values"], dtype=np.float64))
        for name, col in columns.items()
    }
    return Dataset(spec)


def _construct(name: str, data: Dataset, params: dict[str, Any]) -> Any:
    cls = TEST_CLASSES[name]
    if name == "pearson_correlation":
        return cls(data)
    if name == "pearson_equivalence":
        return cls(data, delta_threshold=params["delta_threshold"])
    # Discrete power-divergence family: configured by `yates`.
    return cls(data, yates=params["yates"])


def _assert_close(actual: float | None, expected: Any, field: str, case_id: str) -> None:
    if expected is None:
        # Null fields are skipped (the test does not define them).
        return
    assert actual is not None, f"{case_id}: expected {field}={expected}, got None"
    exp = float(expected)
    if math.isinf(exp) or math.isnan(exp):
        # Defensive handling for inf statistic / nan (none in the current fixture).
        assert math.isinf(actual) == math.isinf(exp), f"{case_id}: {field} inf mismatch"
        assert math.isnan(actual) == math.isnan(exp), f"{case_id}: {field} nan mismatch"
        if math.isinf(exp):
            assert math.copysign(1.0, actual) == math.copysign(1.0, exp), (
                f"{case_id}: {field} inf sign mismatch"
            )
        return
    assert actual == pytest.approx(exp, abs=TOL, rel=0.0), (
        f"{case_id}: {field} mismatch: got {actual!r}, expected {exp!r}"
    )


GOLDEN_CASES = _load_cases()


@pytest.mark.parametrize(
    ("index", "case"),
    list(enumerate(GOLDEN_CASES)),
    ids=[f"{i:02d}-{c['test']}" for i, c in enumerate(GOLDEN_CASES)],
)
def test_golden_case(index: int, case: dict[str, Any]) -> None:
    """Each fixture case reproduces its recorded statistic / p_value / dof."""
    case_id = f"case[{index}] {case['test']}"
    data = _build_dataset(case["columns"])
    test = _construct(case["test"], data, case["params"])

    result = test.run_test(case["x"], case["y"], case["z"])
    expected = case["expected"]

    _assert_close(result.statistic, expected.get("statistic"), "statistic", case_id)

    # p-value: handle exact-zero / inf defensively, otherwise compare within TOL.
    exp_p = float(expected["p_value"])
    if exp_p == 0.0:
        assert result.p_value == pytest.approx(0.0, abs=TOL), (
            f"{case_id}: p_value expected ~0, got {result.p_value!r}"
        )
    else:
        _assert_close(result.p_value, exp_p, "p_value", case_id)

    _assert_close(
        None if result.dof is None else float(result.dof),
        expected.get("dof"),
        "dof",
        case_id,
    )


def test_golden_covers_all_cases() -> None:
    """The fixture has exactly the expected number of cases, all dispatched."""
    assert len(GOLDEN_CASES) == EXPECTED_CASE_COUNT, (
        f"expected {EXPECTED_CASE_COUNT} cases, found {len(GOLDEN_CASES)}"
    )
    names = {c["test"] for c in GOLDEN_CASES}
    assert names == set(TEST_CLASSES), (
        f"fixture tests {names} do not match binding classes {set(TEST_CLASSES)}"
    )
