# Golden Fixture Parity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a deterministic 80-case SciPy/pgmpy parity fixture, verify it independently in CI, and make every binding assert the complete result contract.

**Architecture:** A standalone Python reference generator owns fixed input scenarios and computes expected statistics with NumPy/SciPy, never with Rust. It commits a canonical JSON artifact consumed read-only by Rust, Python, R, and JavaScript; a separate CI job regenerates the artifact in check mode to detect drift.

**Tech Stack:** Python 3.12, NumPy 2.4.2, SciPy 1.17.0, pytest, Ruff, Rust/serde, PyO3/pytest, extendr/testthat, wasm-bindgen/Vitest, GitHub Actions.

## Global Constraints

- Preserve the existing public Rust, Python, R, and JavaScript APIs.
- Emit exactly 80 cases: 12 for each of five discrete tests, 7 Pearson correlation, 6 Pearson equivalence, and 7 Fisher-Z.
- Compute reference values without importing `ci_core` or any binding.
- Keep every JSON number finite; error, NaN, and infinity behavior stays in unit tests.
- Give every case a unique lowercase kebab-case `id` and report it in binding failures.
- Canonicalize computed floats to 12 significant digits and serialize strict JSON with sorted keys, two-space indentation, and a trailing newline.
- Compare binding results with absolute tolerance `1e-7` and zero relative tolerance.
- Pin fixture generation to `numpy==2.4.2` and `scipy==1.17.0`.
- Normal binding tests consume the committed JSON and must not require NumPy or SciPy solely for reference generation.
- Do not restore or add benchmarks in this change.

---

## File Structure

- `tests/fixtures/generate_golden.py`: deterministic scenario catalog, independent reference formulas, validation, canonical serialization, atomic write, and `--check` CLI.
- `tests/fixtures/test_generate_golden.py`: focused tests for the generator contract, determinism, strict serialization, and drift behavior.
- `tests/fixtures/requirements.txt`: exact NumPy/SciPy oracle versions.
- `tests/fixtures/golden.json`: generated 80-case cross-language contract.
- `crates/ci-core/tests/golden.rs`: Rust consumer; deserialize and report stable IDs.
- `crates/ci-python/test/test_golden.py`: Python consumer; report IDs and assert effect size.
- `crates/ci-js/tests/golden.test.js`: JavaScript consumer; report IDs and assert effect size.
- `crates/ci-r/tests/testthat/test-golden.R`: R consumer; report IDs and assert effect size.
- `crates/ci-r/DESCRIPTION`: declare `jsonlite` as a test dependency.
- `.github/workflows/python.yml`: add the platform-independent fixture drift job and correct case counts.
- `.github/workflows/{rust,r,js}.yml`: correct stale case-count comments.
- `README.md`, `CONTRIBUTING.md`: document generation/checking and remove the nonexistent benchmark-tree entry.

---

### Task 1: Build the independent 80-case reference catalog

**Files:**

- Create: `tests/fixtures/requirements.txt`
- Create: `tests/fixtures/test_generate_golden.py`
- Create: `tests/fixtures/generate_golden.py`

**Interfaces:**

- Consumes: NumPy and SciPy only.
- Produces: `build_cases() -> list[dict[str, Any]]` and `validate_cases(cases: list[dict[str, Any]]) -> None`.

- [ ] **Step 1: Add the pinned oracle dependencies**

Create `tests/fixtures/requirements.txt`:

```text
numpy==2.4.2
scipy==1.17.0
```

Install the focused test tools and pinned references:

```bash
python -m pip install pytest ruff -r tests/fixtures/requirements.txt
```

Expected: pip installs the two pinned reference libraries plus pytest and Ruff without dependency conflicts.

- [ ] **Step 2: Write the failing catalog test**

Create `tests/fixtures/test_generate_golden.py`:

```python
from __future__ import annotations

import importlib.util
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


FIXTURE_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = FIXTURE_DIR / "generate_golden.py"
GOLDEN_PATH = FIXTURE_DIR / "golden.json"

EXPECTED_COUNTS = {
    "chi_squared": 12,
    "log_likelihood": 12,
    "cressie_read": 12,
    "freeman_tukey": 12,
    "modified_likelihood": 12,
    "pearson_correlation": 7,
    "pearson_equivalence": 6,
    "fisher_z": 7,
}


def load_generator() -> ModuleType:
    assert GENERATOR_PATH.exists(), f"missing generator: {GENERATOR_PATH}"
    spec = importlib.util.spec_from_file_location("generate_golden", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def by_id(cases: list[dict], case_id: str) -> dict:
    return next(case for case in cases if case["id"] == case_id)


def test_reference_catalog_has_the_approved_contract() -> None:
    generator = load_generator()
    cases = generator.build_cases()
    generator.validate_cases(cases)

    assert len(cases) == 80
    assert Counter(case["test"] for case in cases) == Counter(EXPECTED_COUNTS)
    assert len({case["id"] for case in cases}) == 80

    balanced = by_id(cases, "discrete-balanced-2x2-yates-chi-squared")
    assert balanced["expected"] == {
        "statistic": pytest.approx(0.0, abs=1e-12),
        "p_value": pytest.approx(1.0, abs=1e-12),
        "dof": 1,
        "effect_size": pytest.approx(0.0, abs=1e-12),
    }

    positive = by_id(cases, "pearson-correlation-positive-unconditional")
    assert positive["expected"]["statistic"] == pytest.approx(0.8, abs=1e-10)
    assert positive["expected"]["effect_size"] == pytest.approx(0.8, abs=1e-10)
    assert positive["expected"]["dof"] == 14

    perfect = by_id(cases, "fisher-z-perfect-correlation")
    assert perfect["expected"]["statistic"] > 0.0
    assert perfect["expected"]["effect_size"] == pytest.approx(1.0, abs=1e-10)
    assert perfect["expected"]["dof"] is None


def test_validation_rejects_duplicate_ids() -> None:
    generator = load_generator()
    cases = deepcopy(generator.build_cases())
    cases[1]["id"] = cases[0]["id"]

    with pytest.raises(ValueError, match="duplicate case id"):
        generator.validate_cases(cases)


def test_validation_rejects_nonfinite_results() -> None:
    generator = load_generator()
    cases = deepcopy(generator.build_cases())
    cases[0]["expected"]["statistic"] = float("inf")

    with pytest.raises(ValueError, match="statistic must be finite"):
        generator.validate_cases(cases)


def test_validation_rejects_mismatched_column_lengths() -> None:
    generator = load_generator()
    cases = deepcopy(generator.build_cases())
    cases[0]["columns"]["X"]["values"].pop()

    with pytest.raises(ValueError, match="column lengths differ"):
        generator.validate_cases(cases)
```

- [ ] **Step 3: Run the test and verify the missing generator is the failure**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py::test_reference_catalog_has_the_approved_contract -v
```

Expected: FAIL at `assert GENERATOR_PATH.exists()` with `missing generator`, proving the test covers the missing pipeline.

- [ ] **Step 4: Implement the deterministic catalog and reference formulas**

Create `tests/fixtures/generate_golden.py` with this catalog/oracle layer. Serialization and CLI functions are deliberately deferred to Task 2 so their tests can be written first.

```python
#!/usr/bin/env python3
"""Generate the language-neutral CI-test parity fixture from independent references."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
from scipy import stats


DISCRETE_LAMBDAS = {
    "chi_squared": 1.0,
    "log_likelihood": 0.0,
    "cressie_read": 2.0 / 3.0,
    "freeman_tukey": -0.5,
    "modified_likelihood": -1.0,
}

EXPECTED_COUNTS = {
    "chi_squared": 12,
    "log_likelihood": 12,
    "cressie_read": 12,
    "freeman_tukey": 12,
    "modified_likelihood": 12,
    "pearson_correlation": 7,
    "pearson_equivalence": 6,
    "fisher_z": 7,
}

CASE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
EXPECTED_KEYS = {"statistic", "p_value", "dof", "effect_size"}


def _column(kind: str, values: list[int] | list[float]) -> dict[str, Any]:
    return {"kind": kind, "values": values}


def _discrete_columns(
    strata: list[tuple[tuple[int, ...], list[int], list[int], list[list[int]]]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    z_width = len(strata[0][0])
    z_names = [f"Z{i + 1}" for i in range(z_width)]
    x_values: list[int] = []
    y_values: list[int] = []
    z_values: list[list[int]] = [[] for _ in range(z_width)]

    for z_key, x_levels, y_levels, counts in strata:
        if len(z_key) != z_width or len(counts) != len(x_levels):
            raise ValueError("invalid discrete scenario shape")
        if any(len(row) != len(y_levels) for row in counts):
            raise ValueError("invalid discrete table width")
        for x_index, x_level in enumerate(x_levels):
            for y_index, y_level in enumerate(y_levels):
                count = counts[x_index][y_index]
                if count <= 0:
                    raise ValueError("active contingency cells must be positive")
                x_values.extend([x_level] * count)
                y_values.extend([y_level] * count)
                for z_index, z_level in enumerate(z_key):
                    z_values[z_index].extend([z_level] * count)

    columns = {
        "X": _column("discrete", x_values),
        "Y": _column("discrete", y_values),
    }
    columns.update(
        {
            name: _column("discrete", values)
            for name, values in zip(z_names, z_values, strict=True)
        }
    )
    return columns, z_names


def _discrete_scenario(
    scenario_id: str,
    yates: bool,
    strata: list[tuple[tuple[int, ...], list[int], list[int], list[list[int]]]],
) -> dict[str, Any]:
    columns, z_names = _discrete_columns(strata)
    return {
        "scenario_id": scenario_id,
        "yates": yates,
        "columns": columns,
        "x": "X",
        "y": "Y",
        "z": z_names,
    }


def _discrete_scenarios() -> list[dict[str, Any]]:
    associated = [((), [0, 1], [0, 1], [[12, 3], [4, 11]])]
    unbalanced = [((), [0, 1], [0, 1], [[18, 4], [7, 9]])]
    return [
        _discrete_scenario(
            "discrete-balanced-2x2-yates",
            True,
            [((), [0, 1], [0, 1], [[8, 8], [8, 8]])],
        ),
        _discrete_scenario("discrete-associated-2x2-yates", True, associated),
        _discrete_scenario("discrete-associated-2x2-no-yates", False, associated),
        _discrete_scenario("discrete-unbalanced-2x2-yates", True, unbalanced),
        _discrete_scenario("discrete-unbalanced-2x2-no-yates", False, unbalanced),
        _discrete_scenario(
            "discrete-2x3",
            True,
            [((), [0, 1], [0, 1, 2], [[9, 4, 7], [3, 8, 5]])],
        ),
        _discrete_scenario(
            "discrete-3x2",
            True,
            [((), [0, 1, 2], [0, 1], [[8, 3], [4, 9], [6, 5]])],
        ),
        _discrete_scenario(
            "discrete-3x3",
            True,
            [((), [0, 1, 2], [0, 1, 2], [[8, 3, 4], [5, 9, 2], [4, 6, 10]])],
        ),
        _discrete_scenario(
            "discrete-conditioned-binary",
            True,
            [
                ((0,), [0, 1], [0, 1], [[8, 3], [4, 7]]),
                ((1,), [0, 1], [0, 1], [[5, 6], [7, 4]]),
            ],
        ),
        _discrete_scenario(
            "discrete-conditioned-three-level",
            False,
            [
                ((0,), [0, 1], [0, 1], [[7, 2], [3, 6]]),
                ((1,), [0, 1], [0, 1], [[4, 5], [6, 3]]),
                ((2,), [0, 1], [0, 1], [[5, 3], [2, 8]]),
            ],
        ),
        _discrete_scenario(
            "discrete-conditioned-two-columns",
            True,
            [
                ((0, 0), [0, 1], [0, 1], [[5, 2], [3, 6]]),
                ((0, 1), [0, 1], [0, 1], [[4, 3], [5, 2]]),
                ((1, 0), [0, 1], [0, 1], [[6, 4], [2, 5]]),
                ((1, 1), [0, 1], [0, 1], [[3, 5], [6, 4]]),
            ],
        ),
        _discrete_scenario(
            "discrete-sparse-active-levels",
            True,
            [
                ((0,), [0, 1], [0, 1], [[4, 2], [3, 5]]),
                ((1,), [1, 2], [1, 2], [[2, 4], [5, 3]]),
            ],
        ),
    ]


def _strata_indices(case: dict[str, Any]) -> list[np.ndarray]:
    row_count = len(case["columns"][case["x"]]["values"])
    if not case["z"]:
        return [np.arange(row_count)]

    groups: dict[tuple[float, ...], list[int]] = {}
    for row in range(row_count):
        key = tuple(case["columns"][name]["values"][row] for name in case["z"])
        groups.setdefault(key, []).append(row)
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def _discrete_expected(case: dict[str, Any], lambda_: float) -> dict[str, Any]:
    x = np.asarray(case["columns"][case["x"]]["values"])
    y = np.asarray(case["columns"][case["y"]]["values"])
    statistic = 0.0
    dof = 0

    for indices in _strata_indices(case):
        x_active = np.unique(x[indices])
        y_active = np.unique(y[indices])
        observed = np.zeros((len(x_active), len(y_active)), dtype=np.float64)
        x_index = {value: index for index, value in enumerate(x_active)}
        y_index = {value: index for index, value in enumerate(y_active)}
        for row in indices:
            observed[x_index[x[row]], y_index[y[row]]] += 1.0

        result = stats.chi2_contingency(
            observed,
            correction=case["yates"],
            lambda_=lambda_,
        )
        statistic += float(result.statistic)
        dof += int(result.dof)

    p_value = float(stats.chi2.sf(statistic, dof))
    cardinality = min(len(np.unique(x)), len(np.unique(y)))
    effect_size = math.sqrt(statistic / (len(x) * max(cardinality - 1, 1)))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "dof": dof,
        "effect_size": effect_size,
    }


U = np.asarray(
    [
        -0.40674460841,
        -0.352511993955,
        -0.298279379501,
        -0.244046765046,
        -0.189814150591,
        -0.135581536137,
        -0.081348921682,
        -0.0271163072273,
        0.0271163072273,
        0.081348921682,
        0.135581536137,
        0.189814150591,
        0.244046765046,
        0.298279379501,
        0.352511993955,
        0.40674460841,
    ]
)
V = np.asarray(
    [
        0.463099108522,
        0.277859465113,
        0.119082627906,
        -0.0132314031006,
        -0.119082627906,
        -0.198471046509,
        -0.251396658912,
        -0.277859465113,
        -0.277859465113,
        -0.251396658912,
        -0.198471046509,
        -0.119082627906,
        -0.0132314031006,
        0.119082627906,
        0.277859465113,
        0.463099108522,
    ]
)
W = np.asarray(
    [
        -0.453244808633,
        -0.0906489617267,
        0.142448368428,
        0.265970030561,
        0.299838873404,
        0.263977745688,
        0.178309496144,
        0.0627569735031,
        -0.0627569735031,
        -0.178309496144,
        -0.263977745688,
        -0.299838873404,
        -0.265970030561,
        -0.142448368428,
        0.0906489617267,
        0.453244808633,
    ]
)
T = np.asarray(
    [
        0.398089477628,
        -0.132696492543,
        -0.322262910461,
        -0.293098846166,
        -0.14727852469,
        0.0335386739394,
        0.188108214703,
        0.275600407589,
        0.275600407589,
        0.188108214703,
        0.0335386739394,
        -0.14727852469,
        -0.293098846166,
        -0.322262910461,
        -0.132696492543,
        0.398089477628,
    ]
)


def _linear(*terms: tuple[float, np.ndarray], offset: float = 0.0) -> np.ndarray:
    values = np.full(U.shape, offset, dtype=np.float64)
    for coefficient, vector in terms:
        values += coefficient * vector
    return values


def _continuous_scenario(
    scenario_id: str,
    x: np.ndarray,
    y: np.ndarray,
    z: tuple[np.ndarray, ...] = (),
) -> dict[str, Any]:
    columns = {
        "X": _column("continuous", x.astype(float).tolist()),
        "Y": _column("continuous", y.astype(float).tolist()),
    }
    z_names = []
    for index, values in enumerate(z, start=1):
        name = f"Z{index}"
        columns[name] = _column("continuous", values.astype(float).tolist())
        z_names.append(name)
    return {
        "scenario_id": scenario_id,
        "columns": columns,
        "x": "X",
        "y": "Y",
        "z": z_names,
    }


def _pearson_scenarios() -> list[dict[str, Any]]:
    return [
        _continuous_scenario("orthogonal-unconditional", U, V),
        _continuous_scenario("positive-unconditional", U, _linear((0.8, U), (0.6, V))),
        _continuous_scenario("negative-unconditional", U, _linear((-0.6, U), (0.8, V))),
        _continuous_scenario(
            "confounding-removed",
            _linear((1.2, U), (0.6, V)),
            _linear((-0.9, U), (0.8, W)),
            (U,),
        ),
        _continuous_scenario(
            "residual-association",
            _linear((1.1, U), (1.0, V)),
            _linear((-0.7, U), (0.7, V), (0.3, W)),
            (U,),
        ),
        _continuous_scenario(
            "two-conditioning-columns",
            _linear((0.9, U), (-0.4, V), (0.8, W), (0.2, T)),
            _linear((-0.5, U), (0.7, V), (-0.3, W), (0.9, T)),
            (U, V),
        ),
        _continuous_scenario(
            "scale-offset-invariance",
            _linear((7.0, U), offset=10.0),
            _linear((2.0, V), offset=-3.0),
        ),
    ]


def _equivalence_scenarios() -> list[tuple[dict[str, Any], float]]:
    return [
        (_continuous_scenario("inside-orthogonal", U, V), 0.1),
        (
            _continuous_scenario(
                "inside-weak-positive",
                U,
                _linear((0.08, U), (math.sqrt(1.0 - 0.08**2), V)),
            ),
            0.2,
        ),
        (
            _continuous_scenario(
                "outside-positive",
                U,
                _linear((0.65, U), (math.sqrt(1.0 - 0.65**2), V)),
            ),
            0.1,
        ),
        (
            _continuous_scenario(
                "outside-negative",
                U,
                _linear((-0.55, U), (math.sqrt(1.0 - 0.55**2), V)),
            ),
            0.15,
        ),
        (
            _continuous_scenario(
                "inside-conditioned",
                _linear((1.2, U), (1.0, V)),
                _linear((-0.8, U), (0.05, V), (1.0, W)),
                (U,),
            ),
            0.15,
        ),
        (
            _continuous_scenario(
                "outside-two-conditioning-columns",
                _linear((0.3, U), (-0.4, V), (0.8, W), (0.2, T)),
                _linear((-0.6, U), (0.5, V), (0.1, W), (0.9, T)),
                (U, V),
            ),
            0.2,
        ),
    ]


def _fisher_scenarios() -> list[dict[str, Any]]:
    scenarios = _pearson_scenarios()[:6]
    scenarios.append(_continuous_scenario("perfect-correlation", U, _linear((3.0, U), offset=2.0)))
    return scenarios


def _partial_correlation(case: dict[str, Any]) -> tuple[float, int]:
    x = np.asarray(case["columns"][case["x"]]["values"], dtype=np.float64)
    y = np.asarray(case["columns"][case["y"]]["values"], dtype=np.float64)
    if not case["z"]:
        correlation = float(stats.pearsonr(x, y).statistic)
    else:
        z_columns = [
            np.asarray(case["columns"][name]["values"], dtype=np.float64)
            for name in case["z"]
        ]
        design = np.column_stack([np.ones(len(x)), *z_columns])
        x_beta = np.linalg.lstsq(design, x, rcond=None)[0]
        y_beta = np.linalg.lstsq(design, y, rcond=None)[0]
        x_residual = x - design @ x_beta
        y_residual = y - design @ y_beta
        correlation = float(np.corrcoef(x_residual, y_residual)[0, 1])
    return correlation, len(x) - len(case["z"]) - 2


def _pearson_expected(case: dict[str, Any]) -> dict[str, Any]:
    correlation, dof = _partial_correlation(case)
    t_statistic = correlation * math.sqrt(dof / (1.0 - correlation**2))
    return {
        "statistic": correlation,
        "p_value": float(2.0 * stats.t.sf(abs(t_statistic), df=dof)),
        "dof": dof,
        "effect_size": abs(correlation),
    }


def _equivalence_expected(case: dict[str, Any], delta: float) -> dict[str, Any]:
    correlation, _ = _partial_correlation(case)
    rho = float(np.clip(correlation, -0.999999, 0.999999))
    coefficient = math.atanh(rho)
    z_delta = math.atanh(delta)
    scale = math.sqrt(len(case["columns"][case["x"]]["values"]) - len(case["z"]) - 3)
    p_lower = 1.0 - float(stats.norm.cdf(scale * (coefficient + z_delta)))
    p_upper = float(stats.norm.cdf(scale * (coefficient - z_delta)))
    return {
        "statistic": coefficient,
        "p_value": max(p_lower, p_upper),
        "dof": None,
        "effect_size": abs(correlation),
    }


def _fisher_expected(case: dict[str, Any]) -> dict[str, Any]:
    correlation, _ = _partial_correlation(case)
    rho = float(np.clip(correlation, -0.999999, 0.999999))
    scale = math.sqrt(len(case["columns"][case["x"]]["values"]) - len(case["z"]) - 3)
    statistic = scale * math.atanh(rho)
    return {
        "statistic": statistic,
        "p_value": float(2.0 * stats.norm.sf(abs(statistic))),
        "dof": None,
        "effect_size": abs(correlation),
    }


def _case(
    case_id: str,
    test: str,
    params: dict[str, Any],
    scenario: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": case_id,
        "test": test,
        "params": params,
        "columns": scenario["columns"],
        "x": scenario["x"],
        "y": scenario["y"],
        "z": scenario["z"],
        "expected": expected,
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for scenario in _discrete_scenarios():
        for test, lambda_ in DISCRETE_LAMBDAS.items():
            cases.append(
                _case(
                    f"{scenario['scenario_id']}-{test.replace('_', '-')}",
                    test,
                    {"yates": scenario["yates"]},
                    scenario,
                    _discrete_expected(scenario, lambda_),
                )
            )

    for scenario in _pearson_scenarios():
        cases.append(
            _case(
                f"pearson-correlation-{scenario['scenario_id']}",
                "pearson_correlation",
                {},
                scenario,
                _pearson_expected(scenario),
            )
        )

    for scenario, delta in _equivalence_scenarios():
        cases.append(
            _case(
                f"pearson-equivalence-{scenario['scenario_id']}",
                "pearson_equivalence",
                {"delta_threshold": delta},
                scenario,
                _equivalence_expected(scenario, delta),
            )
        )

    for scenario in _fisher_scenarios():
        cases.append(
            _case(
                f"fisher-z-{scenario['scenario_id']}",
                "fisher_z",
                {},
                scenario,
                _fisher_expected(scenario),
            )
        )
    return cases


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_cases(cases: list[dict[str, Any]]) -> None:
    counts = Counter(case.get("test") for case in cases)
    if len(cases) != 80 or counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"expected the approved 80-case distribution, got {dict(counts)}")

    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or CASE_ID.fullmatch(case_id) is None:
            raise ValueError(f"invalid case id: {case_id!r}")
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)

        test = case["test"]
        params = case["params"]
        if test in DISCRETE_LAMBDAS:
            if set(params) != {"yates"} or not isinstance(params["yates"], bool):
                raise ValueError(f"{case_id}: invalid discrete params")
        elif test == "pearson_equivalence":
            if set(params) != {"delta_threshold"} or not _finite_number(params["delta_threshold"]):
                raise ValueError(f"{case_id}: invalid equivalence params")
            if not 0.0 < params["delta_threshold"] < 1.0:
                raise ValueError(f"{case_id}: delta_threshold must be in (0, 1)")
        elif test in {"pearson_correlation", "fisher_z"}:
            if params:
                raise ValueError(f"{case_id}: unexpected params")
        else:
            raise ValueError(f"{case_id}: unknown test {test!r}")

        columns = case["columns"]
        if not isinstance(columns, dict) or not columns:
            raise ValueError(f"{case_id}: columns must be a non-empty object")
        lengths: set[int] = set()
        for name, column in columns.items():
            if not isinstance(name, str) or column.get("kind") not in {"discrete", "continuous"}:
                raise ValueError(f"{case_id}: invalid column {name!r}")
            values = column.get("values")
            if not isinstance(values, list) or not values or not all(_finite_number(v) for v in values):
                raise ValueError(f"{case_id}: invalid values for column {name}")
            lengths.add(len(values))
        if len(lengths) != 1:
            raise ValueError(f"{case_id}: column lengths differ")

        query = [case["x"], case["y"], *case["z"]]
        if len(query) != len(set(query)) or any(name not in columns for name in query):
            raise ValueError(f"{case_id}: invalid query columns {query}")

        expected = case["expected"]
        if set(expected) != EXPECTED_KEYS:
            raise ValueError(f"{case_id}: result keys must be {sorted(EXPECTED_KEYS)}")
        for field in ("statistic", "p_value", "effect_size"):
            if not _finite_number(expected[field]):
                raise ValueError(f"{case_id}: {field} must be finite")
        dof = expected["dof"]
        if dof is not None and (isinstance(dof, bool) or not isinstance(dof, int) or dof < 0):
            raise ValueError(f"{case_id}: invalid dof {dof!r}")
        if not 0.0 <= expected["p_value"] <= 1.0 or expected["effect_size"] < 0.0:
            raise ValueError(f"{case_id}: invalid probability or effect size")
```

- [ ] **Step 5: Format, lint, and run the catalog tests**

Run:

```bash
ruff format tests/fixtures
ruff check tests/fixtures
pytest tests/fixtures/test_generate_golden.py -v
```

Expected: Ruff succeeds; four pytest tests pass with no SciPy warnings and no import of a project binding.

- [ ] **Step 6: Commit the independent oracle**

```bash
git add tests/fixtures/requirements.txt tests/fixtures/test_generate_golden.py tests/fixtures/generate_golden.py
git commit -m "test: add independent golden reference catalog"
```

---

### Task 2: Add canonical serialization, drift checking, and the committed fixture

**Files:**

- Modify: `tests/fixtures/test_generate_golden.py`
- Modify: `tests/fixtures/generate_golden.py`
- Create: `tests/fixtures/golden.json` (generated, never hand-edited)

**Interfaces:**

- Consumes: `build_cases()` and `validate_cases()` from Task 1.
- Produces: `render_cases(cases) -> str`, `write_fixture(path, rendered) -> None`, `check_fixture(path, rendered) -> bool`, and `main(argv=None) -> int`.

- [ ] **Step 1: Write failing tests for canonical output and non-mutating check mode**

Add `import json` to `tests/fixtures/test_generate_golden.py`, then append:

```python
def test_render_cases_is_strict_and_deterministic() -> None:
    generator = load_generator()
    cases = generator.build_cases()

    first = generator.render_cases(cases)
    second = generator.render_cases(cases)

    assert first == second
    assert first.endswith("\n")
    assert "NaN" not in first
    assert "Infinity" not in first
    decoded = json.loads(first)
    assert len(decoded) == 80
    positive = by_id(decoded, "pearson-correlation-positive-unconditional")
    assert positive["expected"]["statistic"] == 0.8


def test_write_and_check_fixture_are_separate_operations(tmp_path: Path) -> None:
    generator = load_generator()
    rendered = generator.render_cases(generator.build_cases())
    output = tmp_path / "nested" / "golden.json"

    generator.write_fixture(output, rendered)
    assert output.read_text(encoding="utf-8") == rendered
    assert generator.check_fixture(output, rendered) is True

    changed = rendered.replace('"test": "chi_squared"', '"test": "changed"', 1)
    output.write_text(changed, encoding="utf-8")
    assert generator.check_fixture(output, rendered) is False
    assert output.read_text(encoding="utf-8") == changed


def test_cli_check_mode_never_creates_or_rewrites_output(tmp_path: Path) -> None:
    generator = load_generator()
    output = tmp_path / "golden.json"

    assert generator.main(["--check", "--output", str(output)]) == 1
    assert not output.exists()

    assert generator.main(["--output", str(output)]) == 0
    original = output.read_text(encoding="utf-8")
    assert generator.main(["--check", "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == original

    output.write_text(f"{original}\n", encoding="utf-8")
    changed = output.read_text(encoding="utf-8")
    assert generator.main(["--check", "--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == changed
```

- [ ] **Step 2: Run the new tests and verify the expected missing-interface failure**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py -k "render or write or cli" -v
```

Expected: FAIL with `AttributeError: module 'generate_golden' has no attribute 'render_cases'`.

- [ ] **Step 3: Implement canonical rendering, atomic writes, and the CLI**

Replace the standard-library import block near the top of
`tests/fixtures/generate_golden.py` with the complete final block:

```python
import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
```

Add the default output constant after `EXPECTED_KEYS`:

```python
DEFAULT_OUTPUT = Path(__file__).with_name("golden.json")
```

Append these complete functions:

```python
def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        rounded = float(f"{value:.12g}")
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    return value


def render_cases(cases: list[dict[str, Any]]) -> str:
    validate_cases(cases)
    canonical = _canonicalize(cases)
    return json.dumps(canonical, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_fixture(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def check_fixture(path: Path, rendered: str) -> bool:
    return path.exists() and path.read_text(encoding="utf-8") == rendered


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixture without writing it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered = render_cases(build_cases())
    if args.check:
        if check_fixture(args.output, rendered):
            print(f"golden fixture is current: {args.output}")
            return 0
        print(
            f"golden fixture is missing or stale: {args.output}\n"
            "regenerate it with: python tests/fixtures/generate_golden.py",
            file=sys.stderr,
        )
        return 1

    write_fixture(args.output, rendered)
    print(f"wrote 80 golden cases: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused serialization tests and verify they pass**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py -k "render or write or cli" -v
```

Expected: PASS for all three selected tests. The `--check` calls leave the temporary file absent or byte-for-byte unchanged.

- [ ] **Step 5: Add the failing committed-artifact drift test**

Append to `tests/fixtures/test_generate_golden.py`:

```python
def test_committed_fixture_matches_generator() -> None:
    generator = load_generator()
    rendered = generator.render_cases(generator.build_cases())
    assert GOLDEN_PATH.exists(), "golden.json must be committed"
    assert generator.check_fixture(GOLDEN_PATH, rendered), (
        "golden.json is stale; run python tests/fixtures/generate_golden.py"
    )
```

- [ ] **Step 6: Run the drift test and verify the artifact is still missing**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py::test_committed_fixture_matches_generator -v
```

Expected: FAIL with `golden.json must be committed`.

- [ ] **Step 7: Generate the fixture and rerun the complete generator suite**

Run:

```bash
python tests/fixtures/generate_golden.py
pytest tests/fixtures/test_generate_golden.py -v
python tests/fixtures/generate_golden.py --check
```

Expected: the first command reports `wrote 80 golden cases`; all eight generator tests pass; check mode reports `golden fixture is current`.

- [ ] **Step 8: Prove regeneration is byte-stable**

Run:

```bash
sha256sum tests/fixtures/golden.json
python tests/fixtures/generate_golden.py
sha256sum tests/fixtures/golden.json
```

Expected: both SHA-256 values are identical.

- [ ] **Step 9: Re-run formatting, linting, tests, and drift checking**

Run:

```bash
ruff format tests/fixtures
ruff check tests/fixtures
pytest tests/fixtures/test_generate_golden.py -v
python tests/fixtures/generate_golden.py --check
```

Expected: Ruff succeeds, all eight tests pass, and check mode still reports byte-identical output.

- [ ] **Step 10: Commit the generated contract**

```bash
git add tests/fixtures/generate_golden.py tests/fixtures/test_generate_golden.py tests/fixtures/golden.json
git commit -m "test: add reproducible golden parity fixture"
```

---

### Task 3: Make the Rust acceptance gate consume stable case IDs

**Files:**

- Modify: `crates/ci-core/tests/golden.rs:48-62,139-163,165-210`

**Interfaces:**

- Consumes: the `id` and `expected` fields in `tests/fixtures/golden.json`.
- Produces: the existing `golden_fixture_matches` integration test with stable case IDs in every numeric mismatch.

- [ ] **Step 1: Add a compile-failing assertion that requires the fixture ID**

At the start of the `for case in &cases` loop in `golden_fixture_matches`, before `run_case`, add:

```rust
assert!(!case.id.is_empty(), "fixture cases must have a stable id");
```

- [ ] **Step 2: Run the Rust golden target and verify the new contract fails to compile**

Run:

```bash
cargo test -p ci_core --test golden
```

Expected: FAIL with `no field 'id' on type '&Case'`. This proves the consumer does not yet deserialize the new contract field.

- [ ] **Step 3: Deserialize the ID and use it in field failures**

Add `id` to `Case`:

```rust
#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    test: String,
    #[serde(default)]
    params: Params,
    columns: BTreeMap<String, ColumnSpec>,
    x: String,
    y: String,
    z: Vec<String>,
    expected: Expected,
}
```

Replace `assert_field` with:

```rust
fn assert_field(name: &str, case: &Case, expected: Option<f64>, actual: Option<f64>) {
    let Some(exp) = expected else { return };
    let act = actual.unwrap_or_else(|| {
        panic!(
            "[{}:{}] expected {name}={exp} but result had None",
            case.id, case.test
        )
    });

    if exp.is_infinite() || act.is_infinite() {
        assert!(
            exp.total_cmp(&act).is_eq(),
            "[{}:{}] {name} mismatch: expected {exp}, got {act} (x={}, y={}, z={:?})",
            case.id,
            case.test,
            case.x,
            case.y,
            case.z,
        );
        return;
    }

    assert!(
        (act - exp).abs() < TOL,
        "[{}:{}] {name} mismatch: expected {exp}, got {act} (x={}, y={}, z={:?})",
        case.id,
        case.test,
        case.x,
        case.y,
        case.z,
    );
}
```

- [ ] **Step 4: Run the target and complete core suite**

Run:

```bash
cargo test -p ci_core --test golden -- --nocapture
cargo test -p ci_core
```

Expected: the target reports all 80 cases across eight tests and passes; the complete core suite passes with no missing-file error.

- [ ] **Step 5: Commit the Rust consumer update**

```bash
git add crates/ci-core/tests/golden.rs
git commit -m "test(core): report golden fixture case ids"
```

---

### Task 4: Complete the Python four-field parity assertion

**Files:**

- Modify: `crates/ci-python/test/test_golden.py:1-10,96-130`

**Interfaces:**

- Consumes: each fixture case's `id` and `expected.effect_size`.
- Produces: 80 parameterized Python binding checks for statistic, p-value, dof, and effect size.

- [ ] **Step 1: Tighten the parameterized acceptance test**

Update the module documentation to name all four fields:

```python
"""Golden parity test for the data-bound Python bindings.

Loads the shared cross-language fixture ``tests/fixtures/golden.json`` and, for
each of the eight tests, builds a :class:`Dataset` from the case's columns,
constructs the test with the case's parameters, runs it, and asserts that
``statistic`` / ``p_value`` / ``dof`` / ``effect_size`` match the recorded
``expected`` values. This is the binding's numeric parity gate against the
SciPy/pgmpy reference.
"""
```

Replace the parameterization and function signature with stable IDs:

```python
@pytest.mark.parametrize(
    "case",
    GOLDEN_CASES,
    ids=[case["id"] for case in GOLDEN_CASES],
)
def test_golden_case(case: dict[str, Any]) -> None:
    """Each fixture case reproduces every defined result field."""
    case_id = case["id"]
```

Keep the existing dataset construction and statistic/p-value/dof assertions, then add after the dof assertion:

```python
    _assert_close(
        result.effect_size,
        expected.get("effect_size"),
        "effect_size",
        case_id,
    )
```

- [ ] **Step 2: Build the extension and run the golden module**

Run:

```bash
python -m pip install maturin pytest numpy mypy
python -m pip install -e crates/ci-python --no-build-isolation
pytest crates/ci-python/test/test_golden.py -v
```

Expected: 81 tests pass: 80 named fixture cases plus the coverage/count test.

- [ ] **Step 3: Run the complete Python suite and type checker**

Run:

```bash
pytest crates/ci-python/test/
cd crates/ci-python
mypy test/
```

Expected: all Python tests and mypy checks pass.

- [ ] **Step 4: Commit the Python contract check**

```bash
git add crates/ci-python/test/test_golden.py
git commit -m "test(python): assert golden effect sizes"
```

---

### Task 5: Complete the JavaScript four-field parity assertion

**Files:**

- Modify: `crates/ci-js/tests/golden.test.js:1-10,109-154`

**Interfaces:**

- Consumes: each fixture case's `id` and `expected.effect_size`.
- Produces: 80 Vitest binding checks for statistic, p-value, dof, and effect size.

- [ ] **Step 1: Add stable IDs and the missing effect-size assertion**

Change the top-level description to:

```javascript
// Loads the shared cross-language fixture `tests/fixtures/golden.json` (at the
// repository root) and, for each of the eight tests, builds a `Dataset` from the
// case's columns, constructs the test with the case's parameters, runs it, and
// asserts that `statistic` / `pValue` / `dof` / `effectSize` match the recorded
// `expected` values within 1e-7. This is the binding's numeric parity gate
// against the SciPy/pgmpy reference.
```

Inside the parameterized test, replace the index-derived label with:

```javascript
      const caseId = testCase.id;
```

After the existing dof assertion, add:

```javascript
      assertClose(result.effectSize, expected.effect_size, "effectSize", caseId);
```

- [ ] **Step 2: Build the WASM package and run Vitest**

Run:

```bash
wasm-pack build crates/ci-js --target nodejs
cd crates/ci-js/tests
npm ci
npm test
```

Expected: Vitest passes all API tests plus 80 golden cases and the golden coverage/count test.

- [ ] **Step 3: Run JavaScript lint and formatting checks**

Run:

```bash
cd crates/ci-js
npm ci
npx prettier --check .
npx eslint .
```

Expected: all checks pass without rewriting files.

- [ ] **Step 4: Commit the JavaScript contract check**

```bash
git add crates/ci-js/tests/golden.test.js
git commit -m "test(js): assert golden effect sizes"
```

---

### Task 6: Complete the R four-field parity assertion and declare jsonlite

**Files:**

- Modify: `crates/ci-r/tests/testthat/test-golden.R:1-10,115-151`
- Modify: `crates/ci-r/DESCRIPTION:15-19`

**Interfaces:**

- Consumes: each fixture case's `id` and `expected.effect_size` through `jsonlite`.
- Produces: 80 testthat binding checks for statistic, p-value, dof, and effect size; a declared package test dependency.

- [ ] **Step 1: Declare the JSON fixture reader**

Change `Suggests` in `crates/ci-r/DESCRIPTION` to:

```text
Suggests:
    devtools,
    jsonlite,
    rextendr,
    testthat (>= 3.0.0)
```

- [ ] **Step 2: Add stable IDs and the missing effect-size assertion**

Update the file header and test description:

```r
# Loads the shared cross-language fixture `tests/fixtures/golden.json` and, for
# each of the eight tests, builds a `dataset()` from the case's columns,
# constructs the test with the case's parameters, runs it, and asserts that
# `statistic` / `p_value` / `dof` / `effect_size` match the recorded `expected`
# values within a tight tolerance. This is the binding's numeric parity gate
# against the SciPy/pgmpy reference.
```

```r
test_that("each golden case reproduces every defined result field", {
```

Inside the loop, replace the index-based case label with:

```r
    case_id <- case$id
```

After the existing dof assertion, add:

```r
    assert_close(result$effect_size, expected$effect_size, "effect_size", case_id)
```

- [ ] **Step 3: Rebuild and run the R package suite**

Run from the repository root:

```bash
Rscript -e 'rextendr::document("crates/ci-r")'
Rscript -e 'devtools::test("crates/ci-r", reporter = "summary")'
```

Expected: testthat passes the package suite, including the count assertion and all 80 four-field cases.

- [ ] **Step 4: Run R package metadata and style checks**

Run:

```bash
Rscript -e 'devtools::check("crates/ci-r", error_on = "warning")'
Rscript -e 'lintr::lint_package("crates/ci-r")'
```

Expected: `R CMD check` recognizes `jsonlite` as declared and lintr returns no lints.

- [ ] **Step 5: Commit the R contract check**

```bash
git add crates/ci-r/DESCRIPTION crates/ci-r/tests/testthat/test-golden.R
git commit -m "test(r): assert golden effect sizes"
```

---

### Task 7: Gate fixture drift in CI and correct the documentation

**Files:**

- Modify: `.github/workflows/python.yml:1-110`
- Modify: `.github/workflows/rust.yml:1-105`
- Modify: `.github/workflows/js.yml:1-112`
- Modify: `.github/workflows/r.yml:1-116`
- Modify: `README.md:198-218`
- Modify: `CONTRIBUTING.md:70-80,242-262`

**Interfaces:**

- Consumes: `tests/fixtures/requirements.txt`, `test_generate_golden.py`, and `generate_golden.py --check`.
- Produces: a Linux/Python 3.12 CI drift gate and accurate contributor instructions for the 80-case fixture.

- [ ] **Step 1: Add a dedicated fixture job to Python CI**

Insert this job before `python-lint` in `.github/workflows/python.yml`:

```yaml
  golden-fixture:
    name: Golden Fixture Drift
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install reference dependencies
        run: python -m pip install pytest -r tests/fixtures/requirements.txt

      - name: Test fixture generator
        run: pytest tests/fixtures/test_generate_golden.py

      - name: Check committed fixture
        run: python tests/fixtures/generate_golden.py --check
```

Change the Python binding job dependency to:

```yaml
    needs: [golden-fixture, python-lint]
```

Update both Python workflow comments from 73 to 80 cases:

```yaml
      # (crates/ci-python/test/test_golden.py) reproduces all 80 fixture cases.
```

```yaml
      # Verified locally: `pytest crates/ci-python/test/` -> golden = 80/80.
```

- [ ] **Step 2: Correct Rust and JavaScript workflow counts**

In `.github/workflows/rust.yml`, replace both 73-case comments with 80-case wording:

```yaml
# include the 80-case shared golden parity fixture.
```

```yaml
      # Verified locally: `cargo test -p ci_core` (includes the 80-case golden
```

In `.github/workflows/js.yml`, use:

```yaml
# golden parity suite (golden.test.js) covers all 80 fixture cases (81 vitest
# test cases: 80 parametrized + 1 coverage assertion).
```

and:

```yaml
      # the golden suite covers all 80 fixture cases.
```

- [ ] **Step 3: Correct R workflow counts and dependency wording**

Replace the opening R workflow description with:

```yaml
# Toolchain: R + rextendr (which compiles the embedded Rust crate), devtools,
# testthat, and jsonlite (the golden suite reads tests/fixtures/golden.json via
# jsonlite::fromJSON). rextendr::document() regenerates the extendr wrappers,
# then devtools::test() runs the package suite, including the 80-case golden
# parity gate.
```

Replace the dependency comment with:

```yaml
      # jsonlite is declared in Suggests and used by the golden suite; keep it
      # explicit here alongside the package development tools.
```

Replace the final 73-case comment with:

```yaml
      # 80-case golden parity gate (tests/fixtures/golden.json).
```

- [ ] **Step 4: Document regeneration and drift checking in README**

After the paragraph that introduces `tests/fixtures/golden.json` in `README.md`, insert:

````markdown
The fixture is generated independently with pinned NumPy/SciPy references and is committed so
normal binding tests do not need those Python dependencies. Regenerate or verify it from the
repository root:

```bash
python -m pip install -r tests/fixtures/requirements.txt
python tests/fixtures/generate_golden.py
python tests/fixtures/generate_golden.py --check
```
````

- [ ] **Step 5: Correct the contributor tree and fixture instructions**

Remove this nonexistent entry from the repository tree in `CONTRIBUTING.md`:

```text
├── benchmarks/            # Value-parity and runtime benchmarks
```

Replace the fixture regeneration paragraph and command with:

````markdown
If you change a statistic or add a test, install the pinned reference dependencies, regenerate
the fixture, and run its drift check before the binding suites:

```bash
python -m pip install -r tests/fixtures/requirements.txt
python tests/fixtures/generate_golden.py
python tests/fixtures/generate_golden.py --check
```

The generator uses deterministic inputs and independent NumPy/SciPy implementations of the
pgmpy formulas. It never imports the Rust core or a binding. The committed fixture contains 80
cases, and every consumer compares `statistic`, `p_value`, `dof`, and `effect_size` within an
absolute tolerance of `1e-7`.
````

- [ ] **Step 6: Verify workflow syntax and stale-text removal**

Run:

```bash
python -c "import pathlib, yaml; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ['.github/workflows/python.yml', '.github/workflows/rust.yml', '.github/workflows/js.yml', '.github/workflows/r.yml']]; print('yaml-ok')"
rg -n "73-case|73 fixture|all 73|golden = 73" .github README.md CONTRIBUTING.md
rg -n "^├── benchmarks/" CONTRIBUTING.md
git diff --check
```

Expected: `yaml-ok`; both `rg` commands exit with no matches; `git diff --check` has no output.

- [ ] **Step 7: Commit the CI and documentation contract**

```bash
git add .github/workflows/python.yml .github/workflows/rust.yml .github/workflows/js.yml .github/workflows/r.yml README.md CONTRIBUTING.md
git commit -m "ci: verify golden fixture drift"
```

---

### Task 8: Run final cross-language verification

**Files:**

- Verify only; do not change generated expectations to match an implementation failure.

**Interfaces:**

- Consumes: every deliverable from Tasks 1-7.
- Produces: evidence that the independent artifact is current and every available consumer passes.

- [ ] **Step 1: Verify the generator and strict JSON artifact from a clean calculation**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py -v
python tests/fixtures/generate_golden.py --check
python -c "import json, pathlib; cases=json.loads(pathlib.Path('tests/fixtures/golden.json').read_text()); assert len(cases)==80; assert len({c['id'] for c in cases})==80; print('golden-json-ok')"
```

Expected: eight pytest tests pass, check mode reports the fixture is current, and the final command prints `golden-json-ok`.

- [ ] **Step 2: Run Rust formatting, linting, and all core tests**

Run:

```bash
cargo fmt --all -- --check
cargo clippy -p ci_core --all-targets -- -D warnings
cargo test -p ci_core
```

Expected: all three commands exit zero; the golden integration target exercises 80 cases.

- [ ] **Step 3: Run Python formatting, linting, typing, and tests**

Run:

```bash
ruff format --check tests/fixtures crates/ci-python
ruff check tests/fixtures crates/ci-python
mypy crates/ci-python/test/
pytest crates/ci-python/test/
```

Expected: all commands exit zero and the Python golden module exercises 80 cases plus its count check.

- [ ] **Step 4: Run JavaScript and R verification when their toolchains are installed**

Run:

```bash
wasm-pack build crates/ci-js --target nodejs
npm --prefix crates/ci-js/tests ci
npm --prefix crates/ci-js/tests test
Rscript -e 'rextendr::document("crates/ci-r")'
Rscript -e 'devtools::test("crates/ci-r", reporter = "summary")'
```

Expected: both binding suites pass their API tests and all 80 fixture cases. If `wasm-pack` or R is unavailable, record the exact `command not found` evidence and rely on the corresponding CI workflow; do not claim that suite was run locally.

- [ ] **Step 5: Inspect the final repository state**

Run:

```bash
git status --short
git log --oneline -8
git diff HEAD~7..HEAD --check
```

Expected: no uncommitted implementation changes; the recent history contains the focused oracle, fixture, consumer, and CI/documentation commits; the aggregate diff has no whitespace errors.
