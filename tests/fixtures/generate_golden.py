#!/usr/bin/env python3
"""Generate the language-neutral CI-test parity fixture from independent references."""

from __future__ import annotations

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

import numpy as np
from scipy import stats


LAMBDA_BY_TEST = {
    "chi_squared": 1.0,
    "log_likelihood": 0.0,
    "cressie_read": 2.0 / 3.0,
    "freeman_tukey": -0.5,
    "modified_likelihood": -1.0,
}
DISCRETE_LAMBDAS = LAMBDA_BY_TEST
RHO_CLIP = 0.999999
CORRELATION_ZERO_ATOL = 1e-15
EXPECTED_CASE_COUNT = 80
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
DEFAULT_OUTPUT = Path(__file__).with_name("golden.json")

# The R source package carries its own copy so the built archive can run the
# parity suite outside this repository -- CRAN runs a package's tests on its
# check farm, so the copy is load-bearing rather than redundant. Writing both
# destinations in one pass removes the ritual where regenerating the canonical
# fixture and forgetting the sync step left Python CI green and R CI red, in a
# different workflow.
PACKAGED_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "citest-r"
    / "tests"
    / "testthat"
    / "fixtures"
    / "golden.json"
)


def fixture_destinations(primary: Path) -> list[Path]:
    """Every path the fixture must be written to, `primary` first.

    The packaged copy is only included when the caller is writing the canonical
    location; an explicit `--output` elsewhere (used by the tests) writes just
    that one file.
    """
    if primary.resolve() != DEFAULT_OUTPUT.resolve():
        return [primary]
    return [primary, PACKAGED_OUTPUT]
FIXTURE_NUMERIC_TOLERANCE = 1e-7
TOLERATED_EXPECTED_FIELDS = {"statistic", "p_value", "effect_size"}


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
    columns = {"X": _column("discrete", x_values), "Y": _column("discrete", y_values)}
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
            "discrete-2x3", True, [((), [0, 1], [0, 1, 2], [[9, 4, 7], [3, 8, 5]])]
        ),
        _discrete_scenario(
            "discrete-3x2", True, [((), [0, 1, 2], [0, 1], [[8, 3], [4, 9], [6, 5]])]
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
            observed, correction=case["yates"], lambda_=lambda_
        )
        statistic += float(result.statistic)
        dof += int(result.dof)
    p_value = float(stats.chi2.sf(statistic, dof))
    cardinality = min(len(np.unique(x)), len(np.unique(y)))
    if cardinality < 2:
        raise ValueError("degenerate discrete table: global min cardinality < 2")
    effect_size = math.sqrt(statistic / (len(x) * (cardinality - 1)))
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
    scenario_id: str, x: np.ndarray, y: np.ndarray, z: tuple[np.ndarray, ...] = ()
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
                "outside-clip-boundary",
                U,
                _linear((0.9999995, U), (math.sqrt(1.0 - 0.9999995**2), V)),
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
    scenarios.append(
        _continuous_scenario("perfect-correlation", U, _linear((3.0, U), offset=2.0))
    )
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
        correlation = float(np.corrcoef(x - design @ x_beta, y - design @ y_beta)[0, 1])
    if math.isclose(correlation, 0.0, abs_tol=CORRELATION_ZERO_ATOL):
        correlation = 0.0
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
    rho = float(np.clip(correlation, -RHO_CLIP, RHO_CLIP))
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
    rho = float(np.clip(correlation, -RHO_CLIP, RHO_CLIP))
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
        for test, lambda_ in LAMBDA_BY_TEST.items():
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
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def validate_cases(cases: list[dict[str, Any]]) -> None:
    counts = Counter(case.get("test") for case in cases)
    if len(cases) != EXPECTED_CASE_COUNT or counts != Counter(EXPECTED_COUNTS):
        raise ValueError(
            f"expected the approved 80-case distribution, got {dict(counts)}"
        )
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
        if test in LAMBDA_BY_TEST:
            if set(params) != {"yates"} or not isinstance(params["yates"], bool):
                raise ValueError(f"{case_id}: invalid discrete params")
        elif test == "pearson_equivalence":
            if set(params) != {"delta_threshold"} or not _finite_number(
                params["delta_threshold"]
            ):
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
            if not isinstance(name, str) or column.get("kind") not in {
                "discrete",
                "continuous",
            }:
                raise ValueError(f"{case_id}: invalid column {name!r}")
            values = column.get("values")
            if (
                not isinstance(values, list)
                or not values
                or not all(_finite_number(v) for v in values)
            ):
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
        if dof is not None and (
            isinstance(dof, bool) or not isinstance(dof, int) or dof < 0
        ):
            raise ValueError(f"{case_id}: invalid dof {dof!r}")
        if not 0.0 <= expected["p_value"] <= 1.0 or expected["effect_size"] < 0.0:
            raise ValueError(f"{case_id}: invalid probability or effect size")


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
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
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


def _fixture_values_match(
    committed: Any,
    generated: Any,
    path: tuple[str | int, ...] = (),
) -> bool:
    if type(committed) is not type(generated):
        return False
    if isinstance(generated, dict):
        return committed.keys() == generated.keys() and all(
            _fixture_values_match(committed[key], value, (*path, key))
            for key, value in generated.items()
        )
    if isinstance(generated, list):
        return len(committed) == len(generated) and all(
            _fixture_values_match(left, right, (*path, index))
            for index, (left, right) in enumerate(zip(committed, generated))
        )
    if (
        len(path) >= 2
        and path[-2] == "expected"
        and path[-1] in TOLERATED_EXPECTED_FIELDS
    ):
        return (
            _finite_number(committed)
            and _finite_number(generated)
            and math.isclose(
                committed,
                generated,
                rel_tol=0.0,
                abs_tol=FIXTURE_NUMERIC_TOLERANCE,
            )
        )
    return committed == generated


def check_fixture(path: Path, rendered: str) -> bool:
    if not path.exists():
        return False
    try:
        committed = json.loads(path.read_text(encoding="utf-8"))
        generated = json.loads(rendered)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return _fixture_values_match(committed, generated)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixture without writing it",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered = render_cases(build_cases())
    destinations = fixture_destinations(args.output)
    if args.check:
        stale = [path for path in destinations if not check_fixture(path, rendered)]
        if not stale:
            for path in destinations:
                print(f"golden fixture is current: {path}")
            return 0
        for path in stale:
            print(f"golden fixture is missing or stale: {path}", file=sys.stderr)
        print(
            "regenerate it with: python tests/fixtures/generate_golden.py",
            file=sys.stderr,
        )
        return 1
    for path in destinations:
        write_fixture(path, rendered)
        print(f"wrote {EXPECTED_CASE_COUNT} golden cases: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
