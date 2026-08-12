from __future__ import annotations

import importlib.util
import json
import math
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

    boundary = by_id(cases, "pearson-equivalence-outside-clip-boundary")
    assert boundary["expected"]["effect_size"] == pytest.approx(0.9999995, abs=1e-9)
    assert boundary["expected"]["effect_size"] > 0.999999 + 1e-7
    assert boundary["expected"]["statistic"] == pytest.approx(
        math.atanh(0.999999), abs=1e-9
    )


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


def test_validation_rejects_unknown_query_columns() -> None:
    generator = load_generator()
    cases = deepcopy(generator.build_cases())
    cases[0]["x"] = "MISSING"

    with pytest.raises(ValueError, match="invalid query columns"):
        generator.validate_cases(cases)


def test_validation_rejects_mismatched_column_lengths() -> None:
    generator = load_generator()
    cases = deepcopy(generator.build_cases())
    cases[0]["columns"]["X"]["values"].pop()

    with pytest.raises(ValueError, match="column lengths differ"):
        generator.validate_cases(cases)


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


def test_committed_fixture_matches_generator() -> None:
    generator = load_generator()
    rendered = generator.render_cases(generator.build_cases())
    assert GOLDEN_PATH.exists(), "golden.json must be committed"
    assert generator.check_fixture(GOLDEN_PATH, rendered), (
        "golden.json is stale; run python tests/fixtures/generate_golden.py"
    )
