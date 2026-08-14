# Tolerant Golden Fixture Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent harmless runner-dependent floating-point noise from making the committed golden fixture appear stale.

**Architecture:** Keep fixture generation and serialization unchanged. Parse the committed and generated JSON during freshness checks, recursively require exact types and values everywhere except the three floating expected-result fields, and compare only those fields with the existing absolute parity tolerance.

**Tech Stack:** Python 3.12+, standard-library `json` and `math`, pytest.

## Global Constraints

- Use an absolute tolerance of `1e-7` and a relative tolerance of `0.0`.
- Tolerate drift only in `expected.statistic`, `expected.p_value`, and `expected.effect_size`.
- Match case order, identifiers, keys, columns, parameters, queries, `dof`, and JSON value types exactly.
- Missing files, invalid JSON, non-finite tolerated values, and differences outside the tolerance remain stale.
- Do not change rendered fixture bytes or regenerate either committed fixture copy.

---

### Task 1: Semantic Fixture Freshness Comparison

**Files:**
- Modify: `tests/fixtures/generate_golden.py:602-638`
- Test: `tests/fixtures/test_generate_golden.py:145-177`

**Interfaces:**
- Consumes: `check_fixture(path: Path, rendered: str) -> bool` and JSON produced by `render_cases()`.
- Produces: `_fixture_values_match(committed: Any, generated: Any, path: tuple[str | int, ...] = ()) -> bool`; preserves the public `check_fixture(path: Path, rendered: str) -> bool` signature.

- [ ] **Step 1: Write the failing tolerance regression test**

Add a test that parses rendered cases, changes each tolerated expected-result field by `5e-8`, writes canonical JSON, and expects the fixture check to accept it:

```python
def test_check_fixture_tolerates_expected_numeric_roundoff(tmp_path: Path) -> None:
    generator = load_generator()
    rendered = generator.render_cases(generator.build_cases())
    committed = json.loads(rendered)
    expected = committed[0]["expected"]
    for field in ("statistic", "p_value", "effect_size"):
        expected[field] += 5e-8
    output = tmp_path / "golden.json"
    output.write_text(
        json.dumps(committed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    assert generator.check_fixture(output, rendered) is True
```

- [ ] **Step 2: Add preservation tests for significant and structural drift**

Add tests proving that `2e-7` expected-result drift, an input-column change, an expected-result type change, and malformed JSON each return `False`:

```python
def test_check_fixture_rejects_meaningful_or_structural_drift(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    rendered = generator.render_cases(generator.build_cases())
    baseline = json.loads(rendered)
    output = tmp_path / "golden.json"

    changed = deepcopy(baseline)
    changed[0]["expected"]["statistic"] += 2e-7
    output.write_text(json.dumps(changed), encoding="utf-8")
    assert generator.check_fixture(output, rendered) is False

    changed = deepcopy(baseline)
    changed[0]["columns"]["X"]["values"][0] += 1
    output.write_text(json.dumps(changed), encoding="utf-8")
    assert generator.check_fixture(output, rendered) is False

    changed = deepcopy(baseline)
    changed[0]["expected"]["statistic"] = 0
    output.write_text(json.dumps(changed), encoding="utf-8")
    assert generator.check_fixture(output, rendered) is False

    output.write_text("{", encoding="utf-8")
    assert generator.check_fixture(output, rendered) is False
```

- [ ] **Step 3: Run the regression test and verify RED**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py::test_check_fixture_tolerates_expected_numeric_roundoff -q
```

Expected: FAIL because the current `check_fixture()` requires byte-identical text.

- [ ] **Step 4: Implement the minimal semantic comparator**

Add constants and a recursive comparator to `generate_golden.py`:

```python
FIXTURE_NUMERIC_TOLERANCE = 1e-7
TOLERATED_EXPECTED_FIELDS = {"statistic", "p_value", "effect_size"}


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
```

Update `check_fixture()` to parse both documents and return `False` on missing files, decode failures, or semantic mismatches:

```python
def check_fixture(path: Path, rendered: str) -> bool:
    if not path.exists():
        return False
    try:
        committed = json.loads(path.read_text(encoding="utf-8"))
        generated = json.loads(rendered)
    except (OSError, json.JSONDecodeError):
        return False
    return _fixture_values_match(committed, generated)
```

- [ ] **Step 5: Run targeted and complete fixture tests**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py -q
pytest tests -q
python tests/fixtures/generate_golden.py --check
```

Expected: all tests pass and the generator reports the committed fixture is current.

- [ ] **Step 6: Run formatting and asset verification**

Run:

```bash
ruff format --check .
ruff check .
python crates/ci-r/tools/sync_package_assets.py --check
git diff --check
```

Expected: all commands exit successfully; the asset checker reports that R package assets match canonical sources.

- [ ] **Step 7: Commit the implementation**

```bash
git add tests/fixtures/generate_golden.py tests/fixtures/test_generate_golden.py docs/superpowers/plans/2026-08-14-tolerant-golden-fixture-check.md
git commit -m "fix: tolerate golden fixture numeric noise"
```
