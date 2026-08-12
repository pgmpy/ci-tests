# Complete Parity and Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the independent 80-case cross-language parity gate, verify every overlapping case against current pgmpy, repair all CI blockers, and apply behavior-preserving simplifications.

**Architecture:** The existing NumPy/SciPy generator remains the independent oracle and produces a committed strict-JSON contract consumed by Rust, Python, R, and JavaScript. A separate optional pgmpy checker consumes that artifact without participating in generation, while CI validates deterministic fixture drift and each binding. Automatic simplification happens only after behavior and parity are green.

**Tech Stack:** Python 3.12, NumPy 2.4.2, SciPy 1.17.0, pytest, pgmpy `dev` at `8a221e915889f77bc3e503528e63df5803cfd43d`, Rust/Clippy/rustfmt, PyO3/maturin/Ruff/MyPy, wasm-bindgen/Vitest/Prettier/ESLint, extendr/testthat/styler/lintr, GitHub Actions.

## Global Constraints

- Preserve the existing public Rust, Python, R, and JavaScript APIs.
- Preserve the two pre-existing uncommitted edits in `docs/superpowers/plans/2026-07-18-golden-fixture-parity.md` and `docs/superpowers/specs/2026-07-18-golden-fixture-parity-design.md`; do not stage or rewrite them.
- Keep the independent fixture generator free of imports from pgmpy, `ci_core`, and every binding.
- Emit exactly 80 finite successful cases with the distribution in the approved golden-fixture design.
- Compare every result field, including asserting that JSON null maps to an absent/`None`/`NULL`/`null` binding field.
- Keep pgmpy out of build, runtime, normal test, and fixture-generation dependencies.
- Treat Yates-disabled cases as an intentional extension and report, rather than fail, them in pgmpy parity.
- Do not restore benchmarks or add CI-test algorithms.
- Follow test-driven development for new behavior: add or tighten the test, observe the intended failure, implement, then rerun.
- Do not update expected fixture values to hide an implementation mismatch.

---

### Task 1: Materialize the independent oracle and canonical fixture

**Files:**

- Create: `tests/fixtures/requirements.txt`
- Create: `tests/fixtures/test_generate_golden.py`
- Create: `tests/fixtures/generate_golden.py`
- Create: `tests/fixtures/golden.json`
- Reference only: `docs/superpowers/plans/2026-07-18-golden-fixture-parity.md:43-1077`

**Interfaces:**

- Consumes: pinned NumPy/SciPy and the approved catalog/formulas.
- Produces: `build_cases() -> list[dict[str, Any]]`, `validate_cases(cases) -> None`, `render_cases(cases) -> str`, `write_fixture(path) -> None`, `check_fixture(path) -> bool`, and the canonical 80-case JSON artifact.

- [ ] **Step 1: Add the failing generator contract tests**

Follow Tasks 1 and 2 of `docs/superpowers/plans/2026-07-18-golden-fixture-parity.md` exactly, including the current uncommitted refinements for the Pearson-equivalence clipping boundary and non-degenerate Cramér's V. The test file must cover:

```python
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
```

It must also cover duplicate IDs, non-finite expected values, unknown query columns, canonical bytes, deterministic output, and non-mutating `--check` behavior.

- [ ] **Step 2: Verify the red state**

Run:

```bash
pytest tests/fixtures/test_generate_golden.py -v
```

Expected: collection or tests fail specifically because `generate_golden.py` and the required generator interfaces do not exist.

- [ ] **Step 3: Implement the exact approved catalog and formulas**

Copy the code-complete generator implementation from Tasks 1 and 2 of the existing plan into `tests/fixtures/generate_golden.py`. Preserve these numerical contracts:

```python
LAMBDA_BY_TEST = {
    "chi_squared": 1.0,
    "log_likelihood": 0.0,
    "cressie_read": 2.0 / 3.0,
    "freeman_tukey": -0.5,
    "modified_likelihood": -1.0,
}

RHO_CLIP = 0.999999
EXPECTED_CASE_COUNT = 80
```

Use `scipy.stats.chi2_contingency`, `scipy.stats.pearsonr`, `numpy.linalg.lstsq`, Student-t tails, normal tails, 12-significant-digit canonicalization, `allow_nan=False`, sorted keys, two-space indentation, and a trailing newline.

- [ ] **Step 4: Generate and verify the artifact**

Run:

```bash
python tests/fixtures/generate_golden.py
pytest tests/fixtures/test_generate_golden.py -v
python tests/fixtures/generate_golden.py --check
python -c "import json, pathlib; p=pathlib.Path('tests/fixtures/golden.json'); cases=json.loads(p.read_text()); assert len(cases)==80; assert len({c['id'] for c in cases})==80; print('golden-json-ok')"
```

Expected: eight focused tests pass, check mode reports current bytes, and the final command prints `golden-json-ok`.

- [ ] **Step 5: Prove byte stability**

Run:

```bash
sha256sum tests/fixtures/golden.json
python tests/fixtures/generate_golden.py
sha256sum tests/fixtures/golden.json
```

Expected: both hashes are identical.

- [ ] **Step 6: Commit the oracle**

Stage only the four `tests/fixtures` files and commit:

```bash
git add tests/fixtures/requirements.txt tests/fixtures/test_generate_golden.py tests/fixtures/generate_golden.py tests/fixtures/golden.json
git commit -m "test: add reproducible golden parity fixture"
```

---

### Task 2: Enforce the complete fixture contract in every consumer

**Files:**

- Modify: `crates/ci-core/tests/golden.rs`
- Modify: `crates/ci-python/test/test_golden.py`
- Modify: `crates/ci-js/tests/golden.test.js`
- Modify: `crates/ci-r/tests/testthat/test-golden.R`
- Modify: `crates/ci-r/DESCRIPTION`

**Interfaces:**

- Consumes: fixture fields `id`, `test`, `params`, `columns`, `x`, `y`, `z`, and all four `expected` members.
- Produces: stable, case-named four-field assertions in Rust, Python, JavaScript, and R.

- [ ] **Step 1: Tighten Rust null and ID assertions first**

Add `id: String` to `Case`. Change `assert_field` so absence is asserted rather than skipped:

```rust
fn assert_field(name: &str, case: &Case, expected: Option<f64>, actual: Option<f64>) {
    let Some(exp) = expected else {
        assert!(
            actual.is_none(),
            "[{}:{}] expected {name}=None, got {actual:?}",
            case.id,
            case.test,
        );
        return;
    };
    let act = actual.unwrap_or_else(|| {
        panic!(
            "[{}:{}] expected {name}={exp} but result had None",
            case.id, case.test
        )
    });
    // Retain the existing exact-infinity and absolute-tolerance branches.
}
```

Add `assert!(!case.id.is_empty())` before executing each case and use `case.id` in every failure.

- [ ] **Step 2: Verify Rust fails against the newly tightened code before completing it**

Run after adding the ID assertion but before adding `Case.id`:

```bash
cargo test -p ci_core --test golden
```

Expected: compilation fails because `Case` has no `id` field. Complete the deserialization and assertion changes, then rerun and expect all 80 cases to pass.

- [ ] **Step 3: Tighten Python, JavaScript, and R tests**

For Python, parameterize by `case["id"]`, set `case_id = case["id"]`, and add:

```python
_assert_close(result.effect_size, expected.get("effect_size"), "effect_size", case_id)
```

Change `_assert_close` so `expected is None` asserts `actual is None`.

For JavaScript, use `testCase.id`, add:

```javascript
assertClose(result.effectSize, expected.effect_size, "effectSize", caseId);
```

and make a null expectation assert `actual === null || actual === undefined`.

For R, use `case$id`, add:

```r
assert_close(result$effect_size, expected$effect_size, "effect_size", case_id)
```

and make `is.null(expected)` call `expect_null(actual, label = ...)`.

- [ ] **Step 4: Declare R's fixture reader**

Add `jsonlite` to `Suggests`:

```text
Suggests:
    devtools,
    jsonlite,
    rextendr,
    testthat (>= 3.0.0)
```

- [ ] **Step 5: Run consumer gates**

Run:

```bash
cargo test -p ci_core --test golden -- --nocapture
python -m pytest tests/fixtures/test_generate_golden.py crates/ci-python/test/test_golden.py -v
```

The Python binding must be installed first with `python -m pip install -e crates/ci-python --no-build-isolation`. Run JavaScript and R consumers in Task 6 after their toolchains/configuration are repaired.

- [ ] **Step 6: Commit consumer completion**

```bash
git add crates/ci-core/tests/golden.rs crates/ci-python/test/test_golden.py crates/ci-js/tests/golden.test.js crates/ci-r/tests/testthat/test-golden.R crates/ci-r/DESCRIPTION
git commit -m "test: enforce complete golden result contract"
```

---

### Task 3: Add the optional current-pgmpy parity checker

**Files:**

- Create: `tests/fixtures/check_pgmpy_parity.py`
- Create: `tests/fixtures/test_check_pgmpy_parity.py`
- Create: `tests/fixtures/requirements-pgmpy.txt`

**Interfaces:**

- Consumes: `golden.json`, current pgmpy CI-test classes, optional `--pgmpy-source PATH`, and `--tolerance`.
- Produces: `should_skip(case) -> str | None`, `compare_case(case, pgmpy_module, tolerance) -> list[str]`, and a CLI summary with compared/skipped/failed counts.

- [ ] **Step 1: Write failing pure-contract tests**

Create tests with literal expectations:

```python
def test_yates_disabled_discrete_case_is_an_intentional_skip() -> None:
    case = {"test": "chi_squared", "params": {"yates": False}}
    assert checker.should_skip(case) == "pgmpy has no Yates-disable option"


def test_continuous_case_is_not_skipped() -> None:
    case = {"test": "fisher_z", "params": {}}
    assert checker.should_skip(case) is None


def test_compare_fields_reports_a_named_mismatch() -> None:
    errors = checker.compare_fields(
        case_id="pearson-positive",
        expected={"statistic": 0.8, "p_value": 0.01, "dof": 14, "effect_size": 0.8},
        actual={"statistic": 0.7, "p_value": 0.01, "dof": 14, "effect_size": 0.8},
        tolerance=1e-7,
    )
    assert errors == ["pearson-positive: statistic expected 0.8, got 0.7"]


def test_missing_pgmpy_dof_is_not_compared() -> None:
    assert checker.compare_fields(
        case_id="pearson-unconditional",
        expected={"statistic": 0.0, "p_value": 1.0, "dof": 14, "effect_size": 0.0},
        actual={"statistic": 0.0, "p_value": 1.0, "dof": None, "effect_size": 0.0},
        tolerance=1e-7,
    ) == []
```

- [ ] **Step 2: Verify red**

Run:

```bash
pytest tests/fixtures/test_check_pgmpy_parity.py -v
```

Expected: import fails because `check_pgmpy_parity.py` does not exist.

- [ ] **Step 3: Implement pure comparison and input conversion**

Implement:

```python
DISCRETE_TESTS = {
    "chi_squared",
    "log_likelihood",
    "cressie_read",
    "freeman_tukey",
    "modified_likelihood",
}


def should_skip(case: dict[str, Any]) -> str | None:
    if case["test"] in DISCRETE_TESTS and case["params"].get("yates") is False:
        return "pgmpy has no Yates-disable option"
    return None


def build_dataframe(case: dict[str, Any], pandas: ModuleType) -> Any:
    return pandas.DataFrame({name: spec["values"] for name, spec in case["columns"].items()})
```

`compare_fields` compares finite numerics at zero relative tolerance and absolute `tolerance`; it ignores expected `dof` only when pgmpy has no `dof_` attribute, but compares every other field including null.

Implement `compare_case` as the integration boundary:

```python
def compare_case(case: dict[str, Any], pgmpy_module: ModuleType, tolerance: float) -> list[str]:
    data = build_dataframe(case, importlib.import_module("pandas"))
    test = CONSTRUCTORS[case["test"]](pgmpy_module, data, case["params"])
    test.run_test(X=case["x"], Y=case["y"], Z=case["z"])
    actual = {
        "statistic": test.statistic_,
        "p_value": test.p_value_,
        "dof": getattr(test, "dof_", None),
        "effect_size": test.effect_size_,
    }
    return compare_fields(case["id"], case["expected"], actual, tolerance)
```

The CLI loops over all cases, counts `should_skip` reasons, accumulates errors from `compare_case`, prints `compared=<n> skipped=<n> failed=<n>`, and returns exit status one exactly when `failed > 0`.

- [ ] **Step 4: Implement pgmpy dispatch**

Import after processing `--pgmpy-source`, then dispatch:

```python
constructors = {
    "chi_squared": lambda m, df, p: m.ChiSquare(data=df),
    "log_likelihood": lambda m, df, p: m.LogLikelihood(data=df),
    "cressie_read": lambda m, df, p: m.PowerDivergence(data=df, lambda_="cressie-read"),
    "freeman_tukey": lambda m, df, p: m.PowerDivergence(data=df, lambda_="freeman-tuckey"),
    "modified_likelihood": lambda m, df, p: m.ModifiedLogLikelihood(data=df),
    "pearson_correlation": lambda m, df, p: m.Pearsonr(data=df),
    "pearson_equivalence": lambda m, df, p: m.PearsonrEquivalence(
        data=df, delta_threshold=p["delta_threshold"]
    ),
    "fisher_z": lambda m, df, p: m.FisherZ(data=df),
}
```

Call `run_test(X=case["x"], Y=case["y"], Z=case["z"])` and extract `statistic_`, `p_value_`, optional `dof_`, and `effect_size_`.

- [ ] **Step 5: Add optional dependency instructions**

Create `requirements-pgmpy.txt`:

```text
-r requirements.txt
pgmpy @ git+https://github.com/pgmpy/pgmpy.git@8a221e915889f77bc3e503528e63df5803cfd43d
```

The normal generator requirements do not reference this file.

- [ ] **Step 6: Run unit and real integration parity**

Run:

```bash
pytest tests/fixtures/test_check_pgmpy_parity.py -v
python tests/fixtures/check_pgmpy_parity.py --pgmpy-source /home/ankur/work/pgmpy/pgmpy
```

Expected: pure tests pass; real parity exits zero, compares all shared cases, reports Yates-disabled cases as intentional skips, and reports zero mismatches. If current pgmpy reveals a true semantic difference, stop and add a focused regression test before changing the implementation or the declared exception set.

- [ ] **Step 7: Commit pgmpy parity support**

```bash
git add tests/fixtures/check_pgmpy_parity.py tests/fixtures/test_check_pgmpy_parity.py tests/fixtures/requirements-pgmpy.txt
git commit -m "test: compare golden cases with current pgmpy"
```

---

### Task 4: Repair Python packaging, linting, typing, and examples

**Files:**

- Modify: `crates/ci-python/pyproject.toml`
- Modify: `crates/ci-python/ci_python/__init__.py`
- Modify: `crates/ci-python/test/test_api.py`
- Modify: `crates/ci-python/test/test_golden.py`
- Modify: `.github/workflows/python.yml`
- Modify: `crates/ci-python/README.md`

**Interfaces:**

- Consumes: the Python package and fixture tests.
- Produces: a declared NumPy runtime dependency and CI commands that pass Ruff, MyPy, build, and tests.

- [ ] **Step 1: Capture current failures**

Run:

```bash
ruff format --check crates/ci-python
ruff check crates/ci-python
mypy crates/ci-python/test
```

Expected: formatting/lint failures and missing pandas typing evidence as recorded in the branch audit.

- [ ] **Step 2: Align package metadata and lint scope**

Set:

```toml
dependencies = [
  "numpy>=1.26",
]

[project.optional-dependencies]
test = [
  "mypy",
  "pandas",
  "pandas-stubs",
  "pytest",
  "ruff",
]
```

Add targeted Ruff ignores for test-only docstrings and dynamic JSON values instead of weakening all package linting:

```toml
[tool.ruff.lint.per-file-ignores]
"test/**" = ["ANN401", "D103"]
```

Move `numpy` to module scope in `ci_python/__init__.py`; keep pandas lazy, with `# noqa: PLC0415` on the intentional optional import. Correct `_safe_issubdtype` to a valid multi-line docstring.

- [ ] **Step 3: Apply and verify Ruff fixes**

Run:

```bash
ruff check --fix crates/ci-python
ruff format crates/ci-python
ruff format --check crates/ci-python
ruff check crates/ci-python
```

Review the diff and retain only behavior-preserving edits.

- [ ] **Step 4: Fix CI installation and add fixture drift**

Add a `golden-fixture` job that installs `pytest -r tests/fixtures/requirements.txt`, runs both fixture test modules, and runs `generate_golden.py --check`. Make `python-test` depend on it and `python-lint`.

Replace the binding dependency install with:

```yaml
run: pip install maturin numpy pandas pandas-stubs pytest mypy
```

Change all 73-case comments to 80.

- [ ] **Step 5: Correct Python examples and degrees of freedom**

In `crates/ci-python/README.md`, include `FisherZ` in the test table, make every discrete query condition only on discrete columns, make every continuous query use continuous `X`, `Y`, and `Z`, and state that `PearsonCorrelation` reports `n - |Z| - 2` while Fisher-Z/equivalence report `None`.

- [ ] **Step 6: Verify Python completely**

Run:

```bash
python -m pip install -e "crates/ci-python[test]" --no-build-isolation
mypy crates/ci-python/test
pytest crates/ci-python/test -v
```

Expected: MyPy and all API plus 80 golden cases pass.

- [ ] **Step 7: Commit Python repairs**

```bash
git add crates/ci-python/pyproject.toml crates/ci-python/ci_python/__init__.py crates/ci-python/test/test_api.py crates/ci-python/test/test_golden.py crates/ci-python/README.md .github/workflows/python.yml
git commit -m "ci(python): restore lint type and parity gates"
```

---

### Task 5: Repair JavaScript and R quality gates

**Files:**

- Modify: `crates/ci-js/package.json`
- Modify: `crates/ci-js/package-lock.json`
- Modify: `crates/ci-js/eslint.config.mjs`
- Modify: `crates/ci-js/tests/api.test.js`
- Modify: `crates/ci-js/tests/golden.test.js`
- Modify: `.github/workflows/js.yml`
- Modify: `.github/workflows/r.yml`
- Modify: `crates/ci-r/R/cir.R`
- Modify: `crates/ci-r/tests/testthat/test-golden.R`
- Create: `crates/ci-r/.lintr`

**Interfaces:**

- Consumes: the JS/R binding sources and generated R wrapper boundary.
- Produces: declared JS formatting/lint tools and R checks that exclude only generated wrapper code.

- [ ] **Step 1: Add and pin JS quality tooling**

Run from `crates/ci-js`:

```bash
npm install --save-dev --save-exact eslint prettier
```

Replace the empty ESLint config with:

```javascript
import globals from "globals";

export default [
  {
    files: ["**/*.js", "**/*.mjs"],
    ignores: ["pkg/**", "node_modules/**"],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
];
```

If `globals` is not already supplied transitively, install it with `npm install --save-dev --save-exact globals` and commit the lockfile.

- [ ] **Step 2: Format and lint JavaScript**

Run:

```bash
npx prettier --write .
npx eslint .
npx prettier --check .
```

Fix real unused/undefined findings without disabling the rules globally. Update all workflow comments to 80 cases.

- [ ] **Step 3: Make R checks generated-code-aware**

Create `.lintr`:

```text
exclusions: list("R/extendr-wrappers.R")
```

Change the formatting command to exclude the generated wrapper and roxygen examples:

```yaml
run: |
  files <- setdiff(list.files("R", pattern = "[.]R$", full.names = TRUE), "R/extendr-wrappers.R")
  styler::style_file(files, dry = "fail", include_roxygen_examples = FALSE)
```

Keep `lintr::lint_package()` so `.lintr` controls the same exclusion locally and in CI. Change comments to 80 cases and say `jsonlite` is declared in `Suggests`.

- [ ] **Step 4: Apply R formatting to handwritten files**

When R is available, run from `crates/ci-r`:

```bash
Rscript -e 'files <- setdiff(list.files("R", pattern="[.]R$", full.names=TRUE), "R/extendr-wrappers.R"); styler::style_file(files, include_roxygen_examples=FALSE)'
Rscript -e 'lints <- lintr::lint_package(); print(lints); quit(status=as.integer(length(lints) > 0))'
```

Do not manually simplify `R/extendr-wrappers.R`.

- [ ] **Step 5: Commit quality-gate repairs**

```bash
git add crates/ci-js/package.json crates/ci-js/package-lock.json crates/ci-js/eslint.config.mjs crates/ci-js/tests/api.test.js crates/ci-js/tests/golden.test.js .github/workflows/js.yml .github/workflows/r.yml crates/ci-r/R/cir.R crates/ci-r/tests/testthat/test-golden.R crates/ci-r/.lintr
git commit -m "ci: restore JavaScript and R quality gates"
```

---

### Task 6: Correct shared documentation and remove dead dependencies

**Files:**

- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `Cargo.toml`
- Modify: `crates/ci-core/Cargo.toml`
- Modify: `Cargo.lock`

**Interfaces:**

- Consumes: completed fixture/parity workflow and current public behavior.
- Produces: accurate contributor instructions and a smaller dependency graph.

- [ ] **Step 1: Correct documentation**

Document:

```bash
python -m pip install -r tests/fixtures/requirements.txt
python tests/fixtures/generate_golden.py
python tests/fixtures/generate_golden.py --check
python tests/fixtures/check_pgmpy_parity.py --pgmpy-source /path/to/pgmpy
```

State that Pearson correlation reports `dof = n - |Z| - 2`, while Fisher-Z and Pearson equivalence report no degrees of freedom. Remove the nonexistent `benchmarks/` tree entry and all stale 73-case claims.

- [ ] **Step 2: Prove development dependencies are unused**

Run:

```bash
rg -n 'criterion|proptest|anyhow|ndarray' --glob '!Cargo.lock' .
```

Expected: only workspace/core manifest declarations remain for `criterion`, `proptest`, `anyhow`, and `ndarray`.

- [ ] **Step 3: Remove unused dependencies**

Remove those unused workspace dependencies and the corresponding `ci-core` dev-dependencies. Run:

```bash
cargo check -p ci_core -p ci_python -p ci_js
cargo test -p ci_core --lib
```

Expected: Cargo refreshes `Cargo.lock`; builds and unit tests pass.

- [ ] **Step 4: Verify stale-text and metadata cleanup**

Run:

```bash
rg -n '73-case|all 73|golden = 73|^├── benchmarks/' .github README.md CONTRIBUTING.md
git diff --check
```

Expected: no matches and no whitespace errors.

- [ ] **Step 5: Commit documentation and dependency cleanup**

```bash
git add README.md CONTRIBUTING.md Cargo.toml crates/ci-core/Cargo.toml Cargo.lock
git commit -m "docs: document parity workflow and trim dependencies"
```

---

### Task 7: Run all bindings and resolve parity failures by root cause

**Files:**

- Modify only files implicated by a focused failing test.

**Interfaces:**

- Consumes: all completed implementation tasks.
- Produces: full cross-language evidence and focused regression tests for any discovered defect.

- [ ] **Step 1: Run Rust and Python**

```bash
cargo test -p ci_core
pytest tests/fixtures crates/ci-python/test -v
python tests/fixtures/check_pgmpy_parity.py --pgmpy-source /home/ankur/work/pgmpy/pgmpy
```

- [ ] **Step 2: Run JavaScript**

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build crates/ci-js --target nodejs
npm --prefix crates/ci-js/tests ci
npm --prefix crates/ci-js/tests test
npm --prefix crates/ci-js ci
npm --prefix crates/ci-js exec prettier -- --check .
npm --prefix crates/ci-js exec eslint -- .
```

- [ ] **Step 3: Run R**

```bash
Rscript -e 'rextendr::document("crates/ci-r")'
Rscript -e 'devtools::test("crates/ci-r", reporter="summary")'
Rscript -e 'devtools::check("crates/ci-r", args="--no-manual", error_on="warning")'
```

- [ ] **Step 4: Handle any real failure with TDD**

For each failure, add the smallest focused regression test that fails for the observed reason, rerun to confirm red, fix only the root cause, then rerun the focused and complete suite. Never modify `golden.json` directly.

- [ ] **Step 5: Commit only if verification exposed implementation defects**

Use one focused commit per root cause, for example:

```bash
git commit -m "fix(core): align sparse-stratum parity"
```

Skip this step if no implementation defect is found.

---

### Task 8: Apply behavior-preserving simplification tools

**Files:**

- Modify: files selected by the automatic tools and confirmed by diff review.

**Interfaces:**

- Consumes: a green implementation and parity baseline.
- Produces: simpler formatted code with identical public and numeric behavior.

- [ ] **Step 1: Record the pre-simplification baseline**

```bash
git status --short
sha256sum tests/fixtures/golden.json
cargo test -p ci_core --quiet
python tests/fixtures/check_pgmpy_parity.py --pgmpy-source /home/ankur/work/pgmpy/pgmpy
```

- [ ] **Step 2: Run Rust simplification**

```bash
cargo clippy --fix -p ci_core --all-targets --allow-dirty --allow-staged -- -D warnings
cargo fmt --all
```

- [ ] **Step 3: Run Python and JavaScript simplification**

```bash
ruff check --fix crates/ci-python tests/fixtures
ruff format crates/ci-python tests/fixtures
npm --prefix crates/ci-js exec prettier -- --write .
npm --prefix crates/ci-js exec eslint -- . --fix
```

- [ ] **Step 4: Run R simplification on handwritten code**

```bash
Rscript -e 'files <- setdiff(list.files("crates/ci-r/R", pattern="[.]R$", full.names=TRUE), "crates/ci-r/R/extendr-wrappers.R"); styler::style_file(files, include_roxygen_examples=FALSE)'
```

- [ ] **Step 5: Review every automatic edit**

```bash
git diff --stat
git diff -- crates tests .github README.md CONTRIBUTING.md Cargo.toml Cargo.lock
```

Discard no user-owned change. Revert only individual automatic hunks that change behavior, weaken validation, or edit generated code.

- [ ] **Step 6: Re-run complete verification**

Run the exact Task 7 commands again, plus:

```bash
cargo fmt --all -- --check
cargo clippy -p ci_core --all-targets -- -D warnings
ruff format --check crates/ci-python tests/fixtures
ruff check crates/ci-python tests/fixtures
python tests/fixtures/generate_golden.py --check
sha256sum tests/fixtures/golden.json
git diff --check
```

Expected: every command exits zero and the fixture hash matches the pre-simplification hash.

- [ ] **Step 7: Commit simplifications if any remain**

```bash
git add crates tests .github README.md CONTRIBUTING.md Cargo.toml Cargo.lock
git commit -m "refactor: simplify parity implementation"
```

Do not stage the two pre-existing modified July documents.

---

### Task 9: Final branch verification and handoff

**Files:**

- Verify only.

**Interfaces:**

- Consumes: the completed branch.
- Produces: exact final evidence and a concise remaining-risk report.

- [ ] **Step 1: Verify tracked artifact and branch state**

```bash
git status --short --branch
git ls-files tests/fixtures
git diff --check upstream/main...HEAD
```

Expected: only the two known user-owned July document modifications remain unstaged; all fixture/parity files are tracked; aggregate diff has no whitespace errors.

- [ ] **Step 2: Verify workflows parse**

```bash
python -c "import pathlib, yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path('.github/workflows').glob('*.yml')]; print('workflow-yaml-ok')"
```

- [ ] **Step 3: Inspect commits and summarize evidence**

```bash
git log --oneline --decorate upstream/main..HEAD
git diff --stat upstream/main...HEAD
```

Report exact test counts, pgmpy compared/skipped/mismatch counts, unavailable local toolchains if any, simplifier edits retained, and the preserved user-owned files.
