# First-Wave Package Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the approved dependency, npm, CI, test, and documentation duplication without changing numerical results, public APIs, golden bytes, or R archive self-containment.

**Architecture:** Keep `citest` as the only statistics implementation and retain each language's existing thin binding. Simplify inputs and maintenance boundaries around that core: a minimal `statrs` feature set, one npm project, shared discrete-family test contracts, leaner CI, and canonical top-level documentation.

**Tech Stack:** Rust/Cargo, wasm-bindgen/wasm-pack, npm/Vitest/ESLint/Prettier, Python/pytest/PyYAML, R/extendr/devtools, GitHub Actions YAML.

## Global Constraints

- Preserve all eight test types, constructors, fields, metadata, and binding APIs.
- Preserve `tests/fixtures/golden.json` and the packaged R fixture byte-for-byte.
- Keep the R package's synchronized `citest`, package fixture, lockfile, and vendor archive self-contained and offline-buildable.
- Do not hand-edit generated extendr wrappers, `.Rd` files, Cargo/npm locks, golden JSON, or vendor contents.
- Keep `statrs` at `0.18.0`, R Rust MSRV at `1.81`, wasm-pack at `0.15.0`, and Python at `>=3.10,<3.15`.
- Do not alter Gram/residual correlation paths, strata caching/partitioning, the registry design, or binding conversion code.
- Never edit or stage the pre-existing modified July plan and design files.

---

### Task 1: Prune the Rust dependency graph and rebuild R package inputs

**Files:**
- Modify: `tests/test_release_configuration.py`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock` (generated)
- Modify: `crates/citest-js/Cargo.toml`
- Modify: `crates/citest-r/src/rust/Cargo.toml`
- Modify: `crates/citest-r/src/rust/Cargo.lock` (generated)
- Modify: `crates/citest-r/src/rust/citest/Cargo.toml` (synchronized)
- Modify: `crates/citest-r/src/rust/vendor.tar.xz` (generated)

**Interfaces:**
- Consumes: workspace dependency key `workspace.dependencies.statrs` and the R asset synchronization CLI.
- Produces: the same `statrs` 0.18.0 CDF API with default `nalgebra`/`rand` features disabled, plus matching root/R lockfiles and an audited offline vendor archive.

- [ ] **Step 1: Add the failing dependency-boundary test**

Append a test that loads the root, R workspace, and JavaScript Cargo manifests and asserts:

```python
def test_rust_distribution_dependency_tree_is_minimal() -> None:
    root = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text())
    r_rust = tomllib.loads((R_RUST / "Cargo.toml").read_text())
    expected = {"version": "0.18.0", "default-features": False}
    assert root["workspace"]["dependencies"]["statrs"] == expected
    assert r_rust["workspace"]["dependencies"]["statrs"] == expected

    js = tomllib.loads((REPO_ROOT / "crates" / "citest" / "Cargo.toml").read_text())
    assert "getrandom" not in js["dependencies"]
    assert "getrandom" not in js.get("target", {}).get(
        "cfg(target_arch = \"wasm32\")", {}
    ).get("dependencies", {})
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest tests/test_release_configuration.py::test_rust_distribution_dependency_tree_is_minimal -q`

Expected: FAIL because `default-features` is absent and `citest` declares both `getrandom` versions.

- [ ] **Step 3: Make the minimal manifest changes**

Use this workspace dependency in both `Cargo.toml` and `crates/citest-r/src/rust/Cargo.toml`:

```toml
statrs = { version = "0.18.0", default-features = false }
```

Respect each file's existing quote style. Remove the unconditional and target-specific `getrandom` entries from `crates/citest-js/Cargo.toml`; remove the now-empty target dependency table.

- [ ] **Step 4: Regenerate lockfiles and synchronize packaged core inputs**

Run:

```bash
cargo generate-lockfile
cargo generate-lockfile --manifest-path crates/citest-r/src/rust/Cargo.toml
python crates/citest-r/tools/sync_package_assets.py --sync
```

Expected: root and R locks no longer contain the unused `nalgebra`/`rand` subtree, and the packaged core manifest matches the canonical one.

- [ ] **Step 5: Rebuild the R vendor archive from the R lockfile**

Run from the repository root:

```bash
vendor_tmp=$(mktemp -d /tmp/cir-vendor.XXXXXX)
cargo vendor --locked --manifest-path crates/citest-r/src/rust/Cargo.toml "$vendor_tmp/vendor"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$vendor_tmp" -cJf crates/citest-r/src/rust/vendor.tar.xz vendor
```

The temporary directory is outside the repository. The archive must contain exactly the registry packages present in `crates/citest-r/src/rust/Cargo.lock`.

- [ ] **Step 6: Verify GREEN and numerical stability**

Run:

```bash
pytest tests/test_release_configuration.py tests/test_r_package_assets.py -q
cargo tree -p citest --edges normal
cargo test -p citest
cargo clippy -p citest --all-targets -- -D warnings
WASM_PACK_CACHE=/tmp/task7-wasm-pack-cache wasm-pack build crates/citest-js --target nodejs
npm --prefix crates/citest-js test
```

Expected: all tests pass; `cargo tree` contains no `nalgebra`, `rand`, or `getrandom`; WASM and all 103 JS tests pass.

- [ ] **Step 7: Commit the dependency pruning**

```bash
git add Cargo.toml Cargo.lock tests/test_release_configuration.py crates/citest-js/Cargo.toml crates/citest-r/src/rust/Cargo.toml crates/citest-r/src/rust/Cargo.lock crates/citest-r/src/rust/citest crates/citest-r/src/rust/vendor.tar.xz
git commit -m "build: trim unused statistical dependencies"
```

---

### Task 2: Collapse JavaScript onto one npm project

**Files:**
- Modify: `tests/test_release_configuration.py`
- Delete: `crates/citest-js/tests/package.json`
- Delete: `crates/citest-js/tests/package-lock.json`
- Modify: `.github/workflows/js.yml`

**Interfaces:**
- Consumes: `crates/citest-js/package.json`, its lockfile, and the existing `tests/*.test.js` discovery convention.
- Produces: one npm dependency graph used by both existing JavaScript CI jobs; `npm test` remains the test entry point.

- [ ] **Step 1: Add the failing single-project test**

Add a test that asserts the nested manifests are absent and both JS jobs use the root lock and working directory:

```python
def test_js_uses_one_npm_project() -> None:
    js = REPO_ROOT / "crates" / "citest"
    assert not (js / "tests" / "package.json").exists()
    assert not (js / "tests" / "package-lock.json").exists()

    workflow = load_workflow("js.yml")
    for job_name in ("js-lint", "js-test"):
        job = workflow["jobs"][job_name]
        setup = next(step for step in job["steps"] if step.get("uses") == "actions/setup-node@v4")
        assert setup["with"]["cache-dependency-path"] == "crates/citest-js/package-lock.json"
        npm_steps = [step for step in job["steps"] if step.get("run") in {"npm ci", "npm test"}]
        assert all(step["working-directory"] == "crates/citest-js" for step in npm_steps)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest tests/test_release_configuration.py::test_js_uses_one_npm_project -q`

Expected: FAIL because both nested npm files exist and the test job points at `crates/citest-js/tests`.

- [ ] **Step 3: Delete the nested npm project and update CI**

Delete the two nested package files. In `js-test`, change `cache-dependency-path` and both npm step working directories to `crates/citest-js`; rename “Install test dependencies” to “Install dependencies”. Update its verification comment to say the root script discovers `tests/*.test.js`. Preserve job IDs, display names, dependencies, and OS matrix.

- [ ] **Step 4: Verify the clean-install boundary and GREEN test**

Run:

```bash
npm_config_cache=/tmp/ci-tests-first-wave-npm npm --prefix crates/citest-js ci
npm --prefix crates/citest-js test
(cd crates/citest-js && npx prettier --check .)
(cd crates/citest-js && npx eslint .)
pytest tests/test_release_configuration.py::test_js_uses_one_npm_project -q
```

Expected: one root lock installs successfully; both JS files run for 103 passing tests; formatting, lint, and the structural test pass.

- [ ] **Step 5: Commit the npm consolidation**

```bash
git add .github/workflows/js.yml tests/test_release_configuration.py crates/citest-js/tests/package.json crates/citest-js/tests/package-lock.json
git commit -m "build(js): use one npm dependency graph"
```

---

### Task 3: Remove redundant CI setup and build steps

**Files:**
- Modify: `tests/test_release_configuration.py`
- Modify: `.github/workflows/docs.yml`
- Modify: `.github/workflows/rust.yml`

**Interfaces:**
- Consumes: existing workflow job IDs and test commands.
- Produces: unchanged required checks without OpenBLAS installation or a redundant pre-test Cargo build.

- [ ] **Step 1: Add the failing workflow-efficiency test**

```python
def test_ci_omits_unused_openblas_and_redundant_core_build() -> None:
    docs = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text()
    assert "libopenblas-dev" not in docs

    rust = load_workflow("rust.yml")
    steps = rust["jobs"]["rust_test"]["steps"]
    commands = [step.get("run", "") for step in steps if isinstance(step, dict)]
    assert not any("cargo build -p citest" in command for command in commands)
    assert any("cargo test -p citest" in command for command in commands)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest tests/test_release_configuration.py::test_ci_omits_unused_openblas_and_redundant_core_build -q`

Expected: FAIL on both stale workflow commands.

- [ ] **Step 3: Remove only the redundant workflow steps**

Delete the three `Install system dependencies` steps from `docs.yml` and the `Build` step from `rust.yml`. Leave triggers, permissions, concurrency, job IDs/names, matrices, toolchain actions, caches, and test commands unchanged.

- [ ] **Step 4: Verify workflow structure and GREEN test**

Run:

```bash
pytest tests/test_release_configuration.py -q
python -c "import pathlib, yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path('.github/workflows').glob('*.yml')]; print('workflow-yaml-ok')"
cargo doc --no-deps -p citest
```

Expected: release-configuration tests, YAML parsing, and Rust documentation build pass.

- [ ] **Step 5: Commit the CI cleanup**

```bash
git add .github/workflows/docs.yml .github/workflows/rust.yml tests/test_release_configuration.py
git commit -m "ci: remove redundant build setup"
```

---

### Task 4: Centralize discrete-family unit-test contracts

**Files:**
- Modify: `crates/citest/src/ci_tests/discrete_common.rs`
- Modify: `crates/citest/src/ci_tests/chi_squared.rs`
- Modify: `crates/citest/src/ci_tests/log_likelihood.rs`
- Modify: `crates/citest/src/ci_tests/cressie_read.rs`
- Modify: `crates/citest/src/ci_tests/freeman_tukey.rs`
- Modify: `crates/citest/src/ci_tests/modified_likelihood.rs`
- Synchronize: `crates/citest-r/src/rust/citest/**`

**Interfaces:**
- Consumes: the five concrete `CITest` implementations and `discrete_meta` contract.
- Produces: a `#[cfg(test)] pub(crate) fn discrete_dataset(...) -> Dataset` helper and one table-driven family contract test; production code and public types remain unchanged.

- [ ] **Step 1: Add the common contract test before its helper**

Under `#[cfg(test)] mod tests` in `discrete_common.rs`, import all five types and call a not-yet-defined `assert_discrete_contract` for each `(name, test)` pair. The helper contract must assert an independent balanced table yields near-zero statistic, `dof == Some(1)`, `p_value > 0.99`, and metadata `[Discrete]`, symmetric, `PValueGe`.

- [ ] **Step 2: Run the test and confirm RED**

Run: `cargo test -p citest ci_tests::discrete_common::tests::family_members_share_contract`

Expected: compilation fails because `assert_discrete_contract` and/or `discrete_dataset` has not been defined.

- [ ] **Step 3: Implement the test-only dataset and contract helpers**

Add this test-only dataset interface to `discrete_common.rs`:

```rust
#[cfg(test)]
pub(crate) fn discrete_dataset(cols: Vec<(&str, Vec<f64>)>) -> Dataset {
    Dataset::from_columns(
        cols.into_iter()
            .map(|(name, values)| (name.to_string(), ColumnKind::Discrete, values))
            .collect(),
    )
    .unwrap()
}
```

Implement `assert_discrete_contract(test: &dyn CITest, expected_name: &str)` inside the test module and invoke it for `ChiSquared`, `LogLikelihood`, `CressieRead`, `FreemanTukey`, and `ModifiedLikelihood`.

- [ ] **Step 4: Verify the new common contract is GREEN**

Run: `cargo test -p citest ci_tests::discrete_common::tests::family_members_share_contract`

Expected: PASS for all five implementations.

- [ ] **Step 5: Replace duplicate helpers and remove duplicate contract tests**

Import `discrete_dataset as ds` in each concrete module. Delete each local `ds`, unconditional-independent test, metadata test, and now-unused `DataType`/`IndependenceRule` imports. Keep every unique regression described in the global constraints; retain `ColumnKind` in `chi_squared.rs` for its mixed-kind error test.

- [ ] **Step 6: Synchronize the R copy and run all core gates**

Run:

```bash
python crates/citest-r/tools/sync_package_assets.py --sync
cargo fmt --all
cargo test -p citest
cargo clippy -p citest --all-targets -- -D warnings
python crates/citest-r/tools/sync_package_assets.py --check
```

Expected: core tests and the 80-case golden consumer pass; Clippy and R asset drift are clean. The total unit-test count may decrease because five duplicate assertions are replaced by one table-driven test, but unique regressions remain.

- [ ] **Step 7: Commit the test consolidation**

```bash
git add crates/citest/src/ci_tests crates/citest-r/src/rust/citest
git commit -m "test(core): centralize discrete family contracts"
```

---

### Task 5: Repair leaf documentation, example code, and package metadata

**Files:**
- Modify: `tests/test_release_configuration.py`
- Modify: `crates/README.md`
- Modify: `crates/citest/README.md`
- Modify: `crates/citest-js/README.md`
- Modify: `crates/citest-js/examples/test.html`
- Modify: `crates/citest/Cargo.toml`
- Modify: `crates/citest-python/Cargo.toml`
- Modify: `crates/citest-js/Cargo.toml`
- Modify: `crates/citest-js/package.json`
- Synchronize: `crates/citest-r/src/rust/citest/Cargo.toml`

**Interfaces:**
- Consumes: the current data-bound APIs documented by the root README.
- Produces: concise leaf documentation and an executable browser example using `Dataset` and `PearsonCorrelation`; Cargo/npm metadata identifies the pgmpy repository and MIT license.

- [ ] **Step 1: Add the failing stale-content test**

Add assertions that the stale terms are gone and current identifiers are present:

```python
def test_leaf_docs_examples_and_package_metadata_are_current() -> None:
    crates_readme = (REPO_ROOT / "crates" / "README.md").read_text()
    core_readme = (REPO_ROOT / "crates" / "citest" / "README.md").read_text()
    example = (REPO_ROOT / "crates" / "citest" / "examples" / "test.html").read_text()
    assert "Asynchronous API" not in crates_readme
    assert "TestRegistry" not in core_readme
    assert "src/tests/" not in core_readme
    assert "pearson_correlation_test" not in example
    assert "new Dataset" in example
    assert "new PearsonCorrelation" in example

    for relative in ("crates/citest/Cargo.toml", "crates/citest-python/Cargo.toml", "crates/citest-js/Cargo.toml"):
        package = tomllib.loads((REPO_ROOT / relative).read_text())["package"]
        assert package["license"] == "MIT"
        assert package["repository"] == "https://github.com/pgmpy/ci-tests"
        assert "Your Team" not in " ".join(package.get("authors", []))
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest tests/test_release_configuration.py::test_leaf_docs_examples_and_package_metadata_are_current -q`

Expected: FAIL on the stale README/example strings and incomplete Cargo metadata.

- [ ] **Step 3: Replace stale leaf documentation with concise current guides**

`crates/README.md` should list the four package roles and explain that the root Cargo workspace contains core/Python/JS while R is a standalone source-package workspace. `citest/README.md` should describe `Dataset`, `CITest`, the eight tests, and link to root `README.md`/`CONTRIBUTING.md`. `citest/README.md` should show the current `Dataset` + class API, `wasm-pack build crates/citest-js --target nodejs`, and root `npm ci && npm test` commands without duplicating the full root guide.

- [ ] **Step 4: Rewrite the browser example against the current API**

Import `Dataset` and `PearsonCorrelation` from `../pkg/citest_js.js`, await `init()`, build two continuous columns, construct `new PearsonCorrelation(data)`, call `runTest("X", "Y", [])`, and render `JSON.stringify(result, null, 2)`. Remove every removed function-style export.

- [ ] **Step 5: Correct package metadata**

For all three Cargo packages use:

```toml
authors = ["Ankur Ankan <ankurankan@gmail.com>"]
license = "MIT"
repository = "https://github.com/pgmpy/ci-tests"
```

Remove `exclude = [""]` from `citest`. Replace the npm description with `WebAssembly bindings for data-bound conditional-independence testing` and add the pgmpy repository URL using npm's standard repository object.

- [ ] **Step 6: Synchronize the packaged core and verify GREEN**

Run:

```bash
python crates/citest-r/tools/sync_package_assets.py --sync
pytest tests/test_release_configuration.py::test_leaf_docs_examples_and_package_metadata_are_current tests/test_r_package_assets.py -q
cargo metadata --no-deps --format-version 1 > /tmp/ci-tests-first-wave-metadata.json
(cd crates/citest-js && npx prettier --check README.md examples/test.html package.json)
python crates/citest-r/tools/sync_package_assets.py --check
```

Expected: structural tests, manifest parsing, formatting, and asset drift checks pass.

- [ ] **Step 7: Commit the documentation and metadata cleanup**

```bash
git add tests/test_release_configuration.py crates/README.md crates/citest/README.md crates/citest-js/README.md crates/citest-js/examples/test.html crates/citest/Cargo.toml crates/citest-python/Cargo.toml crates/citest-js/Cargo.toml crates/citest-js/package.json crates/citest-r/src/rust/citest/Cargo.toml
git commit -m "docs: align package guides with data-bound API"
```

---

### Task 6: Run full cross-language and archive verification

**Files:**
- Verify only; fix only failures caused by Tasks 1–5 and stage those fixes explicitly.

**Interfaces:**
- Consumes: the final simplified tree.
- Produces: evidence that all language surfaces, parity contracts, generated assets, and the standalone R source archive remain correct.

- [ ] **Step 1: Verify repository cleanliness boundaries**

Run:

```bash
git status --short
git diff --check
python tests/fixtures/generate_golden.py --check
python crates/citest-r/tools/sync_package_assets.py --check
cmp tests/fixtures/golden.json crates/citest-r/tests/testthat/fixtures/golden.json
```

Expected: only the two user-owned July files are unstaged; no whitespace or asset drift; fixture bytes match.

- [ ] **Step 2: Run Rust and fixture-tool gates**

```bash
cargo fmt --all -- --check
cargo clippy -p citest --all-targets -- -D warnings
cargo test -p citest
pytest tests -q
python tests/fixtures/check_pgmpy_parity.py --pgmpy-source /home/ankur/work/pgmpy/pgmpy
```

Expected: Rust and Python repository tests pass; parity reports `compared=65 skipped=15 divergences=2 failed=0`.

- [ ] **Step 3: Run Python binding gates**

```bash
python -m ruff format --check crates/citest-python
python -m ruff check crates/citest-python
(cd crates/citest-python && mypy test/)
pytest crates/citest-python/test/ -q
```

Expected: format/lint/type gates pass and all Python API/golden tests pass.

- [ ] **Step 4: Run JavaScript/WASM gates**

```bash
WASM_PACK_CACHE=/tmp/task7-wasm-pack-cache wasm-pack build crates/citest-js --target nodejs
npm_config_cache=/tmp/ci-tests-first-wave-npm npm --prefix crates/citest-js ci
(cd crates/citest-js && npx prettier --check .)
(cd crates/citest-js && npx eslint .)
npm --prefix crates/citest-js test
```

Expected: WASM builds and all JS gates pass from the single root npm project.

- [ ] **Step 5: Run R lint, tests, and source-archive check**

Create the compiler-name shims and isolated directories:

```bash
mkdir -p /tmp/ci-tests-r-toolchain/bin /tmp/ci-tests-r-lib /tmp/ci-tests-r-first-wave
ln -sf /usr/bin/gcc /tmp/ci-tests-r-toolchain/bin/x86_64-conda-linux-gnu-cc
ln -sf /usr/bin/g++ /tmp/ci-tests-r-toolchain/bin/x86_64-conda-linux-gnu-c++
```

Then run:

```bash
PATH=/tmp/ci-tests-r-toolchain/bin:$PATH LD_LIBRARY_PATH=/home/ankur/miniconda3/envs/expert/lib/R/lib R_LIBS_USER=/tmp/ci-tests-r-lib CARGO_NET_OFFLINE=true /home/ankur/miniconda3/envs/expert/bin/Rscript -e 'rextendr::document("crates/citest-r")'
git diff --exit-code -- crates/citest-r/R/extendr-wrappers.R crates/citest-r/man crates/citest-r/NAMESPACE
PATH=/tmp/ci-tests-r-toolchain/bin:$PATH LD_LIBRARY_PATH=/home/ankur/miniconda3/envs/expert/lib/R/lib R_LIBS_USER=/tmp/ci-tests-r-lib CARGO_NET_OFFLINE=true /home/ankur/miniconda3/envs/expert/bin/Rscript -e 'lints <- lintr::lint_package("crates/citest-r"); print(lints); quit(status = as.integer(length(lints) > 0L))'
PATH=/tmp/ci-tests-r-toolchain/bin:$PATH LD_LIBRARY_PATH=/home/ankur/miniconda3/envs/expert/lib/R/lib R_LIBS_USER=/tmp/ci-tests-r-lib CARGO_NET_OFFLINE=true /home/ankur/miniconda3/envs/expert/bin/Rscript -e 'devtools::test("crates/citest-r", reporter = "summary")'
PATH=/tmp/ci-tests-r-toolchain/bin:$PATH LD_LIBRARY_PATH=/home/ankur/miniconda3/envs/expert/lib/R/lib R_LIBS_USER=/tmp/ci-tests-r-lib CARGO_NET_OFFLINE=true /home/ankur/miniconda3/envs/expert/bin/Rscript -e 'archive <- devtools::build("crates/citest-r", path = "/tmp/ci-tests-r-first-wave", manual = FALSE, vignettes = FALSE); devtools::check_built(archive, cran = FALSE, manual = FALSE, args = "--no-manual", check_dir = "/tmp/ci-tests-r-first-wave/check", error_on = "warning")'
```

Expected: 0 errors, 0 warnings; the existing installed-size NOTE is acceptable. Any regenerated wrapper/manual difference must be attributable to canonical source changes, never a hand edit.

- [ ] **Step 6: Review the final diff and commit verification fixes if any**

Run:

```bash
git diff --stat HEAD~5..HEAD
git diff --check
git status --short
```

If verification required scoped corrections, commit only their explicit paths with `git add <paths>` and a focused `fix:` commit. Leave the July files unstaged.
