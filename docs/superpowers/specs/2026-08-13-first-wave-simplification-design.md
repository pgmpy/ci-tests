# First-Wave Package Simplification Design

## Goal

Reduce build, test, CI, and documentation complexity without changing any
conditional-independence statistic, public binding API, golden-fixture byte, or
R source-package self-containment guarantee.

## Scope

### 1. Rust dependency pruning

- Configure the workspace `statrs` dependency with `default-features = false`.
  The package uses univariate distribution CDFs only and does not use the
  default `nalgebra` or `rand` integrations.
- Remove both direct `getrandom` dependencies from `ci-js`. Once `statrs` no
  longer enables `rand`, neither direct feature-unification dependency is
  needed by the WASM binding.
- Apply the same workspace dependency setting to the standalone R Rust
  workspace, synchronize its packaged `ci-core`, regenerate both Cargo
  lockfiles, and rebuild the R vendor archive from the resulting locked crate
  set.
- Preserve the existing versions of `statrs`, `thiserror`, and all binding
  libraries.

### 2. One JavaScript npm project

- Keep `crates/ci-js/package.json` and its lockfile as the only npm project.
- Delete `crates/ci-js/tests/package.json` and
  `crates/ci-js/tests/package-lock.json`.
- Run Vitest from `crates/ci-js`; test discovery must continue to execute both
  `tests/api.test.js` and `tests/golden.test.js` in the Node environment.
- Update both existing JavaScript CI jobs to install from the root lockfile:
  the lint job performs formatting and ESLint checks, while each OS test job
  builds WASM and runs the root test script. Separate runners still perform
  separate clean installs; no `node_modules` artifact is shared across OSes.

### 3. Remove redundant CI work

- Remove the three OpenBLAS installation steps from documentation jobs. No
  current package or documentation build links to system OpenBLAS.
- Remove the explicit `cargo build -p ci_core` step immediately before
  `cargo test -p ci_core`; Cargo test already builds the same targets.
- Preserve workflow triggers, path filters, concurrency groups, job names, OS
  coverage, and required-check identities.

### 4. Consolidate discrete-family unit-test scaffolding

- Add test-only helpers/contracts under `ci_tests::discrete_common` for the
  repeated dataset construction, independent-table behavior, and common
  metadata invariants of the five power-divergence tests.
- Delete the corresponding duplicate assertions from the five concrete test
  modules.
- Retain all test-specific regressions, including Yates behavior, lambda
  wiring, Cramer's V clamping, wrong-kind handling, and structural-zero
  behavior.
- Do not macro-generate or otherwise change the five production wrapper types.

### 5. Repair stale package documentation and examples

- Replace the stale crate-level READMEs with concise, current descriptions that
  link to the root README and contributing guide.
- Rewrite the JavaScript browser example to use the current data-bound
  `Dataset` and test-class API.
- Replace stale package metadata with the repository's MIT license,
  pgmpy repository URL, and current project descriptions; remove meaningless
  empty manifest fields.
- Keep the root README and contributing guide canonical. Avoid duplicating
  detailed setup or architecture instructions in leaf READMEs.

## Required Boundaries

- The canonical fixture `tests/fixtures/golden.json` and its packaged R copy
  must remain byte-identical and unchanged.
- `crates/ci-r/src/rust/ci-core/**`, the package-local fixture, Cargo lockfile,
  and vendor archive remain present because the R source archive must build
  outside the monorepo and without network access.
- Generated extendr wrappers and `.Rd` files are not hand-edited.
- The Gram-cache/residual-QR paths, strata cache/partition implementation,
  registry shape, and binding conversion layers are out of scope.
- The pre-existing modified July plan and design documents are user-owned and
  must not be edited or staged.

## Verification

The completed tree must pass:

- Rust formatting, Clippy, core tests, and the 80-case Rust golden consumer.
- Python Ruff, MyPy, API tests, and golden tests.
- WASM build, a clean root npm install, Prettier, ESLint, and all JavaScript
  tests from the root npm project.
- R asset drift checking, documentation regeneration drift checking, linting,
  testthat, and a built-source archive check using the `expert` Conda R.
- Golden generator validation and current-pgmpy parity with only the two
  already-declared conditioned Pearson-equivalence `dof` divergences.
- Workflow YAML/release-configuration tests and `git diff --check`.

## Success Criteria

- No numerical or public-API changes.
- One JavaScript npm dependency graph instead of two.
- `statrs` no longer brings the unused `nalgebra`/`rand` subtree into the core,
  bindings, or R vendor archive.
- Repeated discrete unit-test contracts have one definition.
- CI performs no redundant OpenBLAS installation or pre-test Cargo build.
- Leaf documentation and the browser example describe code that exists.
