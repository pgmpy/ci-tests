# Task 4 Report: Centralize Discrete-Family Unit-Test Contracts

## Summary

Added a test-only shared discrete dataset helper and table-driven contract for
the five power-divergence tests. The contract checks the balanced independent
table, degrees of freedom, p-value, and shared metadata. Removed only the
duplicated local dataset builders plus the matching independent-table and
metadata tests; retained the Yates, lambda/effect-size, wrong-kind,
structural-zero, conditional, and Cramer's-V regression tests.

The canonical test sources were synchronized into the packaged R `ci-core`.
No production behavior, July documentation, or golden fixture bytes changed.

## Files

- `crates/ci-core/src/ci_tests/discrete_common.rs`
- `crates/ci-core/src/ci_tests/chi_squared.rs`
- `crates/ci-core/src/ci_tests/log_likelihood.rs`
- `crates/ci-core/src/ci_tests/cressie_read.rs`
- `crates/ci-core/src/ci_tests/freeman_tukey.rs`
- `crates/ci-core/src/ci_tests/modified_likelihood.rs`
- `crates/ci-r/src/rust/ci-core/src/ci_tests/{discrete_common,chi_squared,log_likelihood,cressie_read,freeman_tukey,modified_likelihood}.rs`
- `.superpowers/sdd/2026-08-13-first-wave-simplification/task-4-report.md`

## TDD Evidence

RED: added `family_members_share_contract` before defining
`assert_discrete_contract` or `discrete_dataset`, then ran:

```text
cargo test -p ci_core ci_tests::discrete_common::tests::family_members_share_contract
```

The test compilation failed as expected with `E0425`: cannot find function
`assert_discrete_contract` in scope.

GREEN: added the test-only dataset and contract helpers, then reran the same
targeted command:

```text
running 1 test
test ci_tests::discrete_common::tests::family_members_share_contract ... ok
```

## Verification Commands and Results

```text
python crates/ci-r/tools/sync_package_assets.py --sync
cargo fmt --all
cargo test -p ci_core
cargo clippy -p ci_core --all-targets -- -D warnings
python crates/ci-r/tools/sync_package_assets.py --check
git diff --check
```

All commands exited successfully. `cargo test -p ci_core` passed 59 unit
tests, the 80-case golden-fixture integration test, and the doc test. Clippy
reported no warnings; both R asset commands reported that package assets match
canonical sources.

## Golden Fixture Hashes

```text
8b986cda99c190bb20d570225ff6fbdf9e3c82cdfd24b413f44d89651a127b68  tests/fixtures/golden.json
8b986cda99c190bb20d570225ff6fbdf9e3c82cdfd24b413f44d89651a127b68  crates/ci-r/tests/testthat/fixtures/golden.json
```

`cmp -s` confirmed the two fixtures remain byte-identical, and a path-scoped
`git diff --exit-code` confirmed neither fixture changed.

## Concerns

None.
