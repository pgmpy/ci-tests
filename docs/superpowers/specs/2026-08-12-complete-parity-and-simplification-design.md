# Complete Parity and Simplification Design

**Date:** 2026-08-12

**Status:** Approved for implementation

## Context

The design refactor has established the intended data-bound API across the Rust
core and the Python, R, and JavaScript bindings, but its cross-language numeric
acceptance pipeline is incomplete. The four consumers expect an 80-case golden
fixture that is not tracked, three consumers omit effect-size comparisons, and
the language CI jobs stop in formatting or linting before their test matrices.

The existing golden-fixture design remains authoritative. This addendum covers
the requested completion work, an explicit parity comparison with the current
pgmpy implementation, and a final behavior-preserving simplification pass.

## Approaches Considered

### Independent fixture plus optional pgmpy checker (selected)

Keep NumPy/SciPy as the independent source of expected values and add a separate
checker that runs the same successful scenarios through current pgmpy classes.
This retains a non-circular regression oracle while also detecting semantic
drift from the upstream project whose formulas this library follows.

### Generate the fixture through pgmpy

This would make direct parity simple, but expected values would change whenever
pgmpy changes and the normal oracle would no longer be independent of that
implementation. It would also add pgmpy and its transitive dependencies to the
fixture-generation environment.

### One-off parity analysis

A disposable comparison would answer the immediate question but could not be
repeated after future statistical changes. A checked-in optional checker is
only modestly more work and preserves the result as executable documentation.

## Architecture

The completed verification system has four layers:

1. `tests/fixtures/generate_golden.py` builds and validates deterministic cases,
   computes expected values with pinned NumPy/SciPy formulas, and writes or
   checks canonical strict JSON.
2. `tests/fixtures/golden.json` is the committed 80-case language-neutral
   contract.
3. Rust, Python, R, and JavaScript consumers construct their native datasets and
   compare statistic, p-value, degrees of freedom, and effect size. A JSON null
   is compared as an expected absence rather than silently skipped.
4. `tests/fixtures/check_pgmpy_parity.py` optionally imports pgmpy from an
   installed package or an explicit source checkout and compares overlapping
   result semantics against the canonical cases.

The pgmpy checker is not imported by the generator and is not a normal binding
dependency. Fixture drift CI remains based only on pinned NumPy/SciPy.

## Current pgmpy Contract

The parity target is the `dev` branch of `pgmpy/pgmpy`, verified on 2026-08-12
at commit `8a221e915889f77bc3e503528e63df5803cfd43d`. The checker maps this project's
eight names to pgmpy's `ChiSquare`, `LogLikelihood`, `PowerDivergence` with the
appropriate lambda, `ModifiedLogLikelihood`, `Pearsonr`,
`PearsonrEquivalence`, and `FisherZ` classes.

Only shared semantics are treated as strict parity:

- Yates-enabled discrete cases are compared. Yates-disabled cases are reported
  as intentional extensions because current pgmpy always applies its 2x2
  correction and exposes no disable switch.
- Statistic, p-value, effect size, and available degrees-of-freedom metadata are
  compared within the fixture tolerance.
- Differences in object shape, naming, caching, and binding representation are
  outside numeric parity.

The checker exits nonzero on an unexpected mismatch and prints counts for
compared, intentionally skipped, and failed cases. Its tests cover mapping,
field extraction, tolerance handling, and the Yates exception without requiring
pgmpy as a permanent test dependency.

## Binding and Contract Completion

Every fixture case has a stable kebab-case ID used in parameterized test names
and failure messages. Python, R, and JavaScript add their missing effect-size
assertions. All consumers compare null expectations to null/absent actual
values, ensuring the complete four-field result schema is pinned. R declares
`jsonlite` in `Suggests`.

The existing public APIs remain unchanged. The already-correct un-clipped
Pearson-equivalence effect size is retained and its boundary scenario is pinned
by the independent fixture.

## CI, Packaging, and Documentation

Python CI gains a platform-independent fixture drift job. Existing Python,
JavaScript, and R formatting/lint failures are repaired so their test matrices
can run. Tooling needed by CI is declared or pinned instead of downloaded
implicitly. Python's required NumPy runtime dependency and typing configuration
are made consistent with its public package behavior.

Documentation is corrected to describe 80 cases, the fixture regeneration and
check commands, Pearson-correlation degrees of freedom, valid same-kind usage
examples, and the actual repository tree. Dependencies left solely by removed
benchmarks or unused property tests are removed after usage verification.

## Simplification Pass

No dedicated repository simplifier is installed. After all behavior and parity
checks are green, the available behavior-preserving tools are run in fix mode:
Rust Clippy/rustfmt, Ruff, Prettier/ESLint, and R styler/lintr where the toolchain
is available. Generated binding files are handled by their generator rather
than manually redesigned.

Every automatic edit is reviewed. Only changes that reduce duplication,
unnecessary branching, unused dependencies, or formatting noise without
altering the public API or numeric behavior are kept. The full verification
matrix is rerun after simplification.

## Error Handling

Generator validation rejects malformed schemas, non-finite JSON values,
duplicate IDs, invalid query columns, and unexpected case distributions before
writing. Check mode never changes the fixture. The pgmpy checker distinguishes
unsupported intentional differences from genuine mismatches and never rewrites
expected values to make a comparison pass.

## Acceptance Criteria

- The independent generator, tests, requirements, and canonical 80-case JSON
  artifact are tracked and deterministic.
- Generator check mode detects missing or modified artifacts without writing.
- All four consumers compare the complete result contract with stable case IDs.
- Full Rust and every locally available binding suite pass.
- CI includes fixture drift detection and no language test matrix is blocked by
  known formatting or lint failures.
- The optional checker passes against current pgmpy for every shared semantic
  case and reports intentional Yates-disabled exclusions separately.
- Documentation and package metadata match the implemented behavior.
- Simplification tools introduce no test, parity, API, or fixture drift.

## Non-goals

- Making pgmpy a build, runtime, or normal test dependency.
- Removing this library's explicit Yates-disable extension.
- Adding CI algorithms beyond the existing eight.
- Restoring runtime benchmarks as part of this change.
