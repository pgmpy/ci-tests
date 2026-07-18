# Golden Fixture Parity Gate Design

**Date:** 2026-07-18

**Status:** Approved for implementation

## Context

The refactor introduced fixture consumers for Rust, Python, R, and JavaScript,
but did not commit the fixture they all load or the generator documented in
`CONTRIBUTING.md`. Commit history shows that the original consumers expected 73
cases. A later change added Fisher-Z to every consumer and increased the
contract to 80 cases, again without adding the seven Fisher-Z rows or the
original 73-row artifact.

Consequently, `cargo test -p ci_core` runs all core unit tests successfully and
then fails when `crates/ci-core/tests/golden.rs` tries to open
`tests/fixtures/golden.json`. The other three binding suites fail at the same
boundary. This is a missing source-of-truth pipeline, rather than a statistical
failure in one binding.

## Goals

- Restore a deterministic, committed 80-case numeric parity fixture.
- Generate expected values independently of the Rust implementation.
- Match the statistical semantics of the corresponding pgmpy CI tests,
  including this project's explicit Yates configuration and uniform result
  fields.
- Make fixture drift detectable in CI.
- Assert every defined result field in Rust, Python, R, and JavaScript.
- Make failures identify a stable scenario instead of only an array index.
- Correct stale documentation and workflow comments that still say 73 cases.

## Non-goals

- Reintroducing runtime benchmarks; that is a separate design task.
- Adding new CI-test algorithms or changing their public APIs.
- Using the golden fixture to test rejected inputs or expected errors. Those
  behaviors remain focused unit/API tests because a cross-language numeric
  fixture represents successful result values.
- Adding pgmpy as a build, runtime, or test dependency of any binding.

## Considered Approaches

### 1. NumPy/SciPy reference generator (selected)

A Python generator owns deterministic input scenarios and evaluates them with
NumPy/SciPy implementations of the documented pgmpy formulas. It never imports
or invokes `ci_core` or a language binding. The generated JSON is committed so
normal Rust, R, and JavaScript tests do not require Python or SciPy.

This keeps the oracle independent, lightweight, inspectable, and usable by all
four suites.

### 2. Generate directly through pgmpy

This is close to the upstream surface, but pgmpy is a comparatively heavy and
evolving dependency. Its names and result metadata do not map one-to-one to
this project's unified contract, and this project additionally exposes Yates
as an option. Direct generation would therefore still require adapter formulas
while making regeneration more fragile.

### 3. Generate through `ci_core`

This is the simplest implementation, but it is circular: the expected values
would be produced by the implementation being tested. It could detect binding
marshalling bugs but not statistical regressions in the Rust core, so it is
rejected.

## Architecture

The pipeline has three layers:

1. `tests/fixtures/generate_golden.py` defines the case catalog, evaluates the
   independent reference formulas, validates the resulting cases, and either
   writes or checks the canonical artifact.
2. `tests/fixtures/golden.json` is the generated, language-neutral acceptance
   contract committed to Git.
3. The four existing fixture consumers construct their idiomatic `Dataset`, run
   the named test, and compare all four result fields with the fixture.

No consumer generates expectations. No reference calculation calls into Rust.

## Fixture Contract

The top-level representation remains a JSON array to avoid unnecessary schema
churn. Each case contains:

```json
{
  "id": "discrete-associated-2x2-yates-chi-squared",
  "test": "chi_squared",
  "params": { "yates": true },
  "columns": {
    "X": { "kind": "discrete", "values": [0, 0, 1, 1] },
    "Y": { "kind": "discrete", "values": [0, 1, 0, 1] }
  },
  "x": "X",
  "y": "Y",
  "z": [],
  "expected": {
    "statistic": 0.0,
    "p_value": 1.0,
    "dof": 1,
    "effect_size": 0.0
  }
}
```

Contract rules:

- `id` is unique, stable, lowercase kebab-case, and used in failure messages.
- `test` is one of the eight stable registry names already used by the
  consumers.
- `params` is always present. It contains `yates` for a discrete test,
  `delta_threshold` for Pearson equivalence, and is empty for Pearson
  correlation and Fisher-Z.
- Every column has one supported `kind` and a non-empty numeric `values` array.
  All columns in a case have equal length.
- `x`, `y`, and every member of `z` name distinct columns present in `columns`.
- `expected` always contains `statistic`, `p_value`, `dof`, and `effect_size`.
  `dof` is `null` only for algorithms that do not report it.
- The generated numeric fixture contains only finite JSON numbers. Degenerate
  cases that yield errors, NaN, or infinity belong in algorithm unit tests.

## Case Catalog

The generator emits exactly 80 cases with this fixed distribution:

| Test family | Scenario count | Expanded cases |
| --- | ---: | ---: |
| Five discrete power-divergence tests | 12 shared scenarios | 60 |
| Pearson correlation | 7 | 7 |
| Pearson equivalence | 6 | 6 |
| Fisher-Z | 7 | 7 |
| **Total** |  | **80** |

The 12 discrete scenarios cover:

1. balanced independent 2x2 with Yates enabled;
2. associated 2x2 with Yates enabled;
3. the same associated 2x2 with Yates disabled;
4. unbalanced 2x2 with Yates enabled;
5. the same unbalanced 2x2 with Yates disabled;
6. an unconditional 2x3 table;
7. an unconditional 3x2 table;
8. an unconditional 3x3 table;
9. conditioning on one binary variable;
10. conditioning on one three-level variable;
11. conditioning on two variables; and
12. sparse strata in which globally known levels are inactive in some strata.

Every discrete scenario is expanded across `chi_squared`, `log_likelihood`,
`cressie_read`, `freeman_tukey`, and `modified_likelihood`. Tables are chosen so
every cell inside an active sub-table has a positive observed count and all five
statistics are finite. Globally known rows or columns may still be inactive in
an individual sparse stratum and are removed before the SciPy call.

The seven Pearson scenarios cover near-zero, positive, and negative
unconditional relationships; confounding removed by one conditioning variable;
residual association after one conditioning variable; two conditioning
variables; and scale/offset invariance. The six equivalence scenarios exercise
correlations inside and outside multiple positive margins, both unconditional
and conditional. Fisher-Z reuses six finite Pearson scenarios and adds one
perfect-correlation case to pin its pgmpy-compatible clipping behavior.

All generated inputs are fixed arrays. Random input generation is avoided, even
with a seed, so the fixture does not depend on a random-number implementation.

## Reference Calculations

### Discrete power-divergence family

The generator partitions rows by each observed combination of `z`. Within each
stratum it removes rows and columns with zero marginal counts and calls
`scipy.stats.chi2_contingency` with the case's `correction` value and the
numeric lambda:

- `chi_squared`: `1.0`
- `log_likelihood`: `0.0`
- `cressie_read`: `2.0 / 3.0`
- `freeman_tukey`: `-0.5`
- `modified_likelihood`: `-1.0`

Stratum statistics and degrees of freedom are summed. The aggregate p-value is
`scipy.stats.chi2.sf(statistic, dof)`. Effect size follows pgmpy's Cramer's V
contract:

`sqrt(statistic / (n * max(min(k_x, k_y) - 1, 1)))`.

Here `k_x` and `k_y` are global observed cardinalities, matching `Dataset`
factorization.

### Pearson correlation

Without `z`, the reference obtains the raw correlation with
`scipy.stats.pearsonr`. With `z`, it regresses both variables on `[1, Z]` with
`numpy.linalg.lstsq` and correlates the residuals. In both paths, the uniform
result contract records `dof = n - |Z| - 2` and computes the two-sided p-value
with Student's t survival function using that dof. The effect size is the
absolute raw correlation.

### Pearson equivalence

The reference obtains the same partial correlation, clips it to
`[-0.999999, 0.999999]`, and applies pgmpy's Fisher-transform TOST formulas.
The statistic is the clipped correlation's `atanh`, the p-value is the maximum
of the two one-sided normal probabilities, `dof` is null, and effect size is the
absolute un-clipped correlation.

### Fisher-Z

The reference obtains the same partial correlation, clips it to
`[-0.999999, 0.999999]`, and computes
`sqrt(n - |Z| - 3) * atanh(rho)`. Its p-value is the two-sided standard-normal
tail, `dof` is null, and effect size is the absolute un-clipped correlation.

## Deterministic Generation and Drift Checking

The generator supports two modes:

```bash
python tests/fixtures/generate_golden.py
python tests/fixtures/generate_golden.py --check
```

Normal mode validates and atomically replaces `golden.json`. Check mode builds
the same canonical document in memory, performs no writes, and exits nonzero
with a concise regeneration instruction if the committed bytes differ or the
file is missing.

To reduce harmless BLAS/platform last-bit variation, computed floats are
canonicalized to 12 significant decimal digits before serialization. JSON is
written with sorted keys, two-space indentation, a trailing newline, and
`allow_nan=False`. Twelve significant digits are substantially tighter than
the consumers' absolute `1e-7` acceptance tolerance.

`tests/fixtures/requirements.txt` pins `numpy==2.4.2` and `scipy==1.17.0`, the
reference versions used to regenerate and check the oracle. Normal binding
tests consume only the JSON and do not install these dependencies.

Before output, validation rejects:

- a case count or per-test distribution different from the table above;
- duplicate or malformed IDs;
- unknown tests, params, or column kinds;
- mismatched or empty column lengths;
- missing, repeated, or invalid query columns;
- missing result keys, non-integral dof, or non-finite numbers.

## Consumer Changes

The Rust consumer already asserts all four fields. It will deserialize `id` and
include it in assertion failures.

Python, R, and JavaScript will add the currently omitted `effect_size` /
`effectSize` assertion using the same `1e-7` absolute tolerance. Their test
descriptions will say all four fields. Every consumer continues to assert the
80-case count and the complete eight-test name set.

The R package will add `jsonlite` to `Suggests`, because the checked-in test
suite imports it. CI no longer needs to describe it as an undeclared test
dependency.

## CI and Documentation

Python CI gets one platform-independent fixture job on Python 3.12. It installs
pytest plus `tests/fixtures/requirements.txt`, runs
`pytest tests/fixtures/test_generate_golden.py`, and runs
`generate_golden.py --check`. Binding matrices continue to read the committed
fixture and do not regenerate it.

Workflow comments in Rust, Python, R, and JavaScript are updated from 73 to 80
cases. `README.md` and `CONTRIBUTING.md` document the write/check commands, the
reference dependencies, and the complete four-field parity contract. The
repository tree stops claiming a benchmark directory exists.

## Testing Strategy

Implementation follows red-green-refactor:

1. Add focused generator tests that initially fail because the generator and
   artifact do not exist. They pin the schema, exact 80-case distribution,
   uniqueness, determinism, selected known reference values, and `--check`
   behavior.
2. Implement the generator and produce the fixture until those tests pass.
3. Run the existing Rust golden integration test, which currently reproduces
   the missing-file failure, and make it pass against the independent fixture.
4. Add effect-size assertions to each binding suite and run each suite where
   its toolchain is available.
5. Run formatting, linting, generator drift, and the complete core suite.

Unavailable local language toolchains are reported explicitly; their CI
commands and fixture consumers are still updated and statically inspected.

## Acceptance Criteria

- `tests/fixtures/golden.json` and its generator are tracked in Git.
- Running the generator twice produces byte-identical output.
- `generate_golden.py --check` succeeds on the committed fixture and fails on a
  missing or modified fixture without changing it.
- The fixture contains exactly 80 valid, uniquely identified cases with the
  specified per-test distribution.
- `cargo test -p ci_core` passes, including all 80 golden cases.
- Every available binding suite passes and compares all four result fields.
- CI detects generator/fixture drift independently from binding builds.
- No workflow or contributor documentation still claims 73 cases or a present
  benchmark directory.
