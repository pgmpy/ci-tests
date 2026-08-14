# Tolerant Golden Fixture Check Design

## Context

The golden fixture generator derives expected statistical results with NumPy and
SciPy. Identical source code and pinned package versions can produce harmless
last-digit differences when GitHub changes runner images, CPUs, BLAS kernels, or
system math libraries. The current freshness check compares rendered JSON bytes,
so those differences incorrectly report a stale fixture even though every
binding accepts results within an absolute tolerance of `1e-7`.

## Design

Keep fixture rendering and writing unchanged. Change only the freshness check:

- Parse the committed fixture and freshly rendered fixture as JSON.
- Require the list order, case identifiers, keys, column data, query fields,
  parameters, and `dof` values to match exactly.
- Compare only `expected.statistic`, `expected.p_value`, and
  `expected.effect_size` with an absolute tolerance of `1e-7` and no relative
  tolerance.
- Treat missing files, invalid JSON, type changes, missing or additional keys,
  and differences outside the tolerance as stale.

This retains detection of meaningful generator and scenario changes while
ignoring numerical noise already accepted by the parity tests.

## Testing

Extend the fixture-generator tests to prove that the checker:

1. accepts sub-tolerance changes to each tolerated expected-result field;
2. rejects an expected-result change larger than the tolerance;
3. rejects changes to fixture inputs or structure; and
4. continues to reject malformed or missing fixtures.

Run the fixture tooling tests, the generator `--check` command, formatting and
lint checks, and the cross-language asset synchronization check.
