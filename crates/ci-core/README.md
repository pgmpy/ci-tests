# ci-core

`ci-core` is the dependency-light Rust implementation of data-bound
conditional-independence testing.

## API

Build a [`Dataset`](src/dataset.rs) from named discrete or continuous columns,
then bind it to a test implementing [`CITest`](src/strategy.rs). A bound test
answers `run_test(x, y, z)` and `is_independent(x, y, z, significance_level)`
queries without rebuilding the dataset.

The eight implementations are `ChiSquared`, `LogLikelihood`, `CressieRead`,
`FreemanTukey`, `ModifiedLikelihood`, `PearsonCorrelation`, `FisherZ`, and
`PearsonEquivalence`.

The root [README](../../README.md) is the canonical API guide. See the root
[CONTRIBUTING.md](../../CONTRIBUTING.md) for development and contribution
instructions.
