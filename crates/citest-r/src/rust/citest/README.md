# citest

`citest` is the dependency-light Rust implementation of data-bound
conditional-independence testing.

## API

Build a [`Dataset`](https://github.com/pgmpy/ci-tests/blob/main/crates/citest/src/dataset.rs) from named discrete or continuous columns,
then pass it to a stateless test implementing [`CITest`](https://github.com/pgmpy/ci-tests/blob/main/crates/citest/src/strategy.rs):
`test(&data, x, y, z)` returns the test result, while
`is_independent(&data, x, y, z, alpha)` returns an independence decision.

The eight implementations are `ChiSquared`, `LogLikelihood`, `CressieRead`,
`FreemanTukey`, `ModifiedLikelihood`, `PearsonCorrelation`, `FisherZ`, and
`PearsonEquivalence`.

The root
[README](https://github.com/pgmpy/ci-tests/blob/main/README.md) is the
canonical API guide. See the root
[CONTRIBUTING.md](https://github.com/pgmpy/ci-tests/blob/main/CONTRIBUTING.md)
for development and contribution instructions.
