# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Rust core, the Python package and the npm package share a version. The
CRAN package may trail by a patch release, because CRAN review latency is
outside the project's control.

## [Unreleased]

## [0.1.0] — 2026-08-14

First public release. Eight conditional-independence tests implemented once in
a Rust core and exposed through Python, R and JavaScript/WebAssembly bindings,
with an 80-case golden fixture as the cross-language numeric parity gate.

### Added

- `citest` on crates.io: the Rust core — `Dataset`, the `CITest` trait, and the
  eight tests.
- `citest` on PyPI: PyO3 bindings, `import citest`. Ships `py.typed`.
- `citest` on npm: wasm-bindgen bindings for Node and browsers.
- `citest` on CRAN: extendr bindings, `library(citest)`, including an
  `as_pcalg()` adapter for constraint-based discovery with **pcalg** and a
  base-R `htest` adapter.

### Tests

| Test | Data | Configuration | Independent when |
|---|---|---|---|
| `chi_squared` | discrete | `yates` (default true) | `p >= alpha` |
| `log_likelihood` (G-test) | discrete | `yates` | `p >= alpha` |
| `cressie_read` | discrete | `yates` | `p >= alpha` |
| `freeman_tukey` | discrete | `yates` | `p >= alpha` |
| `modified_likelihood` | discrete | `yates` | `p >= alpha` |
| `pearson_correlation` | continuous | — | `p >= alpha` |
| `fisher_z` | continuous | — | `p >= alpha` |
| `pearson_equivalence` | continuous | `delta_threshold` (default 0.1) | `p < alpha` (TOST) |

### Fixed

- The R golden-parity check used `expect_equal(tolerance = )`, which routes
  under testthat 3e to a *relative* comparison. At the fixture's largest
  statistic this made the R gate roughly 26x looser than the absolute `1e-7`
  the other three harnesses enforce. The R results were always correct; the
  check was not.

### Notes

- Missing data is rejected when a `Dataset` is built rather than silently
  dropped or imputed. Choose a convention and apply it first.
- The R source package bundles its Rust dependencies so it builds without
  network access, per CRAN's Rust policy. `inst/AUTHORS` records the
  authorship and licence of every bundled crate.

[Unreleased]: https://github.com/pgmpy/ci-tests/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pgmpy/ci-tests/releases/tag/v0.1.0
