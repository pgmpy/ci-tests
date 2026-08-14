# Crates

This directory contains the Rust core and language bindings for conditional
independence testing. The root Cargo workspace contains `citest`,
`citest-python`, and `citest-js`; `citest-r` is a standalone source-package
workspace so that the R package can build outside this repository.

## Packages

- [`citest/`](citest/): core Rust implementation and data-bound CI-test API.
- [`citest-python/`](citest-python/): PyO3 bindings, published to PyPI as `citest`.
- [`citest-js/`](citest-js/): wasm-pack bindings for JavaScript and
  WebAssembly, published to npm as `citest`.
- [`citest-r/`](citest-r/): the standalone R source package (CRAN `citest`)
  with its synchronized core workspace.

For the canonical API, setup instructions, and contribution workflow, see the
[root README](../README.md) and [contribution guide](../CONTRIBUTING.md).
