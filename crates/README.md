# Crates

This directory contains the Rust core and language bindings for conditional
independence testing. The root Cargo workspace contains `ci-core`, `ci-python`,
and `ci-js`; `ci-r` is a standalone source-package workspace so that the R
package can build outside this repository.

## Packages

- [`ci-core/`](ci-core/): core Rust implementation and data-bound CI-test API.
- [`ci-python/`](ci-python/): PyO3 bindings for Python.
- [`ci-js/`](ci-js/): wasm-pack bindings for JavaScript and WebAssembly.
- [`ci-r/`](ci-r/): the standalone R source package with its synchronized core
  workspace.

For the canonical API, setup instructions, and contribution workflow, see the
[root README](../README.md) and [contribution guide](../CONTRIBUTING.md).
