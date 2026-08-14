# Crates

This directory contains the Rust core and language bindings for conditional
independence testing. The root Cargo workspace contains `citest`, `citest`,
and `citest`; `citest-r` is a standalone source-package workspace so that the R
package can build outside this repository.

## Packages

- [`citest/`](citest/): core Rust implementation and data-bound CI-test API.
- [`citest/`](citest/): PyO3 bindings for Python.
- [`citest/`](citest/): wasm-pack bindings for JavaScript and WebAssembly.
- [`citest-r/`](citest-r/): the standalone R source package with its synchronized core
  workspace.

For the canonical API, setup instructions, and contribution workflow, see the
[root README](../README.md) and [contribution guide](../CONTRIBUTING.md).
