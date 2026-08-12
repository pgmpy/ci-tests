# ci-core

Core Rust library implementing conditional independence tests.

## Architecture

This crate uses the Strategy pattern:

- **Strategy**: Each CI test implements the `CITest` trait
- **Registry**: The `TestRegistry` maintains a map of test name → implementation
- **Tests**: Individual test implementations in `src/tests/`

## Adding a New Test

1. Create a new file in `src/tests/` (e.g., `student_t.rs`)
2. Implement the `CITest` trait
3. Register it in `src/registry.rs`
4. Add tests in `tests/integration/`

See `CONTRIBUTING.md` in the repository root for detailed guidelines.
