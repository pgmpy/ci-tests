# Task 3 Report: Remove redundant CI setup and build steps

## Summary

Removed the three unused OpenBLAS installation steps from documentation jobs
and the redundant pre-test `cargo build -p ci_core` step from the Rust test
job. Added the structural release-configuration regression test specified in
the task brief. Workflow triggers, permissions, concurrency, job IDs/names,
matrices, toolchain actions, caches, and test commands remain unchanged.

## Files

- `.github/workflows/docs.yml`
- `.github/workflows/rust.yml`
- `tests/test_release_configuration.py`
- `.superpowers/sdd/2026-08-13-first-wave-simplification/task-3-report.md`

## RED

Added `test_ci_omits_unused_openblas_and_redundant_core_build` before editing
the workflows. Ran:

```text
pytest tests/test_release_configuration.py::test_ci_omits_unused_openblas_and_redundant_core_build -q
```

Result: failed as expected at the stale `libopenblas-dev` assertion. The
first assertion stops the test before the stale Cargo build assertion is
evaluated.

## GREEN

After removing only the approved workflow steps, the targeted test passed:

```text
pytest tests/test_release_configuration.py::test_ci_omits_unused_openblas_and_redundant_core_build -q
1 passed in 0.01s
```

## Verification commands and results

```text
pytest tests/test_release_configuration.py -q
12 passed in 0.15s

python -c "import pathlib, yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path('.github/workflows').glob('*.yml')]; print('workflow-yaml-ok')"
workflow-yaml-ok

cargo doc --no-deps -p ci_core
Finished with exit code 0; generated ci_core documentation.

git diff --check
Finished with exit code 0.
```

## Concerns

`cargo doc --no-deps -p ci_core` emitted five existing rustdoc warnings for
private or unresolved intra-documentation links. No warning was introduced by
the Task 3 changes; the docs source was not modified.
