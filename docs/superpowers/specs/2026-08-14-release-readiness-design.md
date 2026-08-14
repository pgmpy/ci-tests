# Release Readiness and Simplification Design

**Date:** 2026-08-14

**Status:** Stage A implemented (see git history from `1db5d30`); Stages B–D specified, not scheduled

## Context

The workspace implements eight conditional-independence tests once in a Rust
core and exposes them through PyO3, wasm-bindgen and extendr bindings, gated by
an 80-case cross-language golden fixture. The statistics are sound and all four
suites pass today. Nothing, however, has ever been published, and an audit of
the four publishable artifacts found that none of them can be published as they
stand.

Two of the four package names are already occupied on their target registries.
Two of the four published artifacts would ship a test that cannot run. Neither
the crate nor the Python distribution ships the licence it declares. Push-mode
CI is bound to branches that do not exist, so merges to the default branch run
no tests at all.

This design covers the work to make all four packages releasable, plus the
architectural simplification that the same audit identified. The work is
divided into four stages; Stage A is specified for immediate implementation and
Stages B–D are specified for later sessions.

## Verified Baseline

All four suites pass before any change, so every later failure is attributable:

| Suite | Result |
|---|---|
| `cargo test -p citest` | 60 unit + 1 golden (80 cases) + 1 doctest, all pass |
| `pytest crates/citest-python/test` | 111 pass |
| `devtools::test("crates/citest-r")` | api + golden, all pass |
| `wasm-pack build` + `npm test` | 103 pass (2 files) |

`rextendr::document("crates/citest-r")` produces no git diff against the committed
tree, confirming the checked-in generated files are in sync with their sources.

## Decisions

These were settled with the maintainer before this document was written.

1. **Scope of breakage.** Converge only where the four surfaces genuinely
   disagree. Each language keeps its idiomatic constructor and naming
   conventions.
2. **Name.** `citest` on all four registries. Directories are renamed to match.
3. **Version.** `0.1.0`, from a single source, published directly (no release
   candidate).
4. **Copyright.** `The pgmpy Developers`. GiPHouse and Ankur Ankan are authors;
   Ankur Ankan is maintainer; pgmpy is copyright holder.
5. **`tests/test_release_configuration.py`.** Deleted; its two genuine contracts
   move into `sync_package_assets.py --check`.
6. **Discrete family.** Macro-generate the five power-divergence types, leaving
   the public Rust API byte-identical.
7. **Sequencing.** Stage A implemented now; B, C and D specified here and
   scheduled separately.
8. **CRAN is a required target, but does not gate the other three.** Its
   blockers are Stage A work (A11), not later cleanup. CRAN review latency is
   outside our control, so crates.io, PyPI and npm publish at `0.1.0` as soon as
   they are ready and CRAN follows when it clears — the R package may trail by a
   patch version. Versions are asserted equal *within* a release, not across
   registries at a single instant.

## Naming

`cir` is occupied on CRAN by an unrelated package at version 2.5.1, and `citest`
is occupied on npm. `citest` was verified free on crates.io, PyPI, npm and CRAN.
It is also valid under the strictest of the four naming rules (CRAN: letters and
digits only, must begin with a letter).

| Path | Crate | Registry identity |
|---|---|---|
| `crates/citest` | `citest` | crates.io `citest` |
| `crates/citest-python` | `citest-python` (`publish = false`) | PyPI `citest`, `import citest` |
| `crates/citest-js` | `citest-js` (`publish = false`) | npm `citest` |
| `crates/citest-r` | `citest-r` (`publish = false`) | CRAN `citest`, `library(citest)` |

The R package additionally requires renaming `R/cir.R`, `src/cir-win.def`, the
S3 classes `cir_dataset` and `cir_test`, the `R_init_cir` entry point, and
regenerating `NAMESPACE`, `man/*.Rd` and `R/extendr-wrappers.R`.

## Stage A — Identity and Release Plumbing

Stage A is the blocker: no package can be published until it lands. It touches
many files but changes no statistical behaviour and no binding API.

### A1. Rename

Rename the four directories and every reference to them: workspace members,
workflow path filters, `sync_package_assets.py` constants, `.Rbuildignore`,
README, CONTRIBUTING, and the R package's internal identifiers listed above.

The rename is committed separately from every behavioural change so that review
can treat it as mechanical.

### A2. Authorship and copyright

The current state is inconsistent rather than uniformly wrong: the three Cargo
manifests credit only Ankur Ankan, the R package credits only GIP House, and
`pyproject.toml` and the JS `package.json` credit nobody. All four converge on
naming both parties.

- Root `LICENSE`: `Copyright (c) 2026 The pgmpy Developers`. It currently names
  no holder at all.
- `crates/citest-r/LICENSE`: `COPYRIGHT HOLDER: The pgmpy Developers`.
- R `Authors@R`:
  ```r
  c(person("GiPHouse", email = "giphouse@example.com", role = "aut"),
    person("Ankur", "Ankan", email = "ankurankan@gmail.com",
           role = c("aut", "cre")),
    person("The pgmpy Developers", role = "cph"))
  ```
- Cargo `authors` and `pyproject.toml` `authors`: both parties.
- README badges and the documentation URL move from
  `GiPHouse/Conditional-Independence-Testing` to `pgmpy/ci-tests`.

**Known pre-submission item.** `giphouse@example.com` uses a domain reserved by
RFC 2606 and can never receive mail. It is retained here at the maintainer's
instruction and does not block CRAN mechanically, because CRAN validates the
maintainer (`cre`) address, which is a real one. It must be replaced with a
deliverable address before CRAN submission.

### A3. One version

Add `[workspace.package] version = "0.1.0"` to the root manifest; each crate
uses `version.workspace = true`. Python takes its version from the crate through
maturin. The npm identity is generated from Cargo by wasm-pack. The R
`DESCRIPTION` carries `Version: 0.1.0` and is asserted equal by the release
workflow.

`crates/citest-js/package.json` is a development harness, not the published
package: it becomes `"private": true` and drops the conflicting `1.0.0` version,
its `license`, and its `description`.

### A4. Licence in every artifact

Add a `LICENSE` file to the core crate and the Python distribution, and declare
it so that it ships. The npm tarball and the R package already carry one.

### A5. Published artifacts ship no tests

The published artifacts currently ship test files without the fixture those
tests need. The fix is not to relocate the fixture so the shipped tests can run
— it is to stop shipping tests. Consumers install a library to use it, not to
run its development suite, and the golden fixture is a development-time
cross-language parity gate rather than something a downstream caller needs.

Measured contents of the artifacts as they build today:

| Artifact | Ships now | Ship tests? |
|---|---|---|
| crates.io | `tests/golden.rs`, without its fixture | No |
| PyPI sdist | Rust *and* Python tests, no fixture, plus `crates/citest-python/.vscode/settings.json` | No |
| PyPI wheel | no tests, but `citest/__pycache__/__init__.cpython-314.pyc` | No |
| npm | `ci_js_bg.wasm`, `citest_js.js`, `citest_js.d.ts` | No — already correct |
| CRAN | `tests/testthat/**` plus its own fixture copy | **Yes — by design** |

Actions:

- Rust: add an explicit `include` to the crate manifest covering `src/**`,
  `README.md`, `LICENSE` and `Cargo.toml`, so `tests/` is not published.
- Python: exclude `test/`, `.vscode/` and `__pycache__/` from both sdist and
  wheel. Shipping a Python 3.14 bytecode cache inside an abi3 wheel that
  advertises 3.10+ is actively wrong, not merely untidy.
- npm: no change.
- **R: keep shipping tests.** CRAN runs a package's own test suite on its check
  farm across platforms, and that is a substantial part of the QA value of being
  on CRAN. This is why `crates/citest-r/tests/testthat/fixtures/golden.json`
  exists as a vendored copy, and why it must remain in the built archive.

**Consequence for the fixture layout.** The canonical fixture stays where it is,
at `tests/fixtures/golden.json`. No relocation is needed, and the centre of
gravity of the parity system does not move. The R copy is load-bearing rather
than redundant duplication, and `sync_package_assets.py --check` remains the
mechanism that keeps the two byte-identical.

### A6. CI that runs

- `branches: [master, development]` becomes `[main]` in `rust.yml`,
  `python.yml`, `js.yml` and `r.yml`. The default branch is `main`, so push CI
  currently never runs; only pull-request triggers fire.
- `r.yml` invokes `sync_package_assets.py` without setting up Python. Add the
  Python setup step.

### A7. Delete the release-configuration test

`tests/test_release_configuration.py` (289 lines) asserts exact CI step names
and `with:` blocks, exact transitive lockfile pins, README prose, and the
placeholder copyright holder it is meant to prevent. It breaks on any legitimate
change and cannot serve as a release gate.

It is deleted. Its two genuine contracts move into
`sync_package_assets.py --check`, which `r.yml` already runs:

- the vendor archive matches the locked registry crate set and every vendored
  crate declares a licence;
- `THIRD-PARTY-NOTICES` covers the redistributed extendr crates.

### A8. Release automation

- `CHANGELOG.md` in Keep a Changelog format with a `0.1.0` entry.
- `.github/workflows/release.yml`, triggered on `v*` tags, which first asserts
  that the crate, Python, npm and R versions agree and match the tag, then
  publishes to crates.io, PyPI and npm. CRAN submission remains manual, as it
  always is.

### A9. Distribution metadata

- `pyproject.toml` gains `description`, `readme`, `license`, `authors`,
  `urls`, `keywords` and the missing classifiers; `py.typed` is added so the
  hand-written stub is visible to type checkers.
- The npm package must ship wasm and JavaScript. Publishing from
  `crates/citest-js` today would ship Rust source with no entry point; the
  published artifact is the wasm-pack output in `pkg/`, and the release workflow
  publishes from there.

### A10. Repair the weakened R parity gate

The four golden harnesses are advertised as one gate at an absolute `1e-7`, and
three of them are:

| Harness | Comparison |
|---|---|
| `crates/citest/tests/golden.rs:191` | `(act - exp).abs() < TOL` |
| `crates/citest-python/test/test_golden.py:87` | `pytest.approx(exp, abs=TOL, rel=0.0)` |
| `crates/citest-js/tests/golden.test.js:122` | `Math.abs(actual - exp) <= TOL` |
| `crates/citest-r/tests/testthat/test-golden.R:91` | `expect_equal(act, exp, tolerance = tol)` |

R's bare `tolerance =` routes, under testthat 3rd edition, to waldo's
`all.equal`-style **relative** difference whenever `mean(abs(expected))` exceeds
the tolerance. The fixture's largest `|statistic|` is `26.1558538058`, so R's
effective bound there is `2.6e-6` — **26.2× looser** than the other three. It is
never stricter, so this is a silently weakened gate rather than a flaky one.

This is promoted from Stage D into Stage A: it is a one-line fix, and shipping
packages as "release ready" while one of the four parity gates is knowingly
weaker than advertised is not defensible. Replace with an explicit absolute
comparison matching the other three.

### A11. CRAN submission readiness

CRAN is a required target for 0.1.0, so the R-specific blockers are Stage A
rather than later cleanup.

**Why vendoring stays.** CRAN's Rust policy permits only two ways to supply
crates: bundle them via `cargo vendor`, or download a pinned version *from a
site under the maintainer's control* with checksum verification — explicitly not
crates.io, and only as an escape hatch for oversized bundles. CRAN build
machines are offline. The existing `vendor.tar.xz` approach is correct and
matches real Rust CRAN packages: `gifski` and `arcgisutils` declare the same
`SystemRequirements: Cargo (Rust's package manager), rustc …, xz`.

**A11.1 — Attribution for redistributed crates (mandatory).** Policy: *"the
authorship and copyright information for the Rust code must be included in the
`DESCRIPTION` file. That includes any Rust sources included as dependencies."*
Today the notice names 3 of 25 crates.

Adopt the `gifski` pattern, which CRAN has accepted:

- `Authors@R` gains
  `person("Authors of the dependency Rust crates", role = "aut", comment = "see AUTHORS file")`.
- `THIRD-PARTY-NOTICES` becomes `inst/AUTHORS`. A bare top-level
  `THIRD-PARTY-NOTICES` is a non-standard file that draws an `R CMD check`
  warning; `inst/AUTHORS` is the conventional location.
- `inst/AUTHORS` lists every redistributed crate with version and licence.

**A11.2 — Two licence facts the current metadata gets wrong.** Verified by
reading each vendored crate's manifest:

- `approx 0.5.1` is **Apache-2.0 only**, not dual-licensed. The bundle is
  therefore not purely MIT-compatible-by-default, and this must be stated.
- `unicode-ident 1.0.24` is `(MIT OR Apache-2.0) AND Unicode-3.0` — the `AND`
  means the Unicode licence applies in addition, and needs its own entry.

`License: MIT + file LICENSE` remains correct for the package's own code
provided `inst/AUTHORS` documents the bundle; that is exactly what `gifski`
does while vendoring non-MIT crates.

**A11.3 — Prune never-compiled crates from the bundle.** `citest` declares
`serde` and `serde_json` as `[dev-dependencies]` for `tests/golden.rs`. The
vendored copy contains no tests — sync copies only `Cargo.toml`, `README.md` and
`src/**` — so seven crates (`serde`, `serde_core`, `serde_derive`,
`serde_json`, `memchr`, `itoa`, `zmij`, 2.68 MB uncompressed, 25% of the
archive) are shipped, licence-documented and never compiled.

Have `sync_package_assets.py` strip `[dev-dependencies]` when copying the
manifest, then re-lock and re-vendor. Result: 25 → 18 crates, and seven fewer
to attribute under A11.1.

Note that both `syn` versions are genuinely required and must stay:
`extendr-macros` and `readonly` need `syn 2.0.119`; `thiserror-impl` needs
`syn 3.0.3`. That duplication is 4.5 MB and is not removable.

**A11.4 — Installed size.** The audit measured 6.2 MB installed against CRAN's
5 MB threshold. This is *not* caused by the vendored crate set —
`vendor.tar.xz` is 1.02 MB, lives only in the source tarball, and `cleanup`
deletes it after build. It is caused by the compiled `cir.so`: the R sub-workspace
is a separate Cargo workspace and declares no `[profile.release]`, so it silently
loses the repository's `lto = true, codegen-units = 1`. Add them there.

**A11.5 — Check what CRAN checks.** R CI never runs `--as-cran`, so none of the
above has ever failed a build. Add it, and fix what it reports. Known items:
`DESCRIPTION` lacks `URL` and `BugReports`; its `Description` omits one of the
eight tests; `man/` documents the internal extendr handles (`Dataset.Rd`) and
the private `.cir_*` helpers (`dot-cir_*.Rd`), which should not be user-facing;
the golden suite hard-requires `jsonlite` from `Suggests`; `tools/` ships a
maintenance script to users.

### A12. Documentation truth-up

- CONTRIBUTING's instruction to run JS tests "from `crates/citest-js/tests`"
  describes a directory removed by an earlier consolidation.
- CONTRIBUTING references `docs/api-examples.md`, which does not exist.
- No README documents installing from a registry; every path is
  build-from-source. Add registry install instructions.

### Stage A acceptance

All four suites pass in the repository, with the R suite run through the
`expert` conda environment.

Artifact contents are asserted, not assumed — each is built and its file list
inspected:

- `cargo package --list -p citest` contains `src/**`, `README.md`, `LICENSE`
  and the manifests, and **no `tests/`**.
- The Python wheel contains `citest/__init__.py`, the `.so`, the `.pyi`,
  `py.typed` and `LICENSE`, and **no `test/`, `.vscode/` or `__pycache__/`**.
  Its `METADATA` carries `Summary`, `Description`, `License`, `Author` and
  `Project-URL`, none of which it has today.
- The Python sdist contains no test files and no editor configuration.
- The wasm-pack output is importable under the new name.
- The built R source archive **does** contain `tests/testthat/**` and its
  fixture copy, and `R CMD check` runs them.
- `R CMD check --as-cran` on the built archive reports no ERROR or WARNING,
  installed size is under CRAN's 5 MB threshold, and `inst/AUTHORS` names every
  crate the vendor archive redistributes.

`sync_package_assets.py --check` passes and covers the vendor-archive and
third-party-notice contracts migrated out of the deleted release test.

## Stage B — Core Simplification

Confined to the core crate; no binding changes.

- Macro-generate the five power-divergence types. `chi_squared.rs`,
  `cressie_read.rs`, `freeman_tukey.rs`, `log_likelihood.rs` and
  `modified_likelihood.rs` are byte-identical apart from a name string and a
  `LAMBDA` constant, at roughly 52 lines each. The generated public API is
  unchanged: the types keep their names, their `yates` field, their derives and
  their per-type documentation.

  **Recorded dissent.** The audit's synthesis argued against this, on the
  grounds that the five files are public API surface carrying doc comments and
  that hiding them inside `macro_rules!` degrades rustdoc and greppability to
  save ~200 lines in a ~3,554-line crate. The maintainer chose to proceed. The
  objection is answered by construction rather than overruled: the macro takes
  a per-invocation `$doc:literal`, so each type keeps its own prose — including
  the details that genuinely differ, such as `ModifiedLikelihood`'s `+∞` at a
  structural zero and `CressieRead`'s "recommended compromise" framing — and
  rustdoc output is unchanged. Greppability is preserved because each
  invocation names its type on one line. If review of the generated rustdoc
  shows otherwise, revert to five files; the decision is cheap to undo.

- **Decide the fate of the Householder-QR residual path.** `partial_correlation`
  (`pearson_correlation.rs:252`) dispatches on `data.gram()`, and
  `GramCache::build` returns `None` only when `n == 0 || p == 0 || p > 2048`
  (`gram.rs:52`). At `p == 0` the residual path errors immediately on
  `data.continuous(x)?`, and at `n == 0` on `check_row_count`. The only input
  for which it returns `Ok` is a dataset with **more than 2048 continuous
  columns**; the widest golden case has 4. That is a second, numerically
  distinct implementation of the library's headline computation with zero
  end-to-end coverage — roughly 210 production and 56 test lines.

  Either delete it and turn the column cap into an explicit documented error, or
  keep it and add an integration test that actually reaches it through
  `CITest::test`. Deleting is recommended: an unvalidated silent algorithm
  switch is a liability, not a safety net. This needs a maintainer decision
  before Stage B begins.
- Add a continuous counterpart to `discrete_common.rs`. `FisherZ` and
  `PearsonEquivalence` duplicate the whole Fisher-z preamble: clip `rho`,
  compute `sqrt(n - |Z| - 3)`, construct the standard normal.
- Tighten visibility. `pub mod discrete` exports no public items. Every
  `ci_tests` submodule is public *and* re-exported, giving each test two public
  paths.
- Validate configuration at construction rather than query time.
  `PearsonEquivalence { delta_threshold: 0.0 }` constructs successfully and then
  fails on every query with `CiError::DegenerateData` — the wrong variant, since
  the fault is configuration, not data.
- Remove unreachable error paths and correct documentation that contradicts the
  code, including the `Dataset::discrete` doc block attached to the private
  `Dataset::column`, and `gram.rs` defending against non-finite inputs that
  `Dataset::from_columns` already rejects.

## Stage C — Binding Convergence

Fix only genuine disagreements; keep idiomatic shapes.

- **Python error model.** Column-reference failures raise `ValueError` while
  query and kind failures raise `CiError`, so `except CiError` silently misses
  half the failure modes. Make `CiError` inherit from `ValueError` and raise it
  uniformly; both spellings then catch everything, and no existing code breaks.
  `CiError.__module__` is also wrong, which breaks pickling.
- **No error carries the core's error kind.** `CiError` is seven variants in
  Rust, but every binding flattens it to a message string. The consequence is
  visible in this repository's own suites, which match on message prose rather
  than on a kind (`test-api.R:143,198`; `test_api.py:139,216`) — the same thing
  a downstream consumer is forced to do. Add a machine-readable discriminant to
  `CiError` and surface it in all three bindings.
- **Stub drift.** The hand-written `.pyi` has already drifted from
  `src/lib.rs` and nothing in CI checks it. Add a consistency gate.
- **R has no `meta()`.** Python and JS expose it; R does not, and therefore
  re-hardcodes the inverted decision rule as the literal
  `.cir_inverted_rule_tests <- c("pearson_equivalence")`. A ninth `PValueLt`
  test would silently invert every pcalg edge decision. Expose `meta()` in R and
  derive the rule from it.
- **Registry is unreachable.** `registry::{all_metas, make_default}` documents
  itself as the single source of truth but has no consumer outside its own unit
  tests; all four bindings re-enumerate the eight tests independently. Expose
  test enumeration in every binding rather than removing the module: the R
  `meta()` work above needs the same machinery, and enumeration is a genuine
  affordance for callers who want to discover the available tests by name.
- **`z` handling diverges.** `z` is optional in Python and R but mandatory in
  JS, where omitting it throws an unactionable glue `TypeError`. A bare-string
  `z` is rejected in Python, treated as one conditioning column in R, and split
  into characters in JS.
- **`index_of` base differs.** 1-based in R, 0-based in Python and JS. Do *not*
  converge this: 1-based indexing is correct R idiom and 0-based is correct for
  Python and JS, so this is correct localization rather than divergence. Document
  the difference explicitly in each binding's reference instead.
- **JS types are `any` at every result boundary.** Serialize `CiResult` and
  `TestMeta` with serde instead of the hand-written `Reflect` boilerplate, and
  emit real TypeScript types. Add an `exports` map and an ESM entry.

## Stage D — Test and CI Architecture

- **The fixture is tracked twice**, at roughly 287 KB per copy for 80 cases,
  and `generate_golden.py` writes only one of the two. The second copy is
  required — the R package must carry it so CRAN can run the suite (see A5) —
  so the fix is not to remove it but to stop regeneration being a multi-command
  ritual: have the generator write both destinations in one pass. Today,
  skipping the sync step leaves Python CI green and R CI red, in a different
  workflow.
- **The pgmpy parity check never runs in CI**, although detecting drift from
  pgmpy is the reason it exists.
- **The Python job rebuilds the extension nine times** for what is a single
  abi3 wheel.
- **The fixture-currency check runs twice** in `python.yml`, once as a pytest
  and once as a CLI invocation.
- **`extendr_module!` duplicates the macro invocation list.** Omitting an entry
  compiles cleanly and silently drops the class from R. Emit both the handle and
  its registration from one table.
- **`docs/superpowers/` is 3,771 lines, about 79% of all markdown in the
  repository**, and consists of executed plans describing repo states that no
  longer exist. Prune to the specs that still describe current intent.

## Risks and Constraints

- **R is verifiable locally but only through conda.** R 4.4.2 lives in the
  `expert` conda environment. Because it is a conda R rather than a system R,
  the embedded Rust fails to link `libR` unless `LD_LIBRARY_PATH` includes
  `R.home("lib")`. CI is unaffected: it uses a standard CRAN R.
- **The rename produces a large mechanical diff.** It is isolated in its own
  commit, ahead of every behavioural change.
- **CRAN gates the 0.1.0 release.** CRAN is a required target, so A11 must land
  before any registry is published — a single version ships everywhere. CRAN
  review latency is outside our control and is the schedule risk for the whole
  release; the other three registries are technically ready much earlier.
- **Re-vendoring rewrites binary and lock artifacts.** A11.3 regenerates
  `vendor.tar.xz` (~1 MB binary) and `crates/citest-r/src/rust/Cargo.lock`, and
  needs network access to populate the crate cache. Keep it in its own commit.
- **`rextendr::document()` is deprecated** in favour of `devtools::document()`;
  `r.yml` still calls the deprecated form.

## Out of Scope

No statistical behaviour changes. No golden fixture value changes. The R source
package keeps its vendored copy of the core crate and its vendored crate
archive, both of
which are required for the built archive to compile outside the monorepo without
network access.
