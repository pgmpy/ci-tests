# Conditional Independence Testing

[![CI](https://img.shields.io/github/actions/workflow/status/pgmpy/ci-tests/rust.yml?branch=main&logo=github&label=CI)](https://github.com/pgmpy/ci-tests/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pgmpy/ci-tests/blob/main/LICENSE)

A fast, multi-language library for **conditional independence (CI) testing**: deciding
whether two variables $X$ and $Y$ are independent given a (possibly empty) set of
conditioning variables $Z$ — written $X \perp Y \mid Z$. CI tests are a fundamental building
block of constraint-based causal discovery and of structure learning for probabilistic
graphical models.

Every test is implemented once in a dependency-light **Rust core** (`citest`) and exposed
through thin, idiomatic bindings for **Python, R, and JavaScript/WebAssembly**. 

## Quick Start

The API is **data-bound**: you bind a dataset once (its discrete columns are factorized a
single time), then query any `X ⟂ Y | Z` against it. Each test exposes the same surface —
`run_test(X, Y, Z)` returns a uniform result (`statistic`, `p_value`, `dof`, `effect_size`),
and `is_independent(X, Y, Z, significance_level)` applies that test's own decision rule.
Per-test configuration (e.g. `yates`, `delta_threshold`) lives in the constructor.

Install from your language's registry:

```bash
pip install citest                 # Python
npm install citest                 # JavaScript / WebAssembly
cargo add citest                   # Rust
```
```r
install.packages("citest")         # R
```

Released wheels, npm tarballs and CRAN binaries need no Rust toolchain. Building
from a source checkout does — install one via [rustup](https://rustup.rs).

### Python

From a source checkout:

```bash
pip install maturin
maturin develop -m crates/citest-python/Cargo.toml
```

```python
import numpy as np, pandas as pd
from citest import Dataset, ChiSquared, FisherZ, PearsonEquivalence

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "A": rng.integers(0, 2, 500), "B": rng.integers(0, 3, 500), "C": rng.integers(0, 2, 500),  # int -> discrete
    "X": rng.standard_normal(500), "Y": rng.standard_normal(500), "Z": rng.standard_normal(500),  # float -> continuous
})

# Bind once; column kinds are inferred from dtypes.
data = Dataset.from_pandas(df)
# You can also pass the DataFrame straight to a test constructor — `ChiSquared(df)` binds it
# implicitly (building a fresh Dataset; share one explicitly to factorize once).

# Discrete chi-squared (Yates' correction on by default).
chi = ChiSquared(data)
res = chi.run_test("A", "C", ["B"])                 # A ⟂ C | B
print(res.statistic, res.p_value, res.dof, res.effect_size)
chi.is_independent("A", "C", ["B"], significance_level=0.05)   # -> bool (independent ⇔ p ≥ α)

# Continuous Pearson equivalence (TOST); per-test config is constructor-only.
eqv = PearsonEquivalence(data, delta_threshold=0.1)
res = eqv.run_test("X", "Y", ["Z"])
print(res.statistic, res.p_value, res.effect_size)  # Pearson equivalence has no degrees of freedom
eqv.is_independent("X", "Y", ["Z"], significance_level=0.05)   # -> bool (independent ⇔ p < α)
```

`x` / `y` accept a column name or an integer index, and `z` is a sequence of names/indices
(default: empty conditioning set). You can also build a `Dataset` directly from a
`{name: (kind, values)}` mapping, or pass that mapping straight to a test constructor; one
`Dataset` can be shared across several tests so the factorization happens once.

### R

From a source checkout (the R package is named `citest`):

```r
# install.packages("devtools")
devtools::install("crates/citest-r")
```

A `fisher_z(data)` factory accompanies the factories below, mirroring the Python/JS `FisherZ`.

```r
library(citest)

set.seed(0)
df <- data.frame(
  A = sample(0:1, 500, TRUE), B = sample(0:2, 500, TRUE), C = sample(0:1, 500, TRUE),  # integer -> discrete
  X = rnorm(500), Y = rnorm(500), Z = rnorm(500)                                        # numeric -> continuous
)

# Bind once; column kinds are inferred from column classes.
data <- dataset(df)

# Discrete chi-squared (Yates' correction on by default).
chi <- chi_squared(data)
res <- run_test(chi, "A", "C", c("B"))              # A ⟂ C | B
c(res$statistic, res$p_value, res$dof, res$effect_size)
is_independent(chi, "A", "C", c("B"), significance_level = 0.05)   # -> logical (p >= alpha)

# Continuous Pearson equivalence (TOST); config is constructor-only.
eqv <- pearson_equivalence(df, delta_threshold = 0.1)
res <- run_test(eqv, "X", "Y", c("Z"))
c(res$statistic, res$p_value)                       # Pearson equivalence has no degrees of freedom
is_independent(eqv, "X", "Y", c("Z"), significance_level = 0.05)   # -> logical (p < alpha)

# Any test adapts to pcalg's indepTest(x, y, S, suffStat) interface:
indepTest <- as_pcalg(chi)
```

`run_test` / `is_independent` return / consume column **names**; `z` is a character vector of
conditioning names. A factory accepts either a pre-built `dataset()` or a raw `data.frame`.

### JavaScript

Build the WebAssembly package with [wasm-pack](https://rustwasm.github.io/wasm-pack/); the
output is written to `crates/citest-js/pkg`:

```bash
wasm-pack build crates/citest-js --target web      # for browsers
wasm-pack build crates/citest-js --target nodejs   # for Node.js
```

There are no dtypes in JS, so each column carries its `kind`. You can pass columns directly
to a constructor — `new ChiSquared(cols)` (implicit; copies once per test object) — or build
a shared `Dataset` first — `new ChiSquared(data)` — so the arrays cross the JS↔wasm boundary
only once and can be reused across multiple tests:

```js
import init, { Dataset, ChiSquared, FisherZ, PearsonEquivalence } from "./pkg/citest_js.js";

await init(); // load the WebAssembly module (web target)

const data = new Dataset({
  A: { kind: "discrete",   values: [/* ... */] },  B: { kind: "discrete",   values: [/* ... */] },  C: { kind: "discrete",   values: [/* ... */] },
  X: { kind: "continuous", values: [/* ... */] },  Y: { kind: "continuous", values: [/* ... */] },  Z: { kind: "continuous", values: [/* ... */] },
});

// Discrete chi-squared (config object optional: { yates }).
const chi = new ChiSquared(data);
const r1 = chi.runTest("A", "C", ["B"]);            // { statistic, pValue, dof, effectSize }
chi.isIndependent("A", "C", ["B"], 0.05);            // -> boolean (p >= alpha)

// Continuous Pearson equivalence (TOST); config is constructor-only.
const eqv = new PearsonEquivalence(data, { deltaThreshold: 0.1 });
const r2 = eqv.runTest("X", "Y", ["Z"]);            // r2.dof === null
eqv.isIndependent("X", "Y", ["Z"], 0.05);            // -> boolean (p < alpha)
```

`runTest` / `isIndependent` take column names (`x`, `y`) and an array of conditioning names
(`z`); the result is a plain object with `statistic`, `pValue`, `dof`, and `effectSize`
(`null` for fields a test does not define).

## Available Tests

| Test | Data type | Constructor config | Decision rule |
|---|---|---|---|
| `chi_squared` | Discrete | `yates` (default `true`) | `p ≥ α` |
| `log_likelihood` (G-test) | Discrete | `yates` (default `true`) | `p ≥ α` |
| `cressie_read` | Discrete | `yates` (default `true`) | `p ≥ α` |
| `freeman_tukey` | Discrete | `yates` (default `true`) | `p ≥ α` |
| `modified_likelihood` | Discrete | `yates` (default `true`) | `p ≥ α` |
| `pearson_correlation` | Continuous | — | `p ≥ α` |
| `fisher_z` | Continuous | — | `p ≥ α` |
| `pearson_equivalence` | Continuous | `delta_threshold` (default `0.1`) | `p < α` (TOST) |

- **Uniform result.** Every `run_test(X, Y, Z)` returns the same `CiResult`: `statistic`,
  `p_value`, `dof`, and `effect_size`. Fields a test does not define are absent
  (`None` / `NULL` / `null`). Pearson correlation reports
  `dof = n - |Z| - 2`, while Fisher-Z and Pearson equivalence report no degrees of freedom.
  The discrete tests report Cramér's V and the continuous tests the (partial) correlation as
  `effect_size`.
- **Independence decision.** `significance_level` (α) is **not** baked into the test; it is
  passed to `is_independent(X, Y, Z, significance_level)`, which applies that test's own rule
  from its metadata — the normal `p ≥ α` for most tests and the inverted `p < α` for the
  equivalence test — so the caller never writes the rule.
- **Conditioning.** Every test accepts a conditioning set `Z` (a list/vector of variables).
  For conditional discrete tests the statistic is summed over the strata defined by `Z`; for
  continuous tests the partial correlation is derived from a lazily cached covariance matrix
  (O(|Z|³) per query after the first continuous test; falls back to per-query regression for
  datasets with more than 2048 continuous columns). Pearson correlation reports
  `dof = n - |Z| - 2`; Fisher-Z and Pearson equivalence have no degrees of freedom.
- **Missing data.** NaN (Python/JS) and NA (R) are rejected when the dataset is bound — drop
  or impute first. Discrete columns accept strings everywhere (factorized to integer codes
  internally).
- **Discrete family.** The discrete tests are members of the power-divergence family and
  differ only in the $\lambda$ parameter: `chi_squared` ($1$), `log_likelihood` ($0$),
  `cressie_read` ($2/3$), `freeman_tukey` ($-1/2$), `modified_likelihood` ($-1$). They share a
  `yates` option (default `true`) applying Yates' continuity correction on 2×2 (sub-)tables,
  matching `scipy.stats.chi2_contingency` and pgmpy.
- **Equivalence test.** `pearson_equivalence` is an equivalence (TOST) test: it takes a
  `delta_threshold` (the equivalence margin on the correlation scale) and uses the **inverted**
  decision rule — it declares independence when `p < α`, i.e. when the partial correlation is
  confidently *within* that margin of zero.
- **Naming.** Python and JavaScript expose these as classes (`ChiSquared`, `LogLikelihood`,
  …); R exposes them as factory functions (`chi_squared`, `log_likelihood`, …).

## Package Structure & Contributing

```
crates/citest     Rust core: all test implementations, the CITest trait, the Dataset, and the registry
crates/citest-python   Python bindings (PyO3)        -> import citest
crates/citest-r        R package (extendr)           -> library(citest)
crates/citest-js       JavaScript / WASM (wasm-pack)
```

All three bindings are thin wrappers that depend only on `citest`, so the statistics live in a
single place; each binding maps its idiomatic input (a pandas/`data.frame`/typed columns) onto
the core `Dataset` and forwards `run_test` / `is_independent`. The Rust core can also be used
directly as a crate. A shared golden fixture (`tests/fixtures/golden.json`) is the canonical
cross-language numeric parity gate. The R source package carries mechanically synchronized
copies of that fixture and `citest` so its built archive can be checked outside the monorepo;
`python crates/citest-r/tools/sync_package_assets.py --check` rejects drift. Full API documentation
is published at <https://pgmpy.github.io/ci-tests/>.

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup,
coding standards, and how to add a new test. Each crate uses its own toolchain, so checks run
per crate rather than across the whole workspace. For a change to `citest`, before opening a PR:

```bash
cargo fmt --all -- --check
cargo clippy -p citest --all-targets -- -D warnings
cargo test -p citest          # includes the shared golden parity test
```

## License

Licensed under the [MIT License](LICENSE).
