# Conditional Independence Testing

[![CI](https://img.shields.io/github/actions/workflow/status/GiPHouse/Conditional-Independence-Testing/rust.yml?branch=main&logo=github&label=CI)](https://github.com/GiPHouse/Conditional-Independence-Testing/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/GiPHouse/Conditional-Independence-Testing/blob/main/LICENSE)

A fast, multi-language library for **conditional independence (CI) testing**: deciding
whether two variables $X$ and $Y$ are independent given a (possibly empty) set of
conditioning variables $Z$ — written $X \perp Y \mid Z$. CI tests are a fundamental building
block of constraint-based causal discovery and of structure learning for probabilistic
graphical models.

Every test is implemented once in a dependency-light **Rust core** (`ci-core`) and exposed
through thin, idiomatic bindings for **Python, R, and JavaScript/WebAssembly**. 

## Quick Start

The API is **data-bound**: you bind a dataset once (its discrete columns are factorized a
single time), then query any `X ⟂ Y | Z` against it. Each test exposes the same surface —
`run_test(X, Y, Z)` returns a uniform result (`statistic`, `p_value`, `dof`, `effect_size`),
and `is_independent(X, Y, Z, significance_level)` applies that test's own decision rule.
Per-test configuration (e.g. `yates`, `delta_threshold`) lives in the constructor.

Each binding compiles the Rust core from source, so you need a Rust toolchain installed
(via [rustup](https://rustup.rs)).

### Python

Build and install the package from the repository root:

```bash
pip install maturin
maturin develop -m crates/ci-python/Cargo.toml
```

```python
import numpy as np, pandas as pd
from ci_python import Dataset, ChiSquared, PearsonEquivalence

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "A": rng.integers(0, 2, 500), "B": rng.integers(0, 3, 500), "C": rng.integers(0, 2, 500),  # int -> discrete
    "X": rng.standard_normal(500), "Y": rng.standard_normal(500), "Z": rng.standard_normal(500),  # float -> continuous
})

# Bind once; column kinds are inferred from dtypes.
data = Dataset.from_pandas(df)

# Discrete chi-squared (Yates' correction on by default).
chi = ChiSquared(data)
res = chi.run_test("A", "C", ["B"])                 # A ⟂ C | B
print(res.statistic, res.p_value, res.dof, res.effect_size)
chi.is_independent("A", "C", ["B"], significance_level=0.05)   # -> bool (independent ⇔ p ≥ α)

# Continuous Pearson equivalence (TOST); per-test config is constructor-only.
eqv = PearsonEquivalence(data, delta_threshold=0.1)
res = eqv.run_test("X", "Y", ["Z"])
print(res.statistic, res.p_value, res.effect_size)  # res.dof is None for continuous tests
eqv.is_independent("X", "Y", ["Z"], significance_level=0.05)   # -> bool (independent ⇔ p < α)
```

`x` / `y` accept a column name or an integer index, and `z` is a sequence of names/indices
(default: empty conditioning set). You can also build a `Dataset` directly from a
`{name: (kind, values)}` mapping, or pass that mapping straight to a test constructor; one
`Dataset` can be shared across several tests so the factorization happens once.

### R

Install the package from the repository root (the R package is named `cir`):

```r
# install.packages("devtools")
devtools::install("crates/ci-r")
```

```r
library(cir)

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
c(res$statistic, res$p_value)                       # res$dof is NULL for continuous tests
is_independent(eqv, "X", "Y", c("Z"), significance_level = 0.05)   # -> logical (p < alpha)

# Any test adapts to pcalg's indepTest(x, y, S, suffStat) interface:
indepTest <- as_pcalg(chi)
```

`run_test` / `is_independent` return / consume column **names**; `z` is a character vector of
conditioning names. A factory accepts either a pre-built `dataset()` or a raw `data.frame`.

### JavaScript

Build the WebAssembly package with [wasm-pack](https://rustwasm.github.io/wasm-pack/); the
output is written to `crates/ci-js/pkg`:

```bash
wasm-pack build crates/ci-js --target web      # for browsers
wasm-pack build crates/ci-js --target nodejs   # for Node.js
```

There are no dtypes in JS, so each column carries its `kind`. Build a `Dataset` once (this
copies the column arrays across the JS↔wasm boundary a single time) and share it across tests:

```js
import init, { Dataset, ChiSquared, PearsonEquivalence } from "./pkg/ci_js.js";

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
| `pearson_equivalence` | Continuous | `delta_threshold` (default `0.1`) | `p < α` (TOST) |

- **Uniform result.** Every `run_test(X, Y, Z)` returns the same `CiResult`: `statistic`,
  `p_value`, `dof`, and `effect_size`. Fields a test does not define are absent
  (`None` / `NULL` / `null`) — e.g. continuous tests report no `dof`. The discrete tests
  report Cramér's V and the continuous tests the (partial) correlation as `effect_size`.
- **Independence decision.** `significance_level` (α) is **not** baked into the test; it is
  passed to `is_independent(X, Y, Z, significance_level)`, which applies that test's own rule
  from its metadata — the normal `p ≥ α` for most tests and the inverted `p < α` for the
  equivalence test — so the caller never writes the rule.
- **Conditioning.** Every test accepts a conditioning set `Z` (a list/vector of variables).
  For conditional discrete tests the statistic is summed over the strata defined by `Z`; for
  continuous tests the partial correlation is taken on the regression residuals
  (intercept included, `dof = n − |Z| − 2`).
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
crates/ci-core     Rust core: all test implementations, the CITest trait, the Dataset, and the registry
crates/ci-python   Python bindings (PyO3)        -> import ci_python
crates/ci-r        R package (extendr)           -> library(cir)
crates/ci-js       JavaScript / WASM (wasm-pack)
```

All three bindings are thin wrappers that depend only on `ci-core`, so the statistics live in a
single place; each binding maps its idiomatic input (a pandas/`data.frame`/typed columns) onto
the core `Dataset` and forwards `run_test` / `is_independent`. The Rust core can also be used
directly as a crate. A shared golden fixture (`tests/fixtures/golden.json`) is consumed by every
language's test suite as the cross-language numeric parity gate. Full API documentation is
published at <https://giphouse.github.io/Conditional-Independence-Testing/>.

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup,
coding standards, and how to add a new test. Each crate uses its own toolchain, so checks run
per crate rather than across the whole workspace. For a change to `ci-core`, before opening a PR:

```bash
cargo fmt --all -- --check
cargo clippy -p ci_core --all-targets -- -D warnings
cargo test -p ci_core          # includes the shared golden parity test
```

## License

Licensed under the [MIT License](LICENSE).
