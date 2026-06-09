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

Each binding compiles the Rust core from source, so you need a Rust toolchain installed
(via [rustup](https://rustup.rs)). In every language a test takes the observation vectors
`x` and `y` and a conditioning matrix `z` whose columns are the conditioning variables;
pass a zero-column matrix for an unconditional test.

### Python

Build and install the package from the repository root:

```bash
pip install maturin
maturin develop -m crates/ci-python/Cargo.toml
```

```python
import numpy as np
from ci_python import ChiSquared

x = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
y = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

test = ChiSquared(boolean=False, significance_level=0.05)

# Unconditional: pass an (n, 0) matrix for an empty conditioning set
p_value, statistic, dof = test.run_test(x, y, np.empty((len(x), 0)))
print(f"p={p_value:.4f}, chi2={statistic:.4f}, df={dof}")

# Conditional: each column of z is one conditioning variable
z = np.array([[1.0]] * 4 + [[2.0]] * 4)
p_value, statistic, dof = test.run_test(x, y, z)
```

Continuous tests (`PearsonCorrelation`, `PearsonEquivalence`) share the same interface but
return `(p_value, coefficient)`. In boolean mode (`boolean=True`) every test returns just a
`bool` independence verdict.

### R

Install the package from the repository root:

```r
# install.packages("devtools")
devtools::install("crates/cir")
```

```r
library(cir)

x <- c(1, 1, 2, 2, 1, 1, 2, 2)
y <- c(1, 2, 1, 2, 1, 2, 1, 2)

# Unconditional: a 0-column matrix is an empty conditioning set
z <- matrix(nrow = length(x), ncol = 0)
result <- chi_squared_test(x, y, z, boolean = FALSE, significance_level = 0.05)
cat("p =", result$p_value, " chi2 =", result$statistic, " df =", result$df, "\n")

# Conditional: each column of z is one conditioning variable
z <- matrix(c(1, 1, 1, 1, 2, 2, 2, 2), ncol = 1)
result <- chi_squared_test(x, y, z, boolean = FALSE, significance_level = 0.05)

# Boolean mode returns only an independence verdict
verdict <- pearson_correlation_test(x, y, z, boolean = TRUE, significance_level = 0.05)
cat("independent:", verdict$independent, "\n")
```

Each function returns a named list with a `kind` field: `"statistic"` (with `p_value`,
`statistic`, `df`), `"pvalue"` (with `p_value`, `coefficient`), or `"boolean"` (with
`independent`).

### JavaScript

Build the WebAssembly package with [wasm-pack](https://rustwasm.github.io/wasm-pack/); the
output is written to `crates/ci-js/pkg`:

```bash
wasm-pack build crates/ci-js --target web      # for browsers
wasm-pack build crates/ci-js --target nodejs   # for Node.js
```

```js
import init, { chi_squared_test } from "./pkg/ci_js.js";

await init(); // load the WebAssembly module (web target)

const x = new Float64Array([1, 1, 2, 2, 1, 1, 2, 2]);
const y = new Float64Array([1, 2, 1, 2, 1, 2, 1, 2]);

// Unconditional: pass an empty Float64Array; z_rows = z_cols = 0
const [pValue, statistic, dof] = chi_squared_test(new Float64Array(0), 0, 0, x, y, false, 0.05);
console.log(`p=${pValue.toFixed(4)}, chi2=${statistic.toFixed(4)}, df=${dof}`);

// Conditional: z is a row-major flattened (z_rows x z_cols) matrix
const z = new Float64Array([1, 1, 1, 1, 2, 2, 2, 2]);
const [pCond] = chi_squared_test(z, 8, 1, x, y, false, 0.05);
```

The conditioning matrix is passed flattened (`z_flat`, `z_rows`, `z_cols`) because
WebAssembly has no native 2-D array type. Discrete tests return `[p_value, statistic, dof]`,
continuous tests return `[p_value, coefficient]`, and boolean mode returns a `bool`.

## Available Tests

| Test | Data type | Numeric output |
|---|---|---|
| `chi_squared` | Discrete | `(p_value, statistic, dof)` |
| `log_likelihood` (G-test) | Discrete | `(p_value, statistic, dof)` |
| `cressie_read` | Discrete | `(p_value, statistic, dof)` |
| `freeman_tukey` | Discrete | `(p_value, statistic, dof)` |
| `modified_likelihood` | Discrete | `(p_value, statistic, dof)` |
| `pearson_correlation` | Continuous | `(p_value, coefficient)` |
| `pearson_equivalence` | Continuous | `(p_value, coefficient)` |

- **Conditioning.** Every test accepts a conditioning matrix `Z` (each column is one
  conditioning variable). For conditional discrete tests the statistic is summed over the
  strata defined by `Z`; for continuous tests the partial correlation is taken on the
  regression residuals.
- **Discrete family.** The discrete tests are members of the power-divergence family and
  differ only in the $\lambda$ parameter: `chi_squared` ($1$), `log_likelihood` ($0$),
  `cressie_read` ($2/3$), `freeman_tukey` ($-1/2$), `modified_likelihood` ($-1$).
- **Boolean mode.** With `boolean=true`, a test returns a single independence verdict at the
  given `significance_level` instead of the numeric tuple.
- **Equivalence test.** `pearson_equivalence` is an equivalence (TOST) test: it additionally
  takes a `delta_threshold` and declares independence when the partial correlation lies
  within that margin of zero.
- **Naming.** Python exposes these as classes (`ChiSquared`, `LogLikelihood`, …); R and
  JavaScript expose them as functions with a `_test` suffix (`chi_squared_test`, …).

## Package Structure & Contributing

```
ci-core     Rust core: all test implementations and the CITest trait
ci-python   Python bindings (PyO3)        -> import ci_python
cir         R package (extendr)           -> library(cir)
ci-js       JavaScript / WASM (wasm-pack)
```

All bindings are thin wrappers that depend only on `ci-core`, so the statistics live in a
single place. The Rust core can also be used directly as a crate. Full API documentation is
published at <https://giphouse.github.io/Conditional-Independence-Testing/>.

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup,
coding standards, and how to add a new test. Before opening a PR:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

Licensed under the [MIT License](LICENSE).
