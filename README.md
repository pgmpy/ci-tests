# Conditional Independence Testing

[![CI](https://img.shields.io/github/actions/workflow/status/GiPHouse/Conditional-Independence-Testing/ci.yml?branch=main&logo=github&label=CI)](https://github.com/GiPHouse/Conditional-Independence-Testing/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/GiPHouse/Conditional-Independence-Testing/blob/main/LICENSE)

A fast, multi-language library for statistical conditional independence testing. The Rust core implements several well-known tests from the power-divergence family (for discrete data) and Pearson-correlation-based tests (for continuous data), with bindings for Python, R, and JavaScript.

## Available Tests

| Test | Data type | Output |
|---|---|---|
| `chi_squared` | Discrete | `(p_value, statistic, dof)` |
| `cressie_read` | Discrete | `(p_value, statistic, dof)` |
| `freeman_tukey` | Discrete | `(p_value, statistic, dof)` |
| `log_likelihood` | Discrete | `(p_value, statistic, dof)` |
| `modified_likelihood` | Discrete | `(p_value, statistic, dof)` |
| `pearson_correlation` | Continuous | `(p_value, coefficient)` |
| `pearson_equivalence` | Continuous | `(p_value, coefficient)` |

All tests accept an optional conditioning matrix Z. Pass an empty matrix for unconditional tests. Every test can also run in **boolean mode**, returning only an independence verdict at a given significance level instead of the raw statistic.

## Crate Structure

```
ci-core      Core test implementations and CITest trait (Rust library)
ci-python    PyO3 bindings — importable as ci_python
cir          extendr bindings — importable as an R package
ci-js        wasm-pack bindings
```

Downstream crates depend only on `ci-core`. The bindings crates are thin wrappers that handle type conversion.

Documentation can be seen at: https://giphouse.github.io/Conditional-Independence-Testing/

## Getting Started

### Rust

Add `ci_core` to your `Cargo.toml`:

```toml
[dependencies]
ci_core = { git = "https://github.com/GiPHouse/Conditional-Independence-Testing" }
```

Run a test:

```rust
use ci_core::ci_tests::chi_squared::ChiSquared;
use ci_core::strategy::{CITest, TestResult};
use ndarray::{Array1, Array2};

let test = ChiSquared::new(/*boolean=*/false, /*significance_level=*/0.05);

let x = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
let z = Array2::zeros((0, 0)); // empty conditioning set

match test.run_test(x, y, z)? {
    TestResult::Statistic(p_value, statistic, dof) => {
        println!("p={p_value:.4}, χ²={statistic:.4}, df={dof}");
    }
    _ => unreachable!(),
}
```

For conditional tests, pass a matrix where each column is one conditioning variable:

```rust
use ndarray::{Array2, Axis, stack};

// Condition on two variables z1 and z2
let z = stack(Axis(1), &[z1.view(), z2.view()])?;
test.run_test(x, y, z)?;
```

### Python

Build and install the Python package from the repository root:

```bash
pip install maturin
maturin develop -m crates/ci-python/Cargo.toml
```

```python
import numpy as np
from ci_python import ChiSquared, PearsonCorrelation

test = ChiSquared(boolean=False, significance_level=0.05)

x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
z = np.empty((len(x), 0))  # unconditional

p_value, statistic, dof = test.run_test(x, y, z)
print(f"p={p_value:.4f}, χ²={statistic:.4f}, df={dof}")
```

For continuous data:

```python
test = PearsonCorrelation(boolean=False, significance_level=0.05)
p_value, coefficient = test.run_test(x, y, z)
```

### R

Install the R package from the repository root:

```r
# From within R, using devtools
devtools::install("crates/cir")
```

```r
library(cir)

x <- c(0, 1, 0, 1, 0)
y <- c(1, 0, 1, 0, 1)
z <- matrix(nrow = length(x), ncol = 0)  # unconditional

result <- chi_squared_test(x, y, z, boolean = FALSE, significance_level = 0.05)
cat("p =", result$p_value, " statistic =", result$statistic, "\n")
```

Boolean mode returns only an independence verdict:

```r
result <- pearson_correlation_test(x, y, z, boolean = TRUE, significance_level = 0.05)
cat("independent:", result$independent, "\n")
```

### JavaScript

Install the JavaScript package from the repository root:

```bash
// For directly using it on the web
wasm-pack build --target web 

// For using it in NodeJs
wasm-pack build --target nodejs
```

```js
import init, { log_likelihood_test } from "../pkg/ci_js.js";
const wasm = await import("../pkg/ci_js.js");

beforeAll(async () => {
  await wasm.default.init();
});

const x = toFloat64(1, 1, 2, 2, 1, 1, 2, 2);
const y = toFloat64(1, 2, 1, 2, 1, 2, 1, 2);
const z = new Float64Array(0);
const z_rows = 0;
const z_cols = 0;
const boolean = false;
const significance_level = 0.05;

const [p_value, statistic, dof] = log_likelihood_test(
  z, 
  z_rows,
  z_cols,
  x,
  y,
  boolean,
  significance_level,
);
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and how to add a new CI test.

Quick check before opening a PR:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

Licensed under the [MIT license](LICENSE).
