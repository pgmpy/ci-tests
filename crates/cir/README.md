# cir

R bindings for the [Conditional Independence Testing](../../README.md) library. Wraps the Rust core via [extendr](https://extendr.github.io) and accepts standard R vectors and matrices directly.

## Available tests

| Function | Data type | Numeric output fields |
|---|---|---|
| `chi_squared_test` | Discrete | `$statistic`, `$p_value`, `$df` |
| `cressie_read_test` | Discrete | `$statistic`, `$p_value`, `$df` |
| `freeman_tukey_test` | Discrete | `$statistic`, `$p_value`, `$df` |
| `log_likelihood_test` | Discrete | `$statistic`, `$p_value`, `$df` |
| `modified_likelihood_test` | Discrete | `$statistic`, `$p_value`, `$df` |
| `pearson_correlation_test` | Continuous | `$p_value`, `$coefficient` |
| `pearson_equivalence_test` | Continuous | `$p_value`, `$coefficient` |

All tests return a named list. Every result also has a `$kind` field: `"statistic"`, `"pvalue"`, or `"boolean"` depending on the mode and test type.

All tests support an optional conditioning matrix `z`. Pass a 0-column matrix for unconditional tests.

## Requirements

- R >= 4.2
- Rust (stable), installed via [rustup](https://rustup.rs)
- `devtools` and `rextendr` R packages

## Installation

Install the required R packages, then load the package from the repository root:

```r
install.packages("pak", repos = "https://cloud.r-project.org")
pak::pak(c("devtools", "rextendr"))
```

```r
setwd("crates/cir")
devtools::load_all()  # compiles Rust and loads the package
```

## Usage

### Numeric mode

Numeric mode returns the raw test statistic and p-value as a named list. Pass `boolean = FALSE`:

```r
library(cir)

x <- c(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
y <- c(1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
z <- matrix(0, nrow = length(x), ncol = 0)  # unconditional

result <- chi_squared_test(x, y, z, FALSE, 0.05)
cat("p =", result$p_value, " statistic =", result$statistic, " df =", result$df, "\n")
```

For continuous data, the result contains `$p_value` and `$coefficient` instead:

```r
result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
cat("p =", result$p_value, " r =", result$coefficient, "\n")
```

### Boolean mode

Boolean mode returns a list with a single `$independent` field: `TRUE` if the null hypothesis of independence is not rejected, `FALSE` if it is rejected. Pass `boolean = TRUE`:

```r
result <- chi_squared_test(x, y, z, TRUE, 0.05)
cat("independent:", result$independent, "\n")
```

### Conditional tests

Pass a conditioning matrix `z` where each column is one conditioning variable:

```r
x <- c(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
y <- c(1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
z <- matrix(c(0, 0, 0, 0, 1, 1, 1, 1), nrow = 8, ncol = 1)

result <- chi_squared_test(x, y, z, FALSE, 0.05)
```

To condition on multiple variables, bind them as columns with `cbind`:

```r
z <- cbind(z1, z2, z3)  # matrix with 3 conditioning columns
result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
```

### Pearson equivalence test

`pearson_equivalence_test` takes one extra argument, `delta_threshold`, which sets the equivalence margin for the TOST procedure. Independence is declared when the partial correlation falls within `[-delta_threshold, delta_threshold]`:

```r
result <- pearson_equivalence_test(x, y, z, FALSE, 0.05, 0.1)
cat("p =", result$p_value, " r =", result$coefficient, "\n")
```

## Running tests

```r
setwd("crates/cir")
devtools::test()
```

## License

Licensed under the [MIT license](../../LICENSE).
