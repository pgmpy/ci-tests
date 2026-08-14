# citest — conditional-independence tests for R

R bindings for the `citest` Rust library: data-bound conditional-independence
(CI) tests with one uniform surface. Bind a data.frame once, then run any
number of `X ⟂ Y | Z` queries against it. See the
[repository README](../../README.md) for the cross-language story.

## Install

Requires Rust 1.81 or newer ([rustup](https://rustup.rs)). From the repository root:

```r
# install.packages("devtools")
devtools::install("crates/citest-r")
```

> Benchmarking note: `devtools::load_all()` / `rextendr::document()` compile a
> **debug** build. For benchmarks, install a release build:
> `NOT_CRAN=true R CMD INSTALL crates/citest-r` and `library(citest)`.

## Quick start

```r
library(citest)

set.seed(0)
df <- data.frame(
  A = sample(0:1, 500, TRUE), B = sample(0:2, 500, TRUE),   # integer -> discrete
  X = rnorm(500), Y = rnorm(500), Z = rnorm(500)            # double  -> continuous
)

data <- dataset(df)        # bind once; factor/character columns are coded for you

chi <- chi_squared(data)                    # Yates' correction on by default
res <- run_test(chi, "A", "B")              # list(statistic, p_value, dof, effect_size)
is_independent(chi, "A", "B", significance_level = 0.05)

fz <- fisher_z(data)                        # the pcalg/causal-learn standard test
run_test(fz, "X", "Y", c("Z"))              # res$dof is NULL for continuous tests
```

Column kinds are inferred: `integer`/`logical`/`factor`/`character` columns are
**discrete**, `double` columns **continuous**. Missing values are rejected when
the dataset is bound — `na.omit(df)` (or impute) first. Every factory also
accepts a raw data.frame directly (`chi_squared(df)`).

## Available tests

| Factory | Data | Config | Independent when |
|---|---|---|---|
| `chi_squared(data, yates = TRUE)` | discrete | `yates` | `p >= alpha` |
| `log_likelihood(data, yates = TRUE)` | discrete | `yates` | `p >= alpha` |
| `cressie_read(data, yates = TRUE)` | discrete | `yates` | `p >= alpha` |
| `freeman_tukey(data, yates = TRUE)` | discrete | `yates` | `p >= alpha` |
| `modified_likelihood(data, yates = TRUE)` | discrete | `yates` | `p >= alpha` |
| `pearson_correlation(data)` | continuous | — | `p >= alpha` |
| `fisher_z(data)` | continuous | — | `p >= alpha` |
| `pearson_equivalence(data, delta_threshold = 0.1)` | continuous | `delta_threshold` | `p < alpha` (TOST) |

`is_independent()` applies each test's own decision rule, so the caller never
writes it — including the equivalence test's inverted rule.

## Integration

```r
# pcalg: any test adapts to the indepTest(x, y, S, suffStat) callback shape.
indepTest <- as_pcalg(fisher_z(dataset(df)))
# pcalg::pc(suffStat = list(), indepTest = indepTest, labels = colnames(df), alpha = 0.05)

# base-R / bnlearn-style: an htest object.
ci_test(chi, "A", "B")
```

## Testing

```r
rextendr::document("crates/citest-r")   # recompile + regenerate wrappers
devtools::test("crates/citest-r")       # includes the shared 80-case golden fixture
archive <- devtools::build("crates/citest-r", manual = FALSE, vignettes = FALSE)
devtools::check_built(archive, args = "--no-manual", error_on = "warning")
```

The source package includes locked, vendored Rust dependencies plus synchronized
copies of `citest` and the golden fixture. From the repository root, refresh and
verify those project-owned copies with:

```bash
python crates/citest-r/tools/sync_package_assets.py --sync
python crates/citest-r/tools/sync_package_assets.py --check
```
