# Golden parity test for the data-bound R bindings.
#
# Loads the shared cross-language fixture `tests/fixtures/golden.json` and, for
# each of the seven tests, builds a `dataset()` from the case's columns,
# constructs the test with the case's parameters, runs it, and asserts that
# `statistic` / `p_value` / `dof` match the recorded `expected` values within a
# tight tolerance. This is the binding's numeric parity gate against the
# scipy/pgmpy reference.

library(cir)

TOL <- 1e-7
EXPECTED_CASE_COUNT <- 73L

# Map the fixture's stable test name to its binding factory function.
TEST_FACTORIES <- list(
  chi_squared = chi_squared,
  log_likelihood = log_likelihood,
  cressie_read = cressie_read,
  freeman_tukey = freeman_tukey,
  modified_likelihood = modified_likelihood,
  pearson_correlation = pearson_correlation,
  pearson_equivalence = pearson_equivalence
)

# Locate the repo-root fixture. Under `devtools::test()`, `test_path()` points
# at this directory (crates/ci-r/tests/testthat), so the repo root is four
# levels up; fall back to walking up from the working directory.
find_golden <- function() {
  candidate <- testthat::test_path("..", "..", "..", "..", "tests", "fixtures", "golden.json")
  if (file.exists(candidate)) {
    return(normalizePath(candidate))
  }
  dir <- normalizePath(getwd())
  repeat {
    candidate <- file.path(dir, "tests", "fixtures", "golden.json")
    if (file.exists(candidate)) {
      return(candidate)
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) {
      break
    }
    dir <- parent
  }
  stop("could not locate tests/fixtures/golden.json")
}

load_cases <- function() {
  jsonlite::fromJSON(find_golden(), simplifyVector = FALSE)
}

# Build a `cir_dataset` from a fixture case's `columns` map. Discrete columns
# are coerced to integer (so `dataset()` infers "discrete"); continuous columns
# stay double ("continuous").
build_dataset <- function(columns) {
  cols <- lapply(columns, function(col) {
    values <- as.numeric(unlist(col$values))
    if (identical(col$kind, "discrete")) {
      as.integer(values)
    } else {
      as.double(values)
    }
  })
  df <- as.data.frame(cols, stringsAsFactors = FALSE, check.names = FALSE)
  dataset(df)
}

construct_test <- function(name, data, params) {
  factory <- TEST_FACTORIES[[name]]
  if (name == "pearson_correlation") {
    factory(data)
  } else if (name == "pearson_equivalence") {
    factory(data, delta_threshold = params$delta_threshold)
  } else {
    # Discrete power-divergence family: configured by `yates`.
    factory(data, yates = params$yates)
  }
}

# Compare one numeric field against its expected value. Null expectations are
# skipped (the test does not define them); Inf / NaN are matched defensively.
assert_close <- function(actual, expected, field, case_id) {
  if (is.null(expected)) {
    return(invisible())
  }
  expect_false(is.null(actual),
    label = sprintf("%s: expected %s=%s, got NULL", case_id, field, expected)
  )
  exp <- as.numeric(expected)
  act <- as.numeric(actual)
  if (is.infinite(exp) || is.nan(exp)) {
    expect_equal(is.infinite(act), is.infinite(exp),
      label = sprintf("%s: %s inf mismatch", case_id, field)
    )
    expect_equal(is.nan(act), is.nan(exp),
      label = sprintf("%s: %s nan mismatch", case_id, field)
    )
    if (is.infinite(exp)) {
      expect_equal(sign(act), sign(exp),
        label = sprintf("%s: %s inf sign mismatch", case_id, field)
      )
    }
    return(invisible())
  }
  expect_equal(act, exp,
    tolerance = TOL,
    label = sprintf("%s: %s got %s, expected %s", case_id, field, act, exp)
  )
}

GOLDEN_CASES <- load_cases()

test_that("golden fixture covers exactly the expected cases", {
  expect_equal(length(GOLDEN_CASES), EXPECTED_CASE_COUNT)
  names_seen <- unique(vapply(GOLDEN_CASES, function(c) c$test, character(1)))
  expect_setequal(names_seen, names(TEST_FACTORIES))
})

test_that("each golden case reproduces its recorded statistic / p_value / dof", {
  for (i in seq_along(GOLDEN_CASES)) {
    case <- GOLDEN_CASES[[i]]
    case_id <- sprintf("case[%d] %s", i - 1L, case$test)

    data <- build_dataset(case$columns)
    test <- construct_test(case$test, data, case$params)

    z <- as.character(unlist(case$z))
    if (length(z) == 0L) {
      z <- character()
    }
    result <- run_test(test, case$x, case$y, z)
    expected <- case$expected

    assert_close(result$statistic, expected$statistic, "statistic", case_id)

    # p-value: handle an exact-zero expectation defensively.
    exp_p <- as.numeric(expected$p_value)
    if (identical(exp_p, 0)) {
      expect_equal(as.numeric(result$p_value), 0, tolerance = TOL,
        label = sprintf("%s: p_value expected ~0, got %s", case_id, result$p_value)
      )
    } else {
      assert_close(result$p_value, exp_p, "p_value", case_id)
    }

    dof_actual <- if (is.null(result$dof)) NULL else as.numeric(result$dof)
    assert_close(dof_actual, expected$dof, "dof", case_id)
  }
})
