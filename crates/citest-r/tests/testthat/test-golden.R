# Golden parity test for the data-bound R bindings.
#
# Loads the packaged copy of the shared cross-language golden fixture and, for
# each of the eight tests, builds a `dataset()` from the case's columns,
# constructs the test with the case's parameters, runs it, and asserts that
# `statistic` / `p_value` / `dof` / `effect_size` match the recorded `expected`
# values within a tight tolerance. This is the binding's numeric parity gate
# against the scipy/pgmpy reference.

library(citest)

tol <- 1e-7
expected_case_count <- 80L

# Map the fixture's stable test name to its binding factory function.
test_factories <- list(
  chi_squared = chi_squared,
  log_likelihood = log_likelihood,
  cressie_read = cressie_read,
  freeman_tukey = freeman_tukey,
  modified_likelihood = modified_likelihood,
  pearson_correlation = pearson_correlation,
  fisher_z = fisher_z,
  pearson_equivalence = pearson_equivalence
)

find_golden <- function() {
  testthat::test_path("fixtures", "golden.json")
}

load_cases <- function() {
  jsonlite::fromJSON(find_golden(), simplifyVector = FALSE)
}

# Build a `citest_dataset` from a fixture case's `columns` map. Discrete columns
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
  factory <- test_factories[[name]]
  if (name %in% c("pearson_correlation", "fisher_z")) {
    factory(data)
  } else if (name == "pearson_equivalence") {
    factory(data, delta_threshold = params$delta_threshold)
  } else {
    # Discrete power-divergence family: configured by `yates`.
    factory(data, yates = params$yates)
  }
}

# Compare one numeric field against its expected value. Null expectations
# require a NULL result; Inf / NaN are matched defensively.
assert_close <- function(actual, expected, field, case_id) {
  if (is.null(expected)) {
    expect_null(actual,
      label = sprintf("%s: expected %s=NULL, got %s", case_id, field, actual)
    )
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
    tolerance = tol,
    label = sprintf("%s: %s got %s, expected %s", case_id, field, act, exp)
  )
}

golden_cases <- load_cases()

test_that("golden fixture covers exactly the expected cases", {
  expect_equal(length(golden_cases), expected_case_count)
  names_seen <- unique(vapply(golden_cases, function(c) c$test, character(1)))
  expect_setequal(names_seen, names(test_factories))
})

test_that("each golden case reproduces its complete recorded result", {
  for (i in seq_along(golden_cases)) {
    case <- golden_cases[[i]]
    case_id <- case$id
    expect_true(nzchar(case_id), label = "fixture case ID must not be empty")
    expect_true(case$test %in% names(test_factories),
      label = sprintf("%s: unknown test %s", case_id, case$test)
    )

    data <- build_dataset(case$columns)
    test <- construct_test(case$test, data, case$params)

    z <- as.character(unlist(case$z))
    if (length(z) == 0L) {
      z <- character()
    }
    result <- run_test(test, case$x, case$y, z)
    expected <- case$expected

    assert_close(result$statistic, expected$statistic, "statistic", case_id)

    assert_close(result$p_value, expected$p_value, "p_value", case_id)

    dof_actual <- if (is.null(result$dof)) NULL else as.numeric(result$dof)
    assert_close(dof_actual, expected$dof, "dof", case_id)
    assert_close(
      result$effect_size,
      expected$effect_size,
      "effect_size",
      case_id
    )
  }
})
