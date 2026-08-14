# Surface tests for the data-bound R API: dataset binding, result shapes,
# factor coding, the equivalence decision inversion, error handling, the htest
# adapter, and the pcalg closure.

library(citest)

discrete_df <- function() {
  set.seed(0)
  data.frame(
    A = sample(0:1, 200, TRUE),
    B = sample(0:1, 200, TRUE),
    C = sample(0:2, 200, TRUE)
  )
}

continuous_df <- function() {
  set.seed(1)
  data.frame(
    X = rnorm(200),
    Y = rnorm(200),
    Z = rnorm(200)
  )
}

test_that("dataset() infers kinds and reports shape / index", {
  data <- dataset(discrete_df())
  expect_s3_class(data, "citest_dataset")
  expect_equal(data$ptr$n_rows(), 200L)
  expect_equal(data$ptr$n_cols(), 3L)
  expect_equal(data$ptr$index_of("A"), 1L) # 1-based
  expect_equal(data$ptr$index_of("C"), 3L)
  expect_true(is.na(data$ptr$index_of("nope")))
  expect_equal(data$kinds, c("discrete", "discrete", "discrete"))
})

test_that("continuous columns are inferred as continuous", {
  data <- dataset(continuous_df())
  expect_equal(unname(data$kinds), c("continuous", "continuous", "continuous"))
})

test_that("run_test returns the uniform list for discrete tests", {
  chi <- chi_squared(dataset(discrete_df()))
  res <- run_test(chi, "A", "B")
  expect_named(res, c("statistic", "p_value", "dof", "effect_size"))
  expect_true(is.numeric(res$statistic))
  expect_true(is.numeric(res$p_value))
  expect_true(is.numeric(res$dof))
})

test_that("z defaults to an empty conditioning set", {
  chi <- chi_squared(dataset(discrete_df()))
  expect_equal(
    run_test(chi, "A", "B")$p_value,
    run_test(chi, "A", "B", character())$p_value
  )
})

test_that("conditioning on a name works", {
  chi <- chi_squared(dataset(discrete_df()))
  res <- run_test(chi, "A", "C", c("B"))
  expect_true(is.numeric(res$p_value))
})

test_that("is_independent returns a single logical", {
  chi <- chi_squared(dataset(discrete_df()))
  out <- is_independent(chi, "A", "C", c("B"), significance_level = 0.05)
  expect_true(is.logical(out))
  expect_length(out, 1L)
})

test_that("is_independent rejects non-finite significance levels", {
  chi <- chi_squared(dataset(discrete_df()))
  for (level in c(NaN, Inf, -Inf)) {
    expect_error(
      is_independent(chi, "A", "B", significance_level = level),
      "significance level must be finite"
    )
  }
})

test_that("pearson_equivalence has NULL dof and the inverted rule", {
  eqv <- pearson_equivalence(dataset(continuous_df()), delta_threshold = 0.1)
  res <- run_test(eqv, "X", "Y", c("Z"))
  expect_null(res$dof)
  expect_true(is.numeric(res$p_value))
  # The equivalence rule is independent <=> p < alpha. With alpha = 1 every
  # finite p-value is < 1, so it must report independent; with alpha = 0 none
  # can be, so it must report dependent. This pins the inversion direction.
  expect_true(is_independent(eqv, "X", "Y", c("Z"), significance_level = 1))
  expect_false(is_independent(eqv, "X", "Y", c("Z"), significance_level = 0))
})

test_that("the standard rule is the non-inverted direction", {
  chi <- chi_squared(dataset(discrete_df()))
  # Standard rule: independent <=> p >= alpha. alpha = 0 => always independent;
  # alpha = 1 => never (finite p < 1).
  expect_true(
    is_independent(chi, "A", "B", character(), significance_level = 0)
  )
  expect_false(
    is_independent(chi, "A", "B", character(), significance_level = 1)
  )
})

test_that("pearson_correlation takes no config", {
  res <- run_test(pearson_correlation(dataset(continuous_df())), "X", "Y")
  # The Pearson correlation t-test reports dof = n - |Z| - 2 (here 200 - 0 - 2).
  expect_equal(res$dof, 198L)
  expect_true(is.numeric(res$p_value))
})

test_that("factory functions accept a raw data.frame", {
  chi <- chi_squared(discrete_df())
  expect_s3_class(chi, "citest_test")
  expect_true(is.numeric(run_test(chi, "A", "B")$p_value))
})

test_that("factor and character columns are coded to discrete", {
  set.seed(3)
  df <- data.frame(
    F = factor(sample(c("a", "b", "c"), 200, TRUE)),
    G = sample(c("yes", "no"), 200, TRUE),
    stringsAsFactors = FALSE
  )
  data <- dataset(df)
  expect_equal(unname(data$kinds), c("discrete", "discrete"))
  res <- run_test(chi_squared(data), "F", "G")
  expect_true(is.numeric(res$statistic))
  # A three-by-two table has (3 - 1) * (2 - 1) degrees of freedom.
  expect_equal(res$dof, 2L)
})

test_that("a dataset can be shared across tests", {
  data <- dataset(continuous_df())
  a <- run_test(pearson_correlation(data), "X", "Y")
  b <- run_test(pearson_equivalence(data, delta_threshold = 0.2), "X", "Y")
  expect_true(is.numeric(a$p_value))
  expect_true(is.numeric(b$p_value))
})

test_that("unknown column names raise an error", {
  chi <- chi_squared(dataset(discrete_df()))
  expect_error(run_test(chi, "A", "nope"), "unknown column")
})

test_that("a continuous column handed to a discrete test errors", {
  chi <- chi_squared(dataset(continuous_df()))
  expect_error(run_test(chi, "X", "Y"), "wrong column kind")
})

test_that("ci_test returns an htest with statistic, parameter, p.value", {
  chi <- chi_squared(dataset(discrete_df()))
  # Unconditional 2x2 table (A, B both binary) => dof = (2-1)*(2-1) = 1.
  h <- ci_test(chi, "A", "B")
  expect_s3_class(h, "htest")
  expect_true(is.numeric(h$statistic))
  expect_equal(unname(h$parameter), 1)
  expect_true(is.numeric(h$p.value))
})

test_that("as_pcalg returns a (x, y, S, suffStat) -> p.value closure", {
  data <- dataset(continuous_df())
  indep <- as_pcalg(pearson_correlation(data))
  expect_type(indep, "closure")
  # 1-based node indices into the bound columns (X=1, Y=2, Z=3).
  p_uncond <- indep(1, 2, integer(0), list())
  p_cond <- indep(1, 2, c(3L), list())
  expect_true(is.numeric(p_uncond) && length(p_uncond) == 1L)
  expect_true(is.numeric(p_cond) && length(p_cond) == 1L)
  # Matches the direct run_test p-value.
  expect_equal(p_uncond, run_test(pearson_correlation(data), "X", "Y")$p_value)
})

test_that("fisher_z runs and reports no dof", {
  set.seed(11)
  n <- 80
  z <- rnorm(n)
  df <- data.frame(
    X = 1.2 * z + 0.4 * rnorm(n),
    Y = 0.7 * z + 0.4 * rnorm(n),
    Z = z
  )
  fz <- fisher_z(dataset(df))
  res <- run_test(fz, "X", "Y", c("Z"))
  expect_null(res$dof)
  expect_true(res$p_value >= 0 && res$p_value <= 1)
  # statistic = sqrt(n - |Z| - 3) * atanh(r)
  pc <- run_test(pearson_correlation(dataset(df)), "X", "Y", c("Z"))
  expect_equal(
    res$statistic,
    sqrt(n - 1 - 3) * atanh(pc$statistic),
    tolerance = 1e-9
  )
})

test_that("NA values error at dataset construction", {
  df <- data.frame(A = c(1L, NA_integer_, 0L), X = c(0.1, 0.2, 0.3))
  expect_error(dataset(df), "missing data")
  df2 <- data.frame(A = c(0L, 1L, 0L), X = c(0.1, NA_real_, 0.3))
  expect_error(dataset(df2), "missing data")
  df3 <- data.frame(A = factor(c("u", NA, "v")), X = c(0.1, 0.2, 0.3))
  expect_error(dataset(df3), "missing data")
  # Character columns are coded via a separate branch of .citest_code_column.
  df4 <- data.frame(A = c("u", NA, "v"), X = c(0.1, 0.2, 0.3),
                    stringsAsFactors = FALSE)
  expect_error(dataset(df4), "missing data")
})

test_that("invalid queries error", {
  df <- data.frame(
    A = sample(0:1, 40, TRUE),
    B = sample(0:1, 40, TRUE),
    C = sample(0:1, 40, TRUE)
  )
  chi <- chi_squared(dataset(df))
  expect_error(run_test(chi, "A", "A"), "invalid query")
  expect_error(run_test(chi, "A", "B", c("A")), "invalid query")
  expect_error(run_test(chi, "A", "B", c("C", "C")), "invalid query")
})

test_that("meta() reports each test's own decision rule", {
  df <- data.frame(
    A = sample(0:1, 40, TRUE),
    B = sample(0:1, 40, TRUE),
    X = rnorm(40),
    Y = rnorm(40)
  )
  ds <- dataset(df)

  m <- meta(chi_squared(ds))
  expect_equal(m$name, "chi_squared")
  expect_equal(m$data_types, "discrete")
  expect_true(m$symmetric)
  expect_equal(m$rule, "p_value_ge")

  expect_equal(meta(fisher_z(ds))$data_types, "continuous")
  # The one test with the inverted convention must say so itself.
  expect_equal(meta(pearson_equivalence(ds))$rule, "p_value_lt")
})

test_that("as_pcalg() refuses inverted-rule tests by asking their metadata", {
  df <- data.frame(X = rnorm(60), Y = rnorm(60), Z = rnorm(60))
  ds <- dataset(df)

  expect_true(is.function(as_pcalg(fisher_z(ds))))
  expect_error(as_pcalg(pearson_equivalence(ds)), "inverted decision rule")

  # The guard reads the rule rather than matching a hardcoded name, so a test
  # whose rule is p_value_lt is refused whatever it is called. Standing in for
  # a future ninth test, which would otherwise silently invert every pcalg edge
  # decision until someone remembered to update a literal.
  disguised <- pearson_equivalence(ds)
  disguised$name <- "some_future_equivalence_test"
  expect_error(as_pcalg(disguised), "inverted decision rule")
})

test_that("registry and R factories agree in both directions", {
  # The core registry is the single source of truth. Without this check a ninth
  # test could be added to the core and silently missing from this binding.
  registered <- vapply(list_tests(), function(m) m$name, character(1))
  expect_length(registered, 8L)

  exported <- ls(asNamespace("citest"))
  for (name in registered) {
    expect_true(
      name %in% exported,
      info = sprintf("%s is registered in the core but has no R factory", name)
    )
  }

  df <- data.frame(A = sample(0:1, 30, TRUE), X = rnorm(30))
  ds <- dataset(df)
  for (name in registered) {
    built <- do.call(name, list(ds))
    expect_s3_class(built, "citest_test")
    expect_equal(meta(built)$name, name)
  }
})

test_that("list_tests() reports each decision rule", {
  metas <- list_tests()
  by_name <- stats::setNames(
    metas,
    vapply(metas, function(m) m$name, character(1))
  )
  expect_equal(by_name$chi_squared$rule, "p_value_ge")
  expect_equal(by_name$chi_squared$data_types, "discrete")
  expect_equal(by_name$pearson_equivalence$rule, "p_value_lt")
  expect_equal(by_name$fisher_z$data_types, "continuous")
})
