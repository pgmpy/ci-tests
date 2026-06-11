# Surface tests for the data-bound R API: dataset binding, result shapes,
# factor coding, the equivalence decision inversion, error handling, the htest
# adapter, and the pcalg closure.

library(cir)

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
  expect_s3_class(data, "cir_dataset")
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
  expect_equal(run_test(chi, "A", "B")$p_value, run_test(chi, "A", "B", character())$p_value)
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
  expect_true(is_independent(chi, "A", "B", character(), significance_level = 0))
  expect_false(is_independent(chi, "A", "B", character(), significance_level = 1))
})

test_that("pearson_correlation takes no config", {
  res <- run_test(pearson_correlation(dataset(continuous_df())), "X", "Y")
  # The Pearson correlation t-test reports dof = n - |Z| - 2 (here 200 - 0 - 2).
  expect_equal(res$dof, 198L)
  expect_true(is.numeric(res$p_value))
})

test_that("factory functions accept a raw data.frame", {
  chi <- chi_squared(discrete_df())
  expect_s3_class(chi, "cir_test")
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
  expect_equal(res$dof, 2L) # (3-1)*(2-1)
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
