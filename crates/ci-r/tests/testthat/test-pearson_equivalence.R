library(cir)
n <- 1000

test_that("can call pearson_equivalence_test and get a pvalue result", {
  set.seed(42)
  x <- rnorm(n)
  y <- rnorm(n)
  z <- matrix(0, nrow = n, ncol = 0)

  result <- pearson_equivalence_test(x, y, z, FALSE, 0.05, 0.1)
  expect_equal(result$kind, "pvalue")
  expect_true(is.numeric(result$p_value))
  expect_true(is.numeric(result$coefficient))
})

test_that("can call pearson_equivalence_test and get a boolean result", {
  set.seed(42)
  x <- rnorm(n)
  y <- rnorm(n)
  z <- matrix(0, nrow = n, ncol = 0)

  result <- pearson_equivalence_test(x, y, z, TRUE, 0.05, 0.1)
  expect_equal(result$kind, "boolean")
  expect_true(is.logical(result$independent))
})
