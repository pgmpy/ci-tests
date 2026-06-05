library(cir)
n <- 1000

test_that("unconditional independent data is not rejected", {
  set.seed(42)
  x <- rnorm(n)
  y <- rnorm(n)
  z <- matrix(0, nrow = n, ncol = 0)

  result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "pvalue")
  expect_true(result$p_value > 0.05)
  expect_true(abs(result$coefficient) < 0.1)
})

test_that("unconditional boolean accepts independent data", {
  set.seed(42)
  x <- rnorm(n)
  y <- rnorm(n)
  z <- matrix(0, nrow = n, ncol = 0)

  result <- pearson_correlation_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_true(result$independent)
})

test_that("unconditional dependent data is rejected", {
  set.seed(42)
  x <- rnorm(n)
  noise <- rnorm(n, sd = 0.1)
  y <- 3 * x + noise
  z <- matrix(0, nrow = n, ncol = 0)

  result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "pvalue")
  expect_true(result$p_value < 0.05)
  expect_true(abs(result$coefficient) > 0.9)
})

test_that("unconditional boolean rejects dependent data", {
  set.seed(42)
  x <- rnorm(n)
  noise <- rnorm(n, sd = 0.1)
  y <- 3 * x + noise
  z <- matrix(0, nrow = n, ncol = 0)

  result <- pearson_correlation_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_false(result$independent)
})

test_that("conditional independent data is not rejected", {
  set.seed(42)
  z_col <- rnorm(n)
  noise_x <- rnorm(n, sd = 0.1)
  noise_y <- rnorm(n, sd = 0.1)
  x <- 3 * z_col + noise_x
  y <- 2 * z_col + noise_y
  z <- matrix(z_col, nrow = n, ncol = 1)

  result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "pvalue")
  expect_true(result$p_value > 0.05)
  expect_true(abs(result$coefficient) < 0.1)
})

test_that("conditional boolean accepts conditionally independent data", {
  set.seed(42)
  z_col <- rnorm(n)
  noise_x <- rnorm(n, sd = 0.1)
  noise_y <- rnorm(n, sd = 0.1)
  x <- 3 * z_col + noise_x
  y <- 2 * z_col + noise_y
  z <- matrix(z_col, nrow = n, ncol = 1)

  result <- pearson_correlation_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_true(result$independent)
})

test_that("conditional dependent data (v-structure collider) is rejected", {
  set.seed(42)
  x <- rnorm(n)
  y <- rnorm(n)
  noise <- rnorm(n, sd = 0.1)
  z_col <- 2 * x + 2 * y + noise
  z <- matrix(z_col, nrow = n, ncol = 1)

  result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "pvalue")
  expect_true(result$p_value < 0.05)
  expect_true(abs(result$coefficient) > 0.9)
})

test_that("conditional boolean rejects dependent data (v-structure collider)", {
  set.seed(42)
  x <- rnorm(n)
  y <- rnorm(n)
  noise <- rnorm(n, sd = 0.1)
  z_col <- 2 * x + 2 * y + noise
  z <- matrix(z_col, nrow = n, ncol = 1)

  result <- pearson_correlation_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_false(result$independent)
})

test_that("conditional independent data with multiple conditioning variables is not rejected", {
  set.seed(42)
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  z3 <- rnorm(n)
  noise_x <- rnorm(n, sd = 0.1)
  noise_y <- rnorm(n, sd = 0.1)
  x <- 0.5 * z1 + 0.5 * z2 + 0.5 * z3 + noise_x
  y <- 0.5 * z1 + 0.5 * z2 + 0.5 * z3 + noise_y
  z <- cbind(z1, z2, z3)

  result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "pvalue")
  expect_true(result$p_value >= 0.05)
  expect_true(abs(result$coefficient) <= 0.1)
})

test_that("minimum input (n=3) with perfect correlation returns coefficient near 1", {
  x <- c(1.0, 2.0, 3.0)
  y <- c(1.0, 2.0, 3.0)
  z <- matrix(0, nrow = 3, ncol = 0)

  result <- pearson_correlation_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "pvalue")
  expect_true(abs(result$coefficient - 1.0) < 1e-10)
  expect_true(result$p_value < 0.05)
})
