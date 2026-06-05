library(cir)

test_that("unconditional independent data is not rejected", {
  x <- c(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
  y <- c(1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
  z <- matrix(0, nrow = 8, ncol = 0)

  result <- cressie_read_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "statistic")
  expect_true(result$statistic < EPS)
  expect_true(result$p_value > 0.99)
  expect_equal(result$df, 1)
})

test_that("unconditional boolean accepts independent data", {
  x <- c(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
  y <- c(1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
  z <- matrix(0, nrow = 8, ncol = 0)

  result <- cressie_read_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_true(result$independent)
})

test_that("unconditional dependent data is rejected", {
  x <- c(1., 1., 1., 1., 2., 2., 2., 2.)
  y <- c(1., 1., 1., 1., 2., 2., 2., 2.)
  z <- matrix(0, nrow = 8, ncol = 0)

  result <- cressie_read_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "statistic")
  expect_true(result$statistic > 5.0)
  expect_true(result$p_value < 0.05)
  expect_equal(result$df, 1)
})

test_that("unconditional boolean rejects dependent data", {
  x <- c(1., 1., 1., 1., 2., 2., 2., 2.)
  y <- c(1., 1., 1., 1., 2., 2., 2., 2.)
  z <- matrix(0, nrow = 8, ncol = 0)

  result <- cressie_read_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_false(result$independent)
})

test_that("conditional independent data is not rejected", {
  x <- c(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
  y <- c(1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
  z <- matrix(c(0., 0., 0., 0., 1., 1., 1., 1.), nrow = 8, ncol = 1)

  result <- cressie_read_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "statistic")
  expect_true(result$statistic < EPS)
  expect_true(result$p_value > 0.99)
  expect_equal(result$df, 2)
})

test_that("conditional boolean accepts conditionally independent data", {
  x <- c(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
  y <- c(1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
  z <- matrix(c(0., 0., 0., 0., 1., 1., 1., 1.), nrow = 8, ncol = 1)

  result <- cressie_read_test(x, y, z, TRUE, 0.05)
  expect_equal(result$kind, "boolean")
  expect_true(result$independent)
})

test_that("conditional dependent data is rejected", {
  x <- c(1., 1., 2., 2., 1., 1., 2., 2.)
  y <- c(1., 1., 2., 2., 1., 1., 2., 2.)
  z <- matrix(c(0., 0., 0., 0., 1., 1., 1., 1.), nrow = 8, ncol = 1)

  result <- cressie_read_test(x, y, z, FALSE, 0.05)
  expect_equal(result$kind, "statistic")
  expect_true(result$statistic > 5.0)
  expect_true(result$p_value < 0.05)
})
