# pcalg integration demo for the cir package.
#
# Builds a continuous dataset with a known chain structure X -> Y -> Z, adapts a
# cir Pearson-correlation test into a pcalg `indepTest` callback via as_pcalg(),
# and runs pcalg::pc() to recover the skeleton/CPDAG. Run with:
#
#   conda run -n expert Rscript crates/ci-r/inst/examples/pcalg_demo.R
#
# (load_all is used so the demo works straight from the source tree; once the
# package is installed, `library(cir)` is enough.)

if (
  requireNamespace("devtools", quietly = TRUE) &&
    dir.exists("crates/ci-r")
) {
  suppressMessages(devtools::load_all("crates/ci-r", quiet = TRUE))
} else {
  library(cir)
}

stopifnot(requireNamespace("pcalg", quietly = TRUE))

set.seed(42)
n <- 500
# Chain X -> Y -> Z: Y depends on X, Z depends on Y. Conditioning on Y should
# render X and Z independent.
x <- rnorm(n)
y <- x + rnorm(n)
z <- y + rnorm(n)
df <- data.frame(X = x, Y = y, Z = z)

data <- dataset(df)
indep_test <- as_pcalg(pearson_correlation(data))

# suffStat is ignored by our adapter (the data is captured in the test), so an
# empty list suffices.
fit <- pcalg::pc(
  suffStat = list(),
  indepTest = indep_test,
  labels = colnames(df),
  alpha = 0.05
)

cat("pcalg::pc() completed.\n")
cat("Estimated CPDAG adjacency matrix:\n")
print(as(fit@graph, "matrix"))

# Sanity touch on the adapter independence direction.
cat(sprintf(
  "\nMarginal X--Z p-value:      %.4g\n",
  indep_test(1, 3, integer(0), list())
))
cat(sprintf(
  "X--Z | Y p-value (should be high): %.4g\n",
  indep_test(1, 3, c(2L), list())
))
cat("\npcalg demo OK\n")
