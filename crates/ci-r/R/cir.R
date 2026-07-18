#' cir: data-bound conditional-independence tests
#'
#' A small, function-style R surface over the Rust core. Bind a data.frame once
#' with [dataset()], construct a test with one of the factory functions
#' ([chi_squared()], [pearson_correlation()], ...), then query it with
#' [run_test()] / [is_independent()]. [as_pcalg()] adapts any test into the
#' `indepTest` callback shape expected by the \pkg{pcalg} package.
#'
#' Column kinds are inferred from the data.frame: `integer`/`logical`/`factor`/
#' `character` columns are treated as **discrete**, `double` columns as
#' **continuous**. Discrete factor/character columns are coded to integer codes
#' on the R side before they reach the Rust core (which only handles numbers).
#'
#' @keywords internal
"_PACKAGE"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# The generated extendr wrappers expose `Dataset`, `ChiSquared`, ... as
# environments with a `$new()` constructor returning an external pointer. The
# user-facing objects below wrap that pointer together with the ordered column
# names (needed to map pcalg's integer node indices back to names).

#' Infer the column kind ("discrete" or "continuous") for one data.frame column.
#'
#' `double` columns are continuous; everything else
#' (`integer`/`logical`/`factor`/`character`) is discrete.
#'
#' @param col A single data.frame column.
#' @return A scalar string, `"discrete"` or `"continuous"`.
#' @keywords internal
.cir_infer_kind <- function(col) {
  if (is.factor(col) || is.character(col) || is.logical(col) || is.integer(col)) {
    "discrete"
  } else if (is.double(col)) {
    "continuous"
  } else {
    stop(
      sprintf("unsupported column type: %s", paste(class(col), collapse = "/")),
      call. = FALSE
    )
  }
}

#' Code one data.frame column to a numeric vector for the Rust core.
#'
#' Discrete factor/character/logical columns are mapped to integer codes (the
#' core re-factorizes them anyway, so the exact codes only need to be
#' value-distinct); numeric columns pass through as doubles.
#'
#' @param col A single data.frame column.
#' @return A numeric vector the same length as `col`.
#' @keywords internal
.cir_code_column <- function(col) {
  if (is.factor(col)) {
    as.numeric(as.integer(col))
  } else if (is.character(col)) {
    as.numeric(as.integer(factor(col)))
  } else if (is.logical(col)) {
    as.numeric(col)
  } else {
    as.numeric(col)
  }
}

#' Bind a data.frame as a `cir` dataset.
#'
#' Builds the data-bound dataset shared by the test factories. Column kinds are
#' inferred from the column classes (integer/logical/factor/character ->
#' discrete, double -> continuous); discrete factor/character columns are coded
#' to integer codes before reaching the Rust core.
#'
#' Missing values are **rejected**: any `NA` (in a numeric column, a factor,
#' or after coding a character column) reaches the core as NaN and errors with
#' "missing data: ...". Choose a missing-data convention (e.g.
#' `na.omit(df)`) before binding.
#'
#' @param df A data.frame (or object coercible to one).
#' @return An object of class `cir_dataset` wrapping the bound dataset and its
#'   column names.
#' @examples
#' df <- data.frame(A = sample(0:1, 50, TRUE), X = rnorm(50))
#' data <- dataset(df)
#' @export
dataset <- function(df) {
  df <- as.data.frame(df)
  if (ncol(df) == 0L) {
    stop("`df` must have at least one column", call. = FALSE)
  }
  names_vec <- colnames(df)
  kinds <- vapply(df, .cir_infer_kind, character(1), USE.NAMES = FALSE)
  coded <- lapply(df, .cir_code_column)
  values <- matrix(
    as.numeric(unlist(coded, use.names = FALSE)),
    nrow = nrow(df),
    ncol = ncol(df)
  )
  ptr <- Dataset$new(names_vec, kinds, values)
  structure(
    list(ptr = ptr, names = names_vec, kinds = kinds),
    class = "cir_dataset"
  )
}

#' @export
print.cir_dataset <- function(x, ...) {
  cat(sprintf(
    "<cir_dataset: %d columns x %d rows>\n",
    x$ptr$n_cols(), x$ptr$n_rows()
  ))
  cat("  columns:", paste(x$names, collapse = ", "), "\n")
  invisible(x)
}

#' Coerce a `dataset()` result or a data.frame into a `cir_dataset`.
#'
#' Lets every test factory accept either a pre-built [dataset()] or a raw
#' data.frame uniformly.
#'
#' @param data A `cir_dataset` or a data.frame.
#' @return A `cir_dataset`.
#' @keywords internal
.cir_as_dataset <- function(data) {
  if (inherits(data, "cir_dataset")) {
    data
  } else {
    dataset(data)
  }
}

# ---------------------------------------------------------------------------
# Test factories
# ---------------------------------------------------------------------------

#' Build a `cir_test` wrapper around a constructed extendr test handle.
#'
#' @param handle The external-pointer test handle from `<Class>$new(...)`.
#' @param ds The bound `cir_dataset` (carries column names for pcalg mapping).
#' @param name The test's stable name (e.g. `"chi_squared"`).
#' @return An object of class `cir_test`.
#' @keywords internal
.cir_make_test <- function(handle, ds, name) {
  structure(
    list(handle = handle, dataset = ds, name = name),
    class = "cir_test"
  )
}

#' Chi-squared (Pearson, lambda = 1) discrete CI test.
#'
#' @param data A [dataset()] result or a data.frame.
#' @param yates Whether to apply Yates' continuity correction on 2x2
#'   sub-tables (default `TRUE`, the scipy/pgmpy convention).
#' @return A `cir_test` bound to `data`.
#' @examples
#' df <- data.frame(A = sample(0:1, 80, TRUE), B = sample(0:1, 80, TRUE))
#' chi <- chi_squared(dataset(df))
#' run_test(chi, "A", "B")
#' @export
chi_squared <- function(data, yates = TRUE) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(ChiSquared$new(ds$ptr, yates), ds, "chi_squared")
}

#' Log-likelihood / G-test (lambda = 0) discrete CI test.
#'
#' @inheritParams chi_squared
#' @return A `cir_test` bound to `data`.
#' @export
log_likelihood <- function(data, yates = TRUE) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(LogLikelihood$new(ds$ptr, yates), ds, "log_likelihood")
}

#' Cressie-Read (lambda = 2/3) discrete CI test.
#'
#' @inheritParams chi_squared
#' @return A `cir_test` bound to `data`.
#' @export
cressie_read <- function(data, yates = TRUE) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(CressieRead$new(ds$ptr, yates), ds, "cressie_read")
}

#' Freeman-Tukey (lambda = -1/2) discrete CI test.
#'
#' @inheritParams chi_squared
#' @return A `cir_test` bound to `data`.
#' @export
freeman_tukey <- function(data, yates = TRUE) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(FreemanTukey$new(ds$ptr, yates), ds, "freeman_tukey")
}

#' Modified log-likelihood (lambda = -1) discrete CI test.
#'
#' @inheritParams chi_squared
#' @return A `cir_test` bound to `data`.
#' @export
modified_likelihood <- function(data, yates = TRUE) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(ModifiedLikelihood$new(ds$ptr, yates), ds, "modified_likelihood")
}

#' Pearson partial-correlation continuous CI test.
#'
#' @param data A [dataset()] result or a data.frame.
#' @return A `cir_test` bound to `data`.
#' @export
pearson_correlation <- function(data) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(PearsonCorrelation$new(ds$ptr), ds, "pearson_correlation")
}

#' Fisher-z continuous CI test.
#'
#' Applies the Fisher z-transform to the (partial) correlation:
#' `sqrt(n - |Z| - 3) * atanh(rho)` is approximately standard normal under
#' independence. This matches `pcalg::gaussCItest`'s statistic and is the
#' de-facto standard continuous CI test in constraint-based causal discovery.
#'
#' @param data A [dataset()] result or a data.frame.
#' @return A `cir_test` bound to `data`.
#' @export
fisher_z <- function(data) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(FisherZ$new(ds$ptr), ds, "fisher_z")
}

#' Pearson equivalence (TOST) continuous CI test.
#'
#' Declares independence when the partial correlation is within the equivalence
#' margin. Note the inverted decision rule: independence holds when the p-value
#' is **below** the significance level (handled by [is_independent()]).
#'
#' @param data A [dataset()] result or a data.frame.
#' @param delta_threshold Equivalence margin on the correlation scale
#'   (default `0.1`).
#' @return A `cir_test` bound to `data`.
#' @export
pearson_equivalence <- function(data, delta_threshold = 0.1) {
  ds <- .cir_as_dataset(data)
  .cir_make_test(
    PearsonEquivalence$new(ds$ptr, delta_threshold), ds, "pearson_equivalence"
  )
}

#' @export
print.cir_test <- function(x, ...) {
  cat(sprintf("<cir_test: %s>\n", x$name))
  invisible(x)
}

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

#' Run a conditional-independence test.
#'
#' @param test A `cir_test` from one of the factory functions.
#' @param x,y Column **names** (length-1 character) for the two variables.
#' @param z Character vector of conditioning column names (default: none).
#' @return A named list `list(statistic, p_value, dof, effect_size)`; absent
#'   fields are `NULL`.
#' @examples
#' df <- data.frame(A = sample(0:1, 80, TRUE), B = sample(0:1, 80, TRUE))
#' run_test(chi_squared(dataset(df)), "A", "B")
#' @export
run_test <- function(test, x, y, z = character()) {
  stopifnot(inherits(test, "cir_test"))
  test$handle$run_test(x, y, .cir_as_names(z))
}

#' Decide conditional independence at a significance level.
#'
#' Applies the test's own decision rule: `p >= significance_level` for the
#' standard tests, and the inverted `p < significance_level` for
#' [pearson_equivalence()].
#'
#' @param test A `cir_test` from one of the factory functions.
#' @param x,y Column names for the two variables.
#' @param z Character vector of conditioning column names (default: none).
#' @param significance_level Significance level alpha (default `0.05`).
#' @return A single logical: `TRUE` if independent at `significance_level`.
#' @export
is_independent <- function(test, x, y, z = character(), significance_level = 0.05) {
  stopifnot(inherits(test, "cir_test"))
  test$handle$is_independent(x, y, .cir_as_names(z), significance_level)
}

#' Normalize a conditioning-set argument to a character vector.
#'
#' Accepts `NULL`, a character vector, or a single name; never a non-character.
#'
#' @param z The `z` argument as supplied by the caller.
#' @return A character vector (possibly empty).
#' @keywords internal
.cir_as_names <- function(z) {
  if (is.null(z)) {
    character()
  } else if (is.character(z)) {
    z
  } else {
    stop("`z` must be a character vector of column names", call. = FALSE)
  }
}

# ---------------------------------------------------------------------------
# pcalg adapter
# ---------------------------------------------------------------------------

# Test names whose decision rule is inverted relative to pcalg's standard
# convention: they declare independence when p < alpha, not p >= alpha. Mirrors
# the core `IndependenceRule::PValueLt` tests. Keep in sync if another
# inverted-rule test is added to the factories above.
.cir_inverted_rule_tests <- c("pearson_equivalence")

#' Adapt a `cir_test` into a \pkg{pcalg} `indepTest` callback.
#'
#' Returns a closure `function(x, y, S, suffStat)` matching the interface
#' [pcalg::pc()] / [pcalg::skeleton()] expect: `x` and `y` are **1-based**
#' integer node indices, `S` is an integer vector of conditioning node indices
#' (possibly empty), and the return value is the test's p-value. The node
#' indices are mapped back to the bound dataset's column names in their bound
#' order, so pass `labels = colnames(df)` (in the same order) to the pcalg
#' driver. `suffStat` is accepted for interface compatibility and ignored (the
#' data is already captured in `test`); pass `suffStat = list()`.
#'
#' @param test A `cir_test` from one of the factory functions. Must be a
#'   standard null-of-independence test; [pearson_equivalence()] is rejected
#'   because its inverted decision rule (independence when `p < alpha`) would be
#'   interpreted backwards by \pkg{pcalg}.
#' @return A function `(x, y, S, suffStat) -> numeric` p-value.
#' @examples
#' \dontrun{
#' df <- data.frame(X = rnorm(200), Y = rnorm(200), Z = rnorm(200))
#' indepTest <- as_pcalg(pearson_correlation(dataset(df)))
#' pcalg::pc(suffStat = list(), indepTest = indepTest,
#'           labels = colnames(df), alpha = 0.05)
#' }
#' @export
as_pcalg <- function(test) {
  stopifnot(inherits(test, "cir_test"))
  # pcalg interprets the returned p-value with the standard convention (remove an
  # edge / declare independence when p >= alpha). A test whose own rule is
  # inverted -- independence when p < alpha, i.e. the core's
  # `IndependenceRule::PValueLt` -- would have every decision silently flipped,
  # yielding a wrong skeleton/CPDAG with no error. Refuse them.
  if (test$name %in% .cir_inverted_rule_tests) {
    stop(sprintf(
      paste0(
        "as_pcalg() supports only standard null-of-independence tests; `%s` ",
        "uses an inverted decision rule (independence when p < alpha), which ",
        "pcalg would interpret backwards and silently invert every edge ",
        "decision. Use a standard test such as fisher_z() or ",
        "pearson_correlation() for constraint-based discovery."
      ),
      test$name
    ), call. = FALSE)
  }
  cols <- test$dataset$names
  handle <- test$handle
  function(x, y, S, suffStat) {
    xi <- cols[[x]]
    yi <- cols[[y]]
    zi <- if (length(S) > 0L) cols[as.integer(S)] else character()
    res <- handle$run_test(xi, yi, zi)
    res$p_value
  }
}

# ---------------------------------------------------------------------------
# Optional base-R `htest` adapter
# ---------------------------------------------------------------------------

#' Run a test and return a base-R `htest` object.
#'
#' A convenience wrapper for base-R / \pkg{bnlearn}-style workflows: the result
#' carries the `statistic` and the `parameter` (degrees of freedom) alongside
#' the `p.value`.
#'
#' @param test A `cir_test` from one of the factory functions.
#' @param x,y Column names for the two variables.
#' @param z Character vector of conditioning column names (default: none).
#' @return An object of class `htest`.
#' @export
ci_test <- function(test, x, y, z = character()) {
  stopifnot(inherits(test, "cir_test"))
  res <- run_test(test, x, y, z)

  statistic <- res$statistic
  if (!is.null(statistic)) {
    names(statistic) <- "statistic"
  }
  parameter <- res$dof
  if (!is.null(parameter)) {
    parameter <- as.numeric(parameter)
    names(parameter) <- "df"
  }

  zlab <- if (length(.cir_as_names(z)) > 0L) {
    paste0(" | ", paste(.cir_as_names(z), collapse = ", "))
  } else {
    ""
  }

  structure(
    list(
      statistic = statistic,
      parameter = parameter,
      p.value = res$p_value,
      estimate = if (is.null(res$effect_size)) NULL else c(effect_size = res$effect_size),
      method = sprintf("cir %s conditional independence test", test$name),
      data.name = sprintf("%s vs %s%s", x, y, zlab)
    ),
    class = "htest"
  )
}
