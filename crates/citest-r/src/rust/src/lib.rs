//! R bindings for the data-bound conditional-independence testing core.
//!
//! The surface mirrors `citest`'s redesigned contract: a [`Dataset`] is built
//! once from named, typed columns, and each test is a *data-bound* handle that
//! captures the dataset (plus its own configuration) and then answers
//! `run_test(x, y, z)` / `is_independent(...)` queries by column **name**.
//!
//! These extendr exports are the low-level layer; the user-facing R API
//! (`dataset()`, the test factories, `run_test()`, `is_independent()`,
//! `as_pcalg()`) lives in `R/citest.R` and wraps the external-pointer objects
//! created here.
//!
//! Discrete columns must already be numeric when they reach Rust: the R side
//! factor/character → integer coding happens before [`Dataset::new`], so the
//! core only ever sees `f64` values (which it factorizes into contiguous codes).
//! Core [`citest::error::CiError`]s and Rust panics surface as R errors via
//! extendr's `Result` support.

// Every fallible export returns `extendr_api::Result<_>`, which extendr turns
// into an R error; the failure modes (unknown column, wrong column kind,
// dimension mismatch, degenerate data) are documented once here rather than
// repeated on every method.
#![allow(clippy::missing_errors_doc)]

use std::sync::Arc;

use citest::dataset::{ColumnKind, Dataset as CoreDataset};
use citest::strategy::{CITest, CiResult as CoreResult};
use extendr_api::prelude::*;
use extendr_api::Result;

mod util;

use util::{result_to_list, to_r_error};

/// A named, typed, immutable table of columns shared by the test handles.
///
/// Built from R via the `new` constructor, which takes the column names, a
/// parallel vector of kind strings (`"discrete"` / `"continuous"`), and the
/// columns flattened into a single `f64` matrix (column-major, one dataset
/// column per matrix column). Discrete columns are factorized into integer
/// codes inside the core. The handle is cheap to clone (it is `Arc`-shared) so
/// several tests can bind the same factorization.
#[extendr]
#[derive(Clone)]
struct Dataset {
    inner: Arc<CoreDataset>,
}

/// Parse a kind string into a [`ColumnKind`].
fn parse_kind(kind: &str) -> Result<ColumnKind> {
    match kind {
        "discrete" => Ok(ColumnKind::Discrete),
        "continuous" => Ok(ColumnKind::Continuous),
        other => Err(Error::Other(format!(
            "column kind must be \"discrete\" or \"continuous\", got {other:?}"
        ))),
    }
}

#[extendr]
impl Dataset {
    /// Build a dataset from `names`, parallel `kinds`, and a `values` matrix
    /// whose `j`-th column holds the values of column `names[j]`.
    ///
    /// The R wrapper (`dataset()`) is responsible for coding factor/character
    /// columns to numbers before calling this, so `values` is always numeric.
    fn new(names: Strings, kinds: Strings, values: RMatrix<f64>) -> Result<Self> {
        let n_cols = names.len();
        if kinds.len() != n_cols {
            return Err(Error::Other(format!(
                "names ({n_cols}) and kinds ({}) must have the same length",
                kinds.len()
            )));
        }
        let nrow = values.nrows();
        let ncol = values.ncols();
        if ncol != n_cols {
            return Err(Error::Other(format!(
                "values has {ncol} columns but {n_cols} names were given"
            )));
        }
        let data = values.data();
        let mut cols: Vec<(String, ColumnKind, Vec<f64>)> = Vec::with_capacity(n_cols);
        // `Strings::iter()` in extendr 0.9 constructs a slice from R's backing
        // pointer. R 4.6 returns a null pointer for `character(0)`, which makes
        // even a zero-length iteration abort under Rust's UB checks. Indexing
        // also handles empty vectors without asking extendr for that pointer.
        for j in 0..n_cols {
            let name = names.elt(j);
            let kind = parse_kind(kinds.elt(j).as_ref())?;
            // Column-major: column `j` occupies the contiguous slice
            // `[j*nrow, (j+1)*nrow)`.
            let start = j * nrow;
            let col = data[start..start + nrow].to_vec();
            cols.push((name.to_string(), kind, col));
        }
        let inner = CoreDataset::from_columns(cols).map_err(to_r_error)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Number of rows (observations).
    fn n_rows(&self) -> i32 {
        // Row counts here are far below i32::MAX; the cast cannot overflow.
        i32::try_from(self.inner.n_rows()).unwrap_or(i32::MAX)
    }

    /// Number of columns.
    fn n_cols(&self) -> i32 {
        i32::try_from(self.inner.n_cols()).unwrap_or(i32::MAX)
    }

    /// 1-based index of the column named `name`, or `NA` if it is absent.
    fn index_of(&self, name: &str) -> Robj {
        match self.inner.index_of(name) {
            Some(idx) => {
                // 1-based for R. Counts stay well within i32.
                let one_based = i32::try_from(idx + 1).unwrap_or(i32::MAX);
                one_based.into()
            }
            None => Robj::from(NA_INTEGER),
        }
    }
}

impl Dataset {
    /// Resolve a single column name to a validated 0-based core index.
    fn column(&self, name: &str) -> Result<usize> {
        self.inner
            .index_of(name)
            .ok_or_else(|| Error::Other(format!("unknown column name: {name:?}")))
    }

    /// Resolve a conditioning set (a vector of names) to 0-based core indices.
    fn columns(&self, names: &Strings) -> Result<Vec<usize>> {
        (0..names.len())
            .map(|i| self.column(names.elt(i).as_ref()))
            .collect()
    }

    /// Resolve `x`, `y`, and the conditioning names `z` to 0-based indices.
    fn resolve(&self, x: &str, y: &str, z: &Strings) -> Result<(usize, usize, Vec<usize>)> {
        Ok((self.column(x)?, self.column(y)?, self.columns(z)?))
    }
}

/// Run an inner test for `x ⟂ y | z` (names) against `data`, returning the
/// uniform result list. Shared by every test handle's `run_test`.
fn run_inner(test: &dyn CITest, data: &Dataset, x: &str, y: &str, z: &Strings) -> Result<Robj> {
    let (xi, yi, zi) = data.resolve(x, y, z)?;
    let result: CoreResult = test.test(&data.inner, xi, yi, &zi).map_err(to_r_error)?;
    Ok(result_to_list(&result))
}

/// Decide independence for `x ⟂ y | z` at `significance_level` using the inner
/// test's declared rule. Shared by every test handle's `is_independent`.
fn is_independent_inner(
    test: &dyn CITest,
    data: &Dataset,
    x: &str,
    y: &str,
    z: &Strings,
    significance_level: f64,
) -> Result<bool> {
    let (xi, yi, zi) = data.resolve(x, y, z)?;
    test.is_independent(&data.inner, xi, yi, &zi, significance_level)
        .map_err(to_r_error)
}

/// Generate a data-bound extendr test handle for a concrete [`CITest`].
///
/// Each generated struct stores the shared [`Dataset`] handle plus a constructed
/// inner test, and exposes the uniform `run_test` / `is_independent` surface.
/// The struct definition carries `#[extendr]` (so it gains `Robj` conversions)
/// and the impl block carries `#[extendr]` (so its methods become R-callable).
/// The constructor (`new`) takes a `Dataset` external pointer plus the test's
/// configuration arguments, mapped onto the concrete test via `$build`.
macro_rules! ci_test_handle {
    (
        $wrapper:ident,
        $core:ty,
        new($($arg:ident : $arg_ty:ty),* $(,)?) $build:block
    ) => {
        #[doc = concat!("Data-bound `", stringify!($wrapper), "` test handle.")]
        #[extendr]
        struct $wrapper {
            data: Dataset,
            inner: $core,
        }

        #[extendr]
        impl $wrapper {
            /// Construct the handle bound to `data` with the given config.
            fn new(data: &Dataset $(, $arg: $arg_ty)*) -> Self {
                let data = data.clone();
                let inner = $build;
                Self { data, inner }
            }

            /// Run the test for `x ⟂ y | z` (column names; `z` a character
            /// vector), returning `list(statistic, p_value, dof, effect_size)`.
            fn run_test(&self, x: &str, y: &str, z: Strings) -> Result<Robj> {
                run_inner(&self.inner, &self.data, x, y, &z)
            }

            /// Decide independence at `significance_level` using the test's rule.
            fn is_independent(
                &self,
                x: &str,
                y: &str,
                z: Strings,
                significance_level: f64,
            ) -> Result<bool> {
                is_independent_inner(&self.inner, &self.data, x, y, &z, significance_level)
            }
        }
    };
}

ci_test_handle!(
    ChiSquared,
    citest::ci_tests::ChiSquared,
    new(yates: bool) { citest::ci_tests::ChiSquared { yates } }
);

ci_test_handle!(
    LogLikelihood,
    citest::ci_tests::LogLikelihood,
    new(yates: bool) { citest::ci_tests::LogLikelihood { yates } }
);

ci_test_handle!(
    CressieRead,
    citest::ci_tests::CressieRead,
    new(yates: bool) { citest::ci_tests::CressieRead { yates } }
);

ci_test_handle!(
    FreemanTukey,
    citest::ci_tests::FreemanTukey,
    new(yates: bool) { citest::ci_tests::FreemanTukey { yates } }
);

ci_test_handle!(
    ModifiedLikelihood,
    citest::ci_tests::ModifiedLikelihood,
    new(yates: bool) { citest::ci_tests::ModifiedLikelihood { yates } }
);

ci_test_handle!(
    PearsonCorrelation,
    citest::ci_tests::PearsonCorrelation,
    new() { citest::ci_tests::PearsonCorrelation::new() }
);

ci_test_handle!(
    FisherZ,
    citest::ci_tests::FisherZ,
    new() { citest::ci_tests::FisherZ::new() }
);

ci_test_handle!(
    PearsonEquivalence,
    citest::ci_tests::PearsonEquivalence,
    new(delta_threshold: f64) {
        citest::ci_tests::PearsonEquivalence { delta_threshold }
    }
);

extendr_module! {
    mod citest;
    impl Dataset;
    impl ChiSquared;
    impl LogLikelihood;
    impl CressieRead;
    impl FreemanTukey;
    impl ModifiedLikelihood;
    impl PearsonCorrelation;
    impl FisherZ;
    impl PearsonEquivalence;
}
