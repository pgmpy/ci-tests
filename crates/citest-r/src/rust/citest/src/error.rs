//! Error type for the conditional-independence core.

use thiserror::Error;

/// Errors that can arise when constructing a [`crate::dataset::Dataset`] or
/// running a conditional-independence test.
#[derive(Debug, Error)]
pub enum CiError {
    /// Columns did not share a common length, or two columns shared a name
    /// (names must be unique).
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// The data is degenerate or unsupported for the requested test (e.g. a
    /// column with zero variance for a correlation test, too few rows, or more
    /// continuous columns than the Gaussian cache supports).
    #[error("degenerate data: {0}")]
    DegenerateData(String),

    /// A numerical routine failed (e.g. distribution construction, or a
    /// rank-deficient conditioning set in the partial-correlation solve).
    #[error("numeric error: {0}")]
    Numeric(String),

    /// A column reference (name or index) did not resolve to any column in
    /// the dataset.
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    /// A column was used with a test that expects a different kind (e.g. a
    /// continuous column passed to a discrete test).
    #[error("wrong column kind: {0}")]
    WrongColumnKind(String),

    /// A column contained NaN / missing values at `Dataset` construction.
    #[error("missing data: {0}")]
    MissingData(String),

    /// The (x, y, z) query itself is malformed (x == y, x/y inside z, or
    /// duplicate entries in z).
    #[error("invalid query: {0}")]
    InvalidQuery(String),

    /// A test was configured with a value outside its permitted range. This is
    /// a fault in the caller's configuration, not in the data, and is reported
    /// when the test is constructed rather than on every query.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}
