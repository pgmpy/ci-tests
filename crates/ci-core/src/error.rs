//! Error type for the conditional-independence core.

use thiserror::Error;

/// Errors that can arise when constructing a [`crate::dataset::Dataset`] or
/// running a conditional-independence test.
#[derive(Debug, Error)]
pub enum CiError {
    /// Columns (or X/Y/Z vectors) did not share a common length.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// The data is degenerate for the requested test (e.g. a column with zero
    /// variance for a correlation test, or too few rows).
    #[error("degenerate data: {0}")]
    DegenerateData(String),

    /// A numerical routine failed (e.g. distribution construction or least
    /// squares).
    #[error("numeric error: {0}")]
    Numeric(String),

    /// A column name was requested that does not exist in the dataset.
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    /// A column was used with a test that expects a different kind (e.g. a
    /// continuous column passed to a discrete test).
    #[error("wrong column kind: {0}")]
    WrongColumnKind(String),
}
