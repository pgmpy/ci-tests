//! Core conditional-independence tests for causal discovery.
//!
//! Provides statistical tests for whether two variables X and Y are independent
//! given a conditioning set Z (X ⊥ Y | Z). Data is held in a [`dataset::Dataset`]
//! built once from named, typed columns; tests refer to variables by column
//! index and return a numeric [`strategy::CiResult`].
//!
//! # Tests
//!
//! | Test | Data type | Module |
//! |------|-----------|--------|
//! | Chi-squared (λ = 1) | Discrete | [`ci_tests::ChiSquared`] |
//! | Log-likelihood / G-test (λ = 0) | Discrete | [`ci_tests::LogLikelihood`] |
//! | Cressie-Read (λ = 2/3) | Discrete | [`ci_tests::CressieRead`] |
//! | Freeman-Tukey (λ = −1/2) | Discrete | [`ci_tests::FreemanTukey`] |
//! | Modified log-likelihood (λ = −1) | Discrete | [`ci_tests::ModifiedLikelihood`] |
//! | Pearson correlation | Continuous | [`ci_tests::PearsonCorrelation`] |
//! | Fisher-z | Continuous | [`ci_tests::FisherZ`] |
//! | Pearson equivalence (TOST) | Continuous | [`ci_tests::PearsonEquivalence`] |
//!
//! [`registry`] enumerates all eight and constructs a default-configured one by
//! name.
//!
//! # Usage
//!
//! ```rust
//! use citest::ci_tests::ChiSquared;
//! use citest::dataset::{ColumnKind, Dataset};
//! use citest::strategy::CITest;
//!
//! let data = Dataset::from_columns(vec![
//!     ("x".into(), ColumnKind::Discrete, vec![1., 1., 2., 2.]),
//!     ("y".into(), ColumnKind::Discrete, vec![1., 2., 1., 2.]),
//! ])
//! .unwrap();
//!
//! let x = data.index_of("x").unwrap();
//! let y = data.index_of("y").unwrap();
//! let result = ChiSquared::new().test(&data, x, y, &[]).unwrap();
//! assert!(result.p_value > 0.0);
//! ```

pub mod ci_tests;
pub mod dataset;
pub mod discrete;
pub mod error;
pub(crate) mod gram;
pub mod registry;
pub mod strategy;

pub use dataset::{ColumnKind, Dataset};
pub use error::CiError;
pub use registry::{all_metas, make_default};
pub use strategy::{validate_query, CITest, CiResult, DataType, IndependenceRule, TestMeta};
