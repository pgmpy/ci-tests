//! Core conditional independence tests for causal discovery.
//!
//! Provides a collection of statistical tests for determining whether two
//! variables X and Y are independent given a conditioning set Z (X ⊥ Y | Z).
//!
//! # Tests
//!
//! | Test | Data type | Module |
//! |------|-----------|--------|
//! | Chi-squared | Discrete | [`ci_tests::ChiSquared`] |
//! | Log-likelihood (G-test) | Discrete | [`ci_tests::LogLikelihood`] |
//! | Cressie-Read | Discrete | [`ci_tests::CressieRead`] |
//! | Freeman-Tukey | Discrete | [`ci_tests::FreemanTukey`] |
//! | Modified log-likelihood | Discrete | [`ci_tests::ModifiedLikelihood`] |
//! | Pearson correlation | Continuous | [`ci_tests::PearsonCorrelation`] |
//! | Pearson equivalence (TOST) | Continuous | [`ci_tests::PearsonEquivalence`] |
//!
//! # Usage
//!
//! All tests implement the [`strategy::CITest`] trait. Construct a test,
//! then call [`strategy::CITest::run_test`] with your data arrays.
//!
//! ```rust
//! use ci_core::ci_tests::ChiSquared;
//! use ci_core::strategy::CITest;
//! use ndarray::{array, Array2};
//!
//! let test = ChiSquared { boolean: false, significance_level: 0.05 };
//! let x = array![1., 1., 2., 2.];
//! let y = array![1., 2., 1., 2.];
//! let z = Array2::zeros((0, 0)); // unconditional
//! let result = test.run_test(x, y, z).unwrap();
//! ```

pub mod ci_tests;
pub mod strategy;
pub mod utils;
