use ndarray::{Array1, Array2};

/// The outcome of a conditional independence test.
#[derive(Debug, Clone, PartialEq)]
pub enum TestResult {
    PValue(f64, f64),
    Statistic(f64, f64, usize),
    Boolean(bool),
}

/// Data types that a `CITest` can be performed on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CITestDataType {
    Continuous,
    Discrete,
    Mixed,
}

/// Trait defining the interface for conditional independence tests.
///
/// All statistical tests for conditional independence must implement this trait
/// to be compatible with the registry system.
pub trait CITest: Send + Sync {
    /// Runs a conditional independence test on the given data.
    ///
    /// # Errors
    ///
    /// Returns an error if the test computation fails (e.g., invalid input dimensions or numerical issues).
    fn run_test(
        &self,
        x_values: Array1<f64>,
        y_values: Array1<f64>,
        z: Array2<f64>,
    ) -> anyhow::Result<TestResult>;

    /// Data types that a test supports.
    fn data_types(&self) -> &'static [CITestDataType];
}
