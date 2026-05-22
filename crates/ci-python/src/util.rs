//! Conversion utilities from core result types to Python objects.

use ci_core::strategy::TestResult;
use pyo3::prelude::*;

/// Converts a [`TestResult`] into a Python object.
///
/// - [`TestResult::Boolean`] — returns a `bool`.
/// - [`TestResult::PValue`] — returns a `(p_value, coefficient)` tuple.
/// - [`TestResult::Statistic`] — returns a `(p_value, statistic, dof)` tuple.
pub fn test_result_to_pyobj(result: &TestResult, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match result {
        TestResult::Boolean(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        TestResult::PValue(p_value, coefficient) => Ok((*p_value, *coefficient)
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        TestResult::Statistic(p_value, statistic, dof) => Ok((*p_value, *statistic, *dof)
            .into_pyobject(py)?
            .into_any()
            .unbind()),
    }
}
