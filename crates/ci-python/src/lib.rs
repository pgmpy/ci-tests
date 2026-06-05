//! Python bindings for the conditional independence testing library.
//!
//! Exposes CI test functions to Python via the `pyo3` framework.
//! Each CI test's `run_test` method accepts paired observation vectors and a conditioning
//! matrix, returning a Python object whose shape depends on whether the test runs in
//! boolean or numeric mode.

mod util;
use pyo3_stub_gen::define_stub_info_gatherer;
mod ci_tests_init;

#[pyo3::pymodule]
mod _ci_python {
    use crate::util::test_result_to_pyobj;
    use ci_core::strategy::CITest;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

    use crate::ci_tests_init;

    include!(concat!(env!("OUT_DIR"), "/ci_tests.rs"));

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        ci_tests_init::init(m)
    }
}

define_stub_info_gatherer!(stub_info);
