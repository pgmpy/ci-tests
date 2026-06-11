//! Python bindings for the data-bound conditional-independence testing core.
//!
//! The surface mirrors `ci_core`'s redesigned contract: a [`Dataset`] is built
//! once from named, typed columns, and each test is a *data-bound* class that
//! takes the dataset (plus its own configuration) in its constructor and then
//! answers `run_test(x, y, z)` / `is_independent(...)` queries.
//!
//! Column references (`x`, `y`, and the entries of `z`) accept either a column
//! name (`str`) or an integer index, resolved against the bound dataset via
//! [`Dataset::index_of`]. Core [`ci_core::error::CiError`]s surface as the
//! Python [`CiError`] exception; no Rust panic is allowed to escape.
//!
//! # Exposed tests
//!
//! Discrete: [`ChiSquared`], [`LogLikelihood`], [`CressieRead`],
//! [`FreemanTukey`], [`ModifiedLikelihood`].
//! Continuous: [`PearsonCorrelation`], [`FisherZ`], [`PearsonEquivalence`].

use std::sync::Arc;

use ci_core::dataset::{ColumnKind, Dataset as CoreDataset};
use ci_core::error::CiError as CoreError;
use ci_core::strategy::{CITest, CiResult as CoreResult, DataType, IndependenceRule, TestMeta};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence, PyString, PyTuple};

pyo3::create_exception!(
    _ci_python,
    CiError,
    pyo3::exceptions::PyException,
    "Raised when the core conditional-independence engine reports an error."
);

/// Map a core [`CoreError`] onto the Python [`CiError`] exception.
fn map_err(err: &CoreError) -> PyErr {
    CiError::new_err(err.to_string())
}

/// Resolve a single column reference (a `str` name or an integer index) to a
/// validated column index within `data`.
fn resolve_column(data: &CoreDataset, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(name) = obj.downcast::<PyString>() {
        let name = name.to_cow()?;
        data.index_of(&name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column name: {name:?}")))
    } else if let Ok(idx) = obj.extract::<isize>() {
        let n_cols = data.n_cols();
        let out_of_range = || {
            PyValueError::new_err(format!(
                "column index {idx} out of range for dataset with {n_cols} columns"
            ))
        };
        // Allow Python-style negative indexing without lossy isize/usize casts.
        let resolved = if idx < 0 {
            let back = usize::try_from(-idx).map_err(|_| out_of_range())?;
            n_cols.checked_sub(back).ok_or_else(out_of_range)?
        } else {
            usize::try_from(idx).map_err(|_| out_of_range())?
        };
        if resolved >= n_cols {
            return Err(out_of_range());
        }
        Ok(resolved)
    } else {
        Err(PyValueError::new_err(
            "column reference must be a str name or an integer index",
        ))
    }
}

/// Resolve a conditioning-set argument: a sequence of names/indices (or `None`).
fn resolve_z(data: &CoreDataset, z: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    let Some(z) = z else { return Ok(Vec::new()) };
    if z.is_none() {
        return Ok(Vec::new());
    }
    // A bare string would iterate character-by-character, which is never intended.
    if z.is_instance_of::<PyString>() {
        return Err(PyValueError::new_err(
            "z must be a sequence of column references, not a single string",
        ));
    }
    let seq = z
        .downcast::<PySequence>()
        .map_err(|_| PyValueError::new_err("z must be a sequence of column names or indices"))?;
    let len = seq.len()?;
    let mut indices = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        indices.push(resolve_column(data, &item)?);
    }
    Ok(indices)
}

/// The numeric outcome of a conditional-independence test.
///
/// Mirrors [`ci_core::strategy::CiResult`]: `statistic`, `dof`, and
/// `effect_size` are `None` when the test does not define them.
#[pyclass(name = "CiResult", module = "ci_python._ci_python", frozen)]
#[derive(Clone)]
pub struct PyCiResult {
    #[pyo3(get)]
    statistic: Option<f64>,
    #[pyo3(get)]
    p_value: f64,
    #[pyo3(get)]
    dof: Option<usize>,
    #[pyo3(get)]
    effect_size: Option<f64>,
}

impl From<CoreResult> for PyCiResult {
    fn from(r: CoreResult) -> Self {
        Self {
            statistic: r.statistic,
            p_value: r.p_value,
            dof: r.dof,
            effect_size: r.effect_size,
        }
    }
}

#[pymethods]
impl PyCiResult {
    fn __repr__(&self) -> String {
        format!(
            "CiResult(statistic={:?}, p_value={}, dof={:?}, effect_size={:?})",
            self.statistic, self.p_value, self.dof, self.effect_size
        )
    }
}

/// A named, typed, immutable table of columns shared by the test classes.
///
/// Build one from a mapping `{name: (kind, values)}` where `kind` is
/// `"discrete"` or `"continuous"` and `values` is a 1-D float64 array, a
/// sequence of numbers, or — for discrete columns — a sequence of strings.
/// Alternatively, pass a `pandas.DataFrame` and kinds are inferred from
/// dtypes. Discrete columns are factorized into integer codes inside the core.
#[pyclass(name = "Dataset", module = "ci_python._ci_python", frozen, subclass)]
#[derive(Clone)]
pub struct PyDataset {
    inner: Arc<CoreDataset>,
}

/// Whether `obj` is a `pandas.DataFrame`, detected structurally (type name +
/// defining module) so pandas is never imported here.
fn is_pandas_dataframe(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let cls = obj.get_type();
    if cls.name()?.to_cow()? != "DataFrame" {
        return Ok(false);
    }
    let module: String = cls.getattr("__module__")?.extract()?;
    Ok(module.starts_with("pandas"))
}

/// Parse a kind string into a [`ColumnKind`].
fn parse_kind(kind: &str) -> PyResult<ColumnKind> {
    match kind {
        "discrete" => Ok(ColumnKind::Discrete),
        "continuous" => Ok(ColumnKind::Continuous),
        other => Err(PyValueError::new_err(format!(
            "column kind must be 'discrete' or 'continuous', got {other:?}"
        ))),
    }
}

/// Extract column values as `Vec<f64>` from a numpy array (fast path), any
/// numeric sequence, or — for discrete columns — a sequence of strings, which
/// is factorized to first-seen integer codes (the core re-factorizes anyway,
/// so codes only need to be value-distinct).
fn extract_values(kind: ColumnKind, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(arr.as_slice()?.to_vec());
    }
    let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    let numeric: PyResult<Vec<f64>> = items.iter().map(PyAnyMethods::extract::<f64>).collect();
    if let Ok(values) = numeric {
        return Ok(values);
    }
    match kind {
        ColumnKind::Discrete => {
            let mut lookup: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            let mut codes = Vec::with_capacity(items.len());
            for item in &items {
                let key: String = item.extract().map_err(|_| {
                    PyValueError::new_err("discrete column values must be numbers or strings")
                })?;
                #[allow(clippy::cast_precision_loss)]
                let next = lookup.len() as f64;
                codes.push(*lookup.entry(key).or_insert(next));
            }
            Ok(codes)
        }
        ColumnKind::Continuous => Err(PyValueError::new_err(
            "continuous column values must be numeric",
        )),
    }
}

#[pymethods]
impl PyDataset {
    /// Build a dataset from a mapping `{name: (kind, values)}`.
    ///
    /// `kind` is `"discrete"` or `"continuous"`; `values` is a float64 array,
    /// a numeric sequence, or — for discrete columns — a sequence of strings
    /// (factorized to first-seen integer codes). Insertion order of the mapping
    /// is preserved as the column order.
    #[new]
    fn new(columns: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cols: Vec<(String, ColumnKind, Vec<f64>)> = Vec::with_capacity(columns.len());
        for (key, value) in columns.iter() {
            let name: String = key
                .extract()
                .map_err(|_| PyValueError::new_err("Dataset column names must be strings"))?;
            let spec = value.downcast::<PyTuple>().map_err(|_| {
                PyValueError::new_err(format!(
                    "column {name:?} must map to a (kind, values) tuple"
                ))
            })?;
            if spec.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "column {name:?} must map to a (kind, values) 2-tuple"
                )));
            }
            let kind = parse_kind(&spec.get_item(0)?.extract::<String>()?)?;
            let values = extract_values(kind, &spec.get_item(1)?)?;
            cols.push((name, kind, values));
        }
        let inner = CoreDataset::from_columns(cols).map_err(|e| map_err(&e))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Number of rows (observations).
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// Number of columns.
    #[getter]
    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    /// Resolve a column name to its integer index, or raise `ValueError`.
    fn index_of(&self, name: &str) -> PyResult<usize> {
        self.inner
            .index_of(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column name: {name:?}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(n_cols={}, n_rows={})",
            self.inner.n_cols(),
            self.inner.n_rows()
        )
    }
}

impl PyDataset {
    /// Resolve a value that is *either* an existing [`PyDataset`], a column
    /// mapping `{name: (kind, values)}`, or a `pandas.DataFrame` into a shared
    /// [`CoreDataset`]. This lets every test constructor accept `Test(data)`,
    /// `Test({...})`, or `Test(df)` uniformly.
    fn coerce(obj: &Bound<'_, PyAny>) -> PyResult<Arc<CoreDataset>> {
        if let Ok(ds) = obj.extract::<PyRef<'_, PyDataset>>() {
            Ok(Arc::clone(&ds.inner))
        } else if let Ok(dict) = obj.downcast::<PyDict>() {
            Ok(PyDataset::new(dict)?.inner)
        } else if is_pandas_dataframe(obj)? {
            // Route through the Python-side `Dataset.from_pandas` so dtype
            // inference lives in one place; pandas stays a soft dependency
            // (this branch only runs when a DataFrame was passed).
            let py = obj.py();
            let dataset_cls = PyModule::import(py, "ci_python")?.getattr("Dataset")?;
            let built = dataset_cls.call_method1("from_pandas", (obj,))?;
            let ds = built.extract::<PyRef<'_, PyDataset>>()?;
            Ok(Arc::clone(&ds.inner))
        } else {
            Err(PyValueError::new_err(
                "expected a Dataset, a {name: (kind, values)} mapping, or a pandas DataFrame",
            ))
        }
    }
}

/// Build the Python dict `{name, data_types, symmetric, rule}` for a meta.
fn meta_to_py(py: Python<'_>, meta: &TestMeta) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", meta.name)?;
    let types: Vec<&str> = meta
        .data_types
        .iter()
        .map(|t| match t {
            DataType::Discrete => "discrete",
            DataType::Continuous => "continuous",
        })
        .collect();
    dict.set_item("data_types", types)?;
    dict.set_item("symmetric", meta.symmetric)?;
    dict.set_item(
        "rule",
        match meta.rule {
            IndependenceRule::PValueGe => "p_value_ge",
            IndependenceRule::PValueLt => "p_value_lt",
        },
    )?;
    Ok(dict.unbind())
}

/// Generate a data-bound `#[pyclass]` wrapper for a concrete [`CITest`].
///
/// Each generated class stores the shared [`CoreDataset`] plus a constructed
/// inner test, and exposes the uniform `run_test` / `is_independent` / `meta`
/// surface. The constructor body (`$ctor`) maps the Python kwargs onto the
/// concrete test's configuration.
macro_rules! ci_test_class {
    (
        $py_name:literal,
        $wrapper:ident,
        $core:path,
        new($($arg:ident : $arg_ty:ty = $default:expr),* $(,)?) $ctor:block
    ) => {
        #[doc = concat!("Data-bound `", $py_name, "` conditional-independence test.")]
        #[pyclass(name = $py_name, module = "ci_python._ci_python", frozen)]
        pub struct $wrapper {
            data: Arc<CoreDataset>,
            inner: $core,
        }

        #[pymethods]
        impl $wrapper {
            #[new]
            #[pyo3(signature = (data $(, $arg = $default)*))]
            fn new(data: &Bound<'_, PyAny> $(, $arg: $arg_ty)*) -> PyResult<Self> {
                let data = PyDataset::coerce(data)?;
                let inner = $ctor;
                Ok(Self { data, inner })
            }

            /// Run the test for `x ⟂ y | z`, returning a [`CiResult`].
            ///
            /// `x` and `y` are column names or indices; `z` is a sequence of
            /// names/indices (default: empty conditioning set).
            #[pyo3(signature = (x, y, z = None))]
            fn run_test(
                &self,
                py: Python<'_>,
                x: &Bound<'_, PyAny>,
                y: &Bound<'_, PyAny>,
                z: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<PyCiResult> {
                let xi = resolve_column(&self.data, x)?;
                let yi = resolve_column(&self.data, y)?;
                let zi = resolve_z(&self.data, z)?;
                let inner = &self.inner;
                let data = &self.data;
                py.allow_threads(|| inner.test(data.as_ref(), xi, yi, &zi))
                    .map(PyCiResult::from)
                    .map_err(|e| map_err(&e))
            }

            /// Decide independence at `significance_level` using the test's rule.
            #[pyo3(signature = (x, y, z = None, significance_level = 0.05))]
            fn is_independent(
                &self,
                py: Python<'_>,
                x: &Bound<'_, PyAny>,
                y: &Bound<'_, PyAny>,
                z: Option<&Bound<'_, PyAny>>,
                significance_level: f64,
            ) -> PyResult<bool> {
                let xi = resolve_column(&self.data, x)?;
                let yi = resolve_column(&self.data, y)?;
                let zi = resolve_z(&self.data, z)?;
                let inner = &self.inner;
                let data = &self.data;
                py.allow_threads(|| {
                    inner.is_independent(data.as_ref(), xi, yi, &zi, significance_level)
                })
                .map_err(|e| map_err(&e))
            }

            /// Static metadata: `{name, data_types, symmetric, rule}`.
            fn meta(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
                meta_to_py(py, &self.inner.meta())
            }

            fn __repr__(&self) -> String {
                concat!($py_name, "(...)").to_string()
            }
        }
    };
}

ci_test_class!(
    "ChiSquared",
    PyChiSquared,
    ci_core::ci_tests::ChiSquared,
    new(yates: bool = true) {
        ci_core::ci_tests::ChiSquared { yates }
    }
);

ci_test_class!(
    "LogLikelihood",
    PyLogLikelihood,
    ci_core::ci_tests::LogLikelihood,
    new(yates: bool = true) {
        ci_core::ci_tests::LogLikelihood { yates }
    }
);

ci_test_class!(
    "CressieRead",
    PyCressieRead,
    ci_core::ci_tests::CressieRead,
    new(yates: bool = true) {
        ci_core::ci_tests::CressieRead { yates }
    }
);

ci_test_class!(
    "FreemanTukey",
    PyFreemanTukey,
    ci_core::ci_tests::FreemanTukey,
    new(yates: bool = true) {
        ci_core::ci_tests::FreemanTukey { yates }
    }
);

ci_test_class!(
    "ModifiedLikelihood",
    PyModifiedLikelihood,
    ci_core::ci_tests::ModifiedLikelihood,
    new(yates: bool = true) {
        ci_core::ci_tests::ModifiedLikelihood { yates }
    }
);

ci_test_class!(
    "PearsonCorrelation",
    PyPearsonCorrelation,
    ci_core::ci_tests::PearsonCorrelation,
    new() {
        ci_core::ci_tests::PearsonCorrelation::new()
    }
);

ci_test_class!(
    "FisherZ",
    PyFisherZ,
    ci_core::ci_tests::FisherZ,
    new() {
        ci_core::ci_tests::FisherZ::new()
    }
);

ci_test_class!(
    "PearsonEquivalence",
    PyPearsonEquivalence,
    ci_core::ci_tests::PearsonEquivalence,
    new(delta_threshold: f64 = 0.1) {
        ci_core::ci_tests::PearsonEquivalence { delta_threshold }
    }
);

/// The native extension module (`ci_python._ci_python`).
#[pymodule]
#[pyo3(name = "_ci_python")]
fn ci_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CiError", m.py().get_type::<CiError>())?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyCiResult>()?;
    m.add_class::<PyChiSquared>()?;
    m.add_class::<PyLogLikelihood>()?;
    m.add_class::<PyCressieRead>()?;
    m.add_class::<PyFreemanTukey>()?;
    m.add_class::<PyModifiedLikelihood>()?;
    m.add_class::<PyPearsonCorrelation>()?;
    m.add_class::<PyFisherZ>()?;
    m.add_class::<PyPearsonEquivalence>()?;
    Ok(())
}
