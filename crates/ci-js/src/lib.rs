//! JavaScript / WebAssembly bindings for the data-bound conditional-independence
//! testing core.
//!
//! The surface mirrors `ci_core`'s redesigned contract: a [`Dataset`] is built
//! once from named, typed columns, and each test is a *data-bound* class that
//! takes the dataset (plus its own configuration) in its constructor and then
//! answers `runTest(x, y, z)` / `isIndependent(...)` queries.
//!
//! Each test constructor accepts either a [`Dataset`] instance (cheaply shared
//! across tests without re-copying) or a raw columns object `{ name: { kind,
//! values } }` for one-shot usage.
//!
//! ```js
//! import { Dataset, ChiSquared, PearsonCorrelation, FisherZ } from "../pkg/ci_js.js";
//!
//! // Option A: shared Dataset (preferred when reusing across tests)
//! const data = new Dataset({
//!   A: { kind: "discrete",   values: [0, 1, 0, 1] },
//!   B: { kind: "discrete",   values: [1, 0, 1, 0] },
//! });
//! const chi = new ChiSquared(data);            // optional config: { yates }
//! const r   = chi.runTest("A", "B", []);       // { statistic, pValue, dof, effectSize }
//! chi.isIndependent("A", "B", [], 0.05);       // boolean
//!
//! // Option B: inline raw object (one-shot; columns cross the wasm boundary once)
//! const chi2 = new ChiSquared({
//!   A: { kind: "discrete", values: [0, 1, 0, 1] },
//!   B: { kind: "discrete", values: [1, 0, 1, 0] },
//! });
//!
//! // FisherZ (continuous):
//! const fz = new FisherZ(data);
//! fz.runTest("X", "Y", ["Z"]);                 // { statistic, pValue, dof: null, effectSize }
//! ```
//!
//! Discrete column `values` may be a `number[]`, a `Float64Array`, **or** a
//! `string[]` (first-seen string-to-code factorization). Continuous columns
//! require numeric values. Core [`CiError`]s and Rust panics surface as thrown
//! JS `Error`s.

// This crate is a thin wasm-bindgen FFI layer: every fallible function returns
// the same `Result<_, JsValue>` (a thrown JS `Error`) whose failure modes —
// unknown column, wrong column kind, dimension mismatch, degenerate data — are
// documented once on the types/module rather than repeated per method.
#![allow(clippy::missing_errors_doc)]

use std::rc::Rc;

use ci_core::dataset::{ColumnKind, Dataset as CoreDataset};
use ci_core::error::CiError as CoreError;
use ci_core::strategy::{CITest, CiResult as CoreResult, DataType, IndependenceRule, TestMeta};
use serde::Deserialize;
use wasm_bindgen::convert::TryFromJsValue;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
export type ColumnKind = "discrete" | "continuous";
export interface ColumnSpec { kind: ColumnKind; values: number[] | Float64Array | string[]; }
export type ColumnsObject = Record<string, ColumnSpec>;
export type DataInput = Dataset | ColumnsObject;
export interface DiscreteOptions { yates?: boolean; }
export interface EquivalenceOptions { deltaThreshold?: number; }
"#;

/// Install the panic hook so Rust panics surface as readable JS errors. Called
/// automatically the first time a [`Dataset`] is constructed; also exported for
/// callers that want to install it eagerly.
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Map a core [`CoreError`] onto a thrown JS `Error`.
fn to_js_error(err: &CoreError) -> JsValue {
    js_sys::Error::new(&err.to_string()).into()
}

/// Build a JS `Error` from an arbitrary message.
fn js_error(msg: &str) -> JsValue {
    js_sys::Error::new(msg).into()
}

// ---------------------------------------------------------------------------
// Column / dataset deserialization
// ---------------------------------------------------------------------------

/// Parse a kind string into a [`ColumnKind`].
fn parse_kind(kind: &str) -> Result<ColumnKind, JsValue> {
    match kind {
        "discrete" => Ok(ColumnKind::Discrete),
        "continuous" => Ok(ColumnKind::Continuous),
        other => Err(js_error(&format!(
            "column kind must be \"discrete\" or \"continuous\", got {other:?}"
        ))),
    }
}

/// Read one column's `values` array: a `Float64Array`, an array of numbers, or
/// (for discrete columns) an array of strings factorized to first-seen codes
/// (the core re-factorizes anyway, so codes only need to be value-distinct).
fn extract_values(name: &str, kind: ColumnKind, values: &JsValue) -> Result<Vec<f64>, JsValue> {
    if let Some(arr) = values.dyn_ref::<js_sys::Float64Array>() {
        return Ok(arr.to_vec());
    }
    let Some(arr) = values.dyn_ref::<js_sys::Array>() else {
        return Err(js_error(&format!(
            "column {name:?}: values must be an array or Float64Array"
        )));
    };
    let len = arr.length() as usize;
    let mut out = Vec::with_capacity(len);
    // Decide numeric vs string mode from the first element.
    let string_mode = len > 0 && arr.get(0).as_string().is_some();
    if string_mode {
        if kind != ColumnKind::Discrete {
            return Err(js_error(&format!(
                "column {name:?}: continuous column values must be numeric"
            )));
        }
        let mut lookup: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for (i, v) in arr.iter().enumerate() {
            let Some(s) = v.as_string() else {
                return Err(js_error(&format!(
                    "column {name:?}: mixed string/number values at index {i}"
                )));
            };
            #[allow(clippy::cast_precision_loss)]
            let next = lookup.len() as f64;
            out.push(*lookup.entry(s).or_insert(next));
        }
    } else {
        for (i, v) in arr.iter().enumerate() {
            let Some(num) = v.as_f64() else {
                let hint = if v.as_string().is_some() {
                    "mixed string/number values"
                } else {
                    "values must be finite numbers (no null/undefined)"
                };
                return Err(js_error(&format!("column {name:?}: {hint} at index {i}")));
            };
            out.push(num);
        }
    }
    Ok(out)
}

/// Build a core [`CoreDataset`] from a `{ name: { kind, values } }` JS object.
///
/// Columns are read via `Object.entries`, which preserves the object's
/// (insertion) key order, so the resulting column indices match the order the
/// caller wrote — matching the Python binding. (Every test refers to columns by
/// name, so ordering does not affect results, only `indexOf`.)
fn dataset_from_value(value: &JsValue) -> Result<CoreDataset, JsValue> {
    if !value.is_object() {
        return Err(js_error(
            "expected a columns object { name: { kind, values } }",
        ));
    }
    let obj: &js_sys::Object = value.unchecked_ref();
    let entries = js_sys::Object::entries(obj);
    let mut cols: Vec<(String, ColumnKind, Vec<f64>)> =
        Vec::with_capacity(entries.length() as usize);
    for entry in entries.iter() {
        // Each `entry` is a `[name, { kind, values }]` pair.
        let pair: js_sys::Array = entry.unchecked_into();
        let name = pair
            .get(0)
            .as_string()
            .ok_or_else(|| js_error("column names must be strings"))?;
        let spec = pair.get(1);
        let field = |key: &str| {
            js_sys::Reflect::get(&spec, &JsValue::from_str(key))
                .map_err(|_| js_error(&format!("column {name:?} must be {{ kind, values }}")))
        };
        let kind_str = field("kind")?
            .as_string()
            .ok_or_else(|| js_error(&format!("column {name:?}: kind must be a string")))?;
        let kind = parse_kind(&kind_str)?;
        let values = extract_values(&name, kind, &field("values")?)?;
        cols.push((name, kind, values));
    }
    CoreDataset::from_columns(cols).map_err(|e| to_js_error(&e))
}

// ---------------------------------------------------------------------------
// Dataset wasm class
// ---------------------------------------------------------------------------

/// A named, typed, immutable table of columns shared by the test classes.
///
/// Construct from a JS object `{ name: { kind, values } }` where `kind` is
/// `"discrete"` or `"continuous"` and `values` is a `number[]`, `Float64Array`,
/// or (for discrete columns) a `string[]` (factorized first-seen). Discrete
/// columns are factorized into integer codes inside the core. Reuse one
/// `Dataset` across several tests to avoid re-copying the column arrays across
/// the wasm boundary.
#[wasm_bindgen]
#[derive(Clone)]
pub struct Dataset {
    inner: Rc<CoreDataset>,
}

#[wasm_bindgen]
impl Dataset {
    /// Build a dataset from a JS object `{ name: { kind, values } }`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        #[wasm_bindgen(unchecked_param_type = "ColumnsObject")] columns: &JsValue,
    ) -> Result<Dataset, JsValue> {
        console_error_panic_hook::set_once();
        Ok(Dataset {
            inner: Rc::new(dataset_from_value(columns)?),
        })
    }

    /// Number of rows (observations).
    #[wasm_bindgen(js_name = nRows)]
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// Number of columns.
    #[wasm_bindgen(js_name = nCols)]
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    /// Index of the column named `name`, or `undefined` if it is not present.
    #[wasm_bindgen(js_name = indexOf)]
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.inner.index_of(name)
    }

    /// Internal: return a fresh handle sharing this dataset (used by the test
    /// constructors to accept a `Dataset` without consuming the caller's
    /// object). Stable marker for instance detection; not part of the public
    /// API surface.
    #[wasm_bindgen(js_name = _cloneHandle)]
    #[must_use]
    pub fn clone_handle(&self) -> Dataset {
        self.clone()
    }
}

/// Resolve a constructor `data` argument: a `Dataset` instance (detected by
/// its `_cloneHandle` marker; we consume a *fresh clone*, never the caller's
/// object) or a raw `{ name: { kind, values } }` columns object.
fn coerce_dataset(value: &JsValue) -> Result<Rc<CoreDataset>, JsValue> {
    let marker = js_sys::Reflect::get(value, &JsValue::from_str("_cloneHandle"))
        .unwrap_or(JsValue::UNDEFINED);
    if let Some(f) = marker.dyn_ref::<js_sys::Function>() {
        let cloned = f.call0(value)?;
        let ds =
            Dataset::try_from_js_value(cloned).map_err(|_| js_error("invalid Dataset handle"))?;
        return Ok(ds.inner);
    }
    Ok(Rc::new(dataset_from_value(value)?))
}

/// Resolve a single column reference (a name `string`) to a validated column
/// index within `data`.
fn resolve_column(data: &CoreDataset, name: &str) -> Result<usize, JsValue> {
    data.index_of(name)
        .ok_or_else(|| js_error(&format!("unknown column name: {name:?}")))
}

/// Resolve a conditioning-set argument: a JS array of column names.
fn resolve_z(data: &CoreDataset, z: &[String]) -> Result<Vec<usize>, JsValue> {
    z.iter().map(|name| resolve_column(data, name)).collect()
}

/// Resolve the `x`, `y`, and `z` column references of a query in one step,
/// short-circuiting on the first unknown name (x, then y, then z).
fn resolve_xyz(
    data: &CoreDataset,
    x: &str,
    y: &str,
    z: &[String],
) -> Result<(usize, usize, Vec<usize>), JsValue> {
    Ok((
        resolve_column(data, x)?,
        resolve_column(data, y)?,
        resolve_z(data, z)?,
    ))
}

/// Convert a core [`CoreResult`] into a plain JS object
/// `{ statistic, pValue, dof, effectSize }` (`null` for `None`).
#[allow(
    clippy::cast_precision_loss,
    reason = "dof is a small count; it never approaches f64's 2^53 exact-integer limit"
)]
fn result_to_js(result: &CoreResult) -> JsValue {
    let obj = js_sys::Object::new();
    let set = |key: &str, value: JsValue| {
        // Setting a property on a fresh Object never fails.
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(key), &value);
    };
    set(
        "statistic",
        result.statistic.map_or(JsValue::NULL, JsValue::from_f64),
    );
    set("pValue", JsValue::from_f64(result.p_value));
    set(
        "dof",
        result
            .dof
            .map_or(JsValue::NULL, |d| JsValue::from_f64(d as f64)),
    );
    set(
        "effectSize",
        result.effect_size.map_or(JsValue::NULL, JsValue::from_f64),
    );
    obj.into()
}

/// Convert a [`TestMeta`] into a plain JS object
/// `{ name, dataTypes, symmetric, rule }`.
fn meta_to_js(meta: &TestMeta) -> JsValue {
    let obj = js_sys::Object::new();
    let set = |key: &str, value: JsValue| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(key), &value);
    };
    set("name", JsValue::from_str(meta.name));
    let types = js_sys::Array::new();
    for t in meta.data_types {
        let s = match t {
            DataType::Discrete => "discrete",
            DataType::Continuous => "continuous",
        };
        types.push(&JsValue::from_str(s));
    }
    set("dataTypes", types.into());
    set("symmetric", JsValue::from_bool(meta.symmetric));
    set(
        "rule",
        JsValue::from_str(match meta.rule {
            IndependenceRule::PValueGe => "p_value_ge",
            IndependenceRule::PValueLt => "p_value_lt",
        }),
    );
    obj.into()
}

/// Generate a data-bound wasm test class for a concrete [`CITest`].
///
/// Each generated class stores the shared [`CoreDataset`] plus a constructed
/// inner test, and exposes the uniform `runTest` / `isIndependent` / `meta`
/// surface. The constructor accepts a [`Dataset`] instance *or* a raw columns
/// object `{ name: { kind, values } }`; passing a shared `Dataset` avoids
/// re-copying the column arrays across the wasm boundary when the same data is
/// used by multiple tests.
macro_rules! ci_test_class {
    (
        $js_name:literal,
        $wrapper:ident,
        $core:ty,
        $config:ident,
        $cfg:ident,
        $config_ts:literal,
        $allowed_keys:expr,
        $build:expr
    ) => {
        #[doc = concat!("Data-bound `", $js_name, "` conditional-independence test.")]
        #[wasm_bindgen(js_name = $js_name)]
        pub struct $wrapper {
            data: Rc<CoreDataset>,
            inner: $core,
        }

        #[wasm_bindgen(js_class = $js_name)]
        impl $wrapper {
            /// Construct the test bound to `data` (a `Dataset` instance or a raw
            /// columns object `{ name: { kind, values } }`) with an optional
            /// `config` object. Pass a shared `Dataset` to avoid re-copying the
            /// column arrays across the wasm boundary when several tests share the
            /// same data.
            #[wasm_bindgen(constructor)]
            pub fn new(
                #[wasm_bindgen(unchecked_param_type = "DataInput")] data: &JsValue,
                #[wasm_bindgen(unchecked_param_type = $config_ts)] config: Option<js_sys::Object>,
            ) -> Result<$wrapper, JsValue> {
                console_error_panic_hook::set_once();
                let data = coerce_dataset(data)?;
                // serde-wasm-bindgen reads only the *known* fields off a JS
                // object, so unknown keys would be silently ignored; reject
                // them explicitly (config typos must not pass unnoticed).
                let allowed: &[&str] = $allowed_keys;
                if let Some(ref obj) = config {
                    for key in js_sys::Object::keys(obj).iter() {
                        let key = key.as_string().unwrap_or_default();
                        if !allowed.contains(&key.as_str()) {
                            return Err(js_error(&format!(
                                "invalid config object: unknown key {key:?} (allowed: {allowed:?})"
                            )));
                        }
                    }
                }
                let $cfg: $config = match config {
                    Some(obj) => serde_wasm_bindgen::from_value(obj.into())
                        .map_err(|e| js_error(&format!("invalid config object: {e}")))?,
                    None => $config::default(),
                };
                let inner = $build;
                Ok($wrapper { data, inner })
            }

            /// Run the test for `x ⟂ y | z`, returning a plain object
            /// `{ statistic, pValue, dof, effectSize }` (`null` for absent
            /// fields). `x`/`y` are column names; `z` is an array of names.
            #[wasm_bindgen(js_name = runTest)]
            pub fn run_test(&self, x: &str, y: &str, z: Vec<String>) -> Result<JsValue, JsValue> {
                let (xi, yi, zi) = resolve_xyz(&self.data, x, y, &z)?;
                self.inner
                    .test(&self.data, xi, yi, &zi)
                    .map(|r| result_to_js(&r))
                    .map_err(|e| to_js_error(&e))
            }

            /// Decide independence at `significanceLevel` (default `0.05` when
            /// omitted, matching the Python binding) using the test's rule.
            /// The core rejects a non-finite `significanceLevel`.
            #[wasm_bindgen(js_name = isIndependent)]
            pub fn is_independent(
                &self,
                x: &str,
                y: &str,
                z: Vec<String>,
                significance_level: Option<f64>,
            ) -> Result<bool, JsValue> {
                let significance_level = significance_level.unwrap_or(0.05);
                let (xi, yi, zi) = resolve_xyz(&self.data, x, y, &z)?;
                self.inner
                    .is_independent(&self.data, xi, yi, &zi, significance_level)
                    .map_err(|e| to_js_error(&e))
            }

            /// Static metadata: `{ name, dataTypes, symmetric, rule }`.
            #[wasm_bindgen]
            #[must_use]
            pub fn meta(&self) -> JsValue {
                meta_to_js(&self.inner.meta())
            }
        }
    };
}

/// Config for the discrete power-divergence tests: `{ yates }` (default `true`).
#[derive(Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct DiscreteConfig {
    yates: bool,
}

impl Default for DiscreteConfig {
    fn default() -> Self {
        // Yates' continuity correction on by default (scipy/pgmpy convention).
        Self { yates: true }
    }
}

/// Config for [`PearsonCorrelation`] and [`FisherZ`]: no options.
#[derive(Deserialize, Default)]
#[serde(default)]
struct EmptyConfig {}

/// Config for [`PearsonEquivalence`]: `{ deltaThreshold }` (default `0.1`).
#[derive(Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct EquivalenceConfig {
    delta_threshold: f64,
}

impl Default for EquivalenceConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 0.1,
        }
    }
}

ci_test_class!(
    "ChiSquared",
    ChiSquared,
    ci_core::ci_tests::ChiSquared,
    DiscreteConfig,
    cfg,
    "DiscreteOptions | undefined",
    &["yates"],
    ci_core::ci_tests::ChiSquared { yates: cfg.yates }
);

ci_test_class!(
    "LogLikelihood",
    LogLikelihood,
    ci_core::ci_tests::LogLikelihood,
    DiscreteConfig,
    cfg,
    "DiscreteOptions | undefined",
    &["yates"],
    ci_core::ci_tests::LogLikelihood { yates: cfg.yates }
);

ci_test_class!(
    "CressieRead",
    CressieRead,
    ci_core::ci_tests::CressieRead,
    DiscreteConfig,
    cfg,
    "DiscreteOptions | undefined",
    &["yates"],
    ci_core::ci_tests::CressieRead { yates: cfg.yates }
);

ci_test_class!(
    "FreemanTukey",
    FreemanTukey,
    ci_core::ci_tests::FreemanTukey,
    DiscreteConfig,
    cfg,
    "DiscreteOptions | undefined",
    &["yates"],
    ci_core::ci_tests::FreemanTukey { yates: cfg.yates }
);

ci_test_class!(
    "ModifiedLikelihood",
    ModifiedLikelihood,
    ci_core::ci_tests::ModifiedLikelihood,
    DiscreteConfig,
    cfg,
    "DiscreteOptions | undefined",
    &["yates"],
    ci_core::ci_tests::ModifiedLikelihood { yates: cfg.yates }
);

ci_test_class!(
    "PearsonCorrelation",
    PearsonCorrelation,
    ci_core::ci_tests::PearsonCorrelation,
    EmptyConfig,
    _cfg,
    "Record<string, never> | undefined",
    &[],
    ci_core::ci_tests::PearsonCorrelation::new()
);

ci_test_class!(
    "PearsonEquivalence",
    PearsonEquivalence,
    ci_core::ci_tests::PearsonEquivalence,
    EquivalenceConfig,
    cfg,
    "EquivalenceOptions | undefined",
    &["deltaThreshold"],
    ci_core::ci_tests::PearsonEquivalence {
        delta_threshold: cfg.delta_threshold
    }
);

ci_test_class!(
    "FisherZ",
    FisherZ,
    ci_core::ci_tests::FisherZ,
    EmptyConfig,
    _cfg,
    "Record<string, never> | undefined",
    &[],
    ci_core::ci_tests::FisherZ::new()
);
