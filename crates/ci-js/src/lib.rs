//! JavaScript / WebAssembly bindings for the data-bound conditional-independence
//! testing core.
//!
//! The surface mirrors `ci_core`'s redesigned contract: a [`Dataset`] is built
//! once from named, typed columns, and each test is a *data-bound* class that
//! takes the dataset (plus its own configuration) in its constructor and then
//! answers `runTest(x, y, z)` / `isIndependent(...)` queries.
//!
//! ```js
//! import { Dataset, ChiSquared, PearsonEquivalence } from "../pkg/ci_js.js";
//!
//! const data = new Dataset({
//!   A: { kind: "discrete",   values: [0, 1, 0, 1] },
//!   B: { kind: "discrete",   values: [1, 0, 1, 0] },
//! });
//! const chi = new ChiSquared(data);            // optional config: { yates }
//! const r = chi.runTest("A", "B", []);         // { statistic, pValue, dof, effectSize }
//! chi.isIndependent("A", "B", [], 0.05);       // boolean
//! ```
//!
//! A [`Dataset`] is built once from a JS object `{ name: { kind, values } }`
//! (deserialized with `serde-wasm-bindgen`), and the test classes take that
//! `Dataset`, so the column arrays cross the JS↔wasm boundary only once even when
//! several tests share the same data. Core [`CiError`]s and Rust panics surface
//! as thrown JS `Error`s.

// This crate is a thin wasm-bindgen FFI layer: every fallible function returns
// the same `Result<_, JsValue>` (a thrown JS `Error`) whose failure modes —
// unknown column, wrong column kind, dimension mismatch, degenerate data — are
// documented once on the types/module rather than repeated per method.
#![allow(clippy::missing_errors_doc)]

use std::rc::Rc;

use ci_core::dataset::{ColumnKind, Dataset as CoreDataset};
use ci_core::error::CiError as CoreError;
use ci_core::strategy::{
    CITest, CiResult as CoreResult, DataType, IndependenceRule, TestMeta,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

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

/// A single column as supplied from JS: `{ kind, values }`.
///
/// `values` accepts either a `number[]` or a `Float64Array` (both deserialize to
/// `Vec<f64>` through `serde-wasm-bindgen`).
#[derive(Deserialize)]
struct ColumnSpec {
    kind: String,
    values: Vec<f64>,
}

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
    let mut cols: Vec<(String, ColumnKind, Vec<f64>)> = Vec::with_capacity(entries.length() as usize);
    for entry in entries.iter() {
        // Each `entry` is a `[name, { kind, values }]` pair.
        let pair: js_sys::Array = entry.unchecked_into();
        let name = pair
            .get(0)
            .as_string()
            .ok_or_else(|| js_error("column names must be strings"))?;
        let col: ColumnSpec = serde_wasm_bindgen::from_value(pair.get(1)).map_err(|e| {
            js_error(&format!(
                "column {name:?} must be {{ kind, values }}: {e}"
            ))
        })?;
        let kind = parse_kind(&col.kind)?;
        cols.push((name, kind, col.values));
    }
    CoreDataset::from_columns(cols).map_err(|e| to_js_error(&e))
}

// ---------------------------------------------------------------------------
// Dataset wasm class
// ---------------------------------------------------------------------------

/// A named, typed, immutable table of columns shared by the test classes.
///
/// Construct from a JS object `{ name: { kind, values } }` where `kind` is
/// `"discrete"` or `"continuous"` and `values` is a `number[]` or
/// `Float64Array`. Discrete columns are factorized into integer codes inside the
/// core. Reuse one `Dataset` across several tests to avoid re-copying the column
/// arrays across the wasm boundary.
#[wasm_bindgen]
#[derive(Clone)]
pub struct Dataset {
    inner: Rc<CoreDataset>,
}

#[wasm_bindgen]
impl Dataset {
    /// Build a dataset from a JS object `{ name: { kind, values } }`.
    #[wasm_bindgen(constructor)]
    pub fn new(columns: &JsValue) -> Result<Dataset, JsValue> {
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
/// surface. The constructor accepts the dataset (a [`Dataset`] object or a raw
/// columns object) and an optional config object; `$build` maps that config onto
/// the concrete test's configuration.
macro_rules! ci_test_class {
    (
        $js_name:literal,
        $wrapper:ident,
        $core:ty,
        $config:ident,
        $cfg:ident,
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
            /// Construct the test bound to `data` (a `Dataset`) with an optional
            /// `config` object. Build the dataset once with `new Dataset({...})`;
            /// reusing it across tests avoids re-copying the columns across the
            /// wasm boundary.
            #[wasm_bindgen(constructor)]
            pub fn new(data: &Dataset, config: Option<js_sys::Object>) -> Result<$wrapper, JsValue> {
                console_error_panic_hook::set_once();
                let data = Rc::clone(&data.inner);
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
            pub fn run_test(
                &self,
                x: &str,
                y: &str,
                z: Vec<String>,
            ) -> Result<JsValue, JsValue> {
                let xi = resolve_column(&self.data, x)?;
                let yi = resolve_column(&self.data, y)?;
                let zi = resolve_z(&self.data, &z)?;
                self.inner
                    .test(&self.data, xi, yi, &zi)
                    .map(|r| result_to_js(&r))
                    .map_err(|e| to_js_error(&e))
            }

            /// Decide independence at `significanceLevel` using the test's rule.
            #[wasm_bindgen(js_name = isIndependent)]
            pub fn is_independent(
                &self,
                x: &str,
                y: &str,
                z: Vec<String>,
                significance_level: f64,
            ) -> Result<bool, JsValue> {
                let xi = resolve_column(&self.data, x)?;
                let yi = resolve_column(&self.data, y)?;
                let zi = resolve_z(&self.data, &z)?;
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

/// Config for [`PearsonCorrelation`]: no options.
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
        Self { delta_threshold: 0.1 }
    }
}

ci_test_class!(
    "ChiSquared",
    ChiSquared,
    ci_core::ci_tests::ChiSquared,
    DiscreteConfig,
    cfg,
    ci_core::ci_tests::ChiSquared { yates: cfg.yates }
);

ci_test_class!(
    "LogLikelihood",
    LogLikelihood,
    ci_core::ci_tests::LogLikelihood,
    DiscreteConfig,
    cfg,
    ci_core::ci_tests::LogLikelihood { yates: cfg.yates }
);

ci_test_class!(
    "CressieRead",
    CressieRead,
    ci_core::ci_tests::CressieRead,
    DiscreteConfig,
    cfg,
    ci_core::ci_tests::CressieRead { yates: cfg.yates }
);

ci_test_class!(
    "FreemanTukey",
    FreemanTukey,
    ci_core::ci_tests::FreemanTukey,
    DiscreteConfig,
    cfg,
    ci_core::ci_tests::FreemanTukey { yates: cfg.yates }
);

ci_test_class!(
    "ModifiedLikelihood",
    ModifiedLikelihood,
    ci_core::ci_tests::ModifiedLikelihood,
    DiscreteConfig,
    cfg,
    ci_core::ci_tests::ModifiedLikelihood { yates: cfg.yates }
);

ci_test_class!(
    "PearsonCorrelation",
    PearsonCorrelation,
    ci_core::ci_tests::PearsonCorrelation,
    EmptyConfig,
    _cfg,
    ci_core::ci_tests::PearsonCorrelation::new()
);

ci_test_class!(
    "PearsonEquivalence",
    PearsonEquivalence,
    ci_core::ci_tests::PearsonEquivalence,
    EquivalenceConfig,
    cfg,
    ci_core::ci_tests::PearsonEquivalence { delta_threshold: cfg.delta_threshold }
);
