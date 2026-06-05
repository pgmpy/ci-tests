//! Build script that automatically generates `PyO3` bindings for `ci-core`'s CI tests.
//!
//! At compile time this script recursively scans the `../ci-core/src` directory and
//! parses every `.rs` file to gather every struct that implements the `CITest`
//! trait and their definitions.
//!
//! Currently, the generated code imports all tests from `::ci_core::ci_tests`;
//! therefore, the compilation of `OUT_DIR/ci_tests.rs` fails with an import error
//! if there are CI tests located somewhere else.
//!
//! Only structs with named fields are supported; unnamed (tuple) structs cause
//! a panic. A `CITest` impl whose struct definition cannot be found in
//! `ci-core/src` is also treated as a hard error.
//!
//! Two files are written into `OUT_DIR` for later `include!`ing:
//!   - `ci_tests.rs` – the generated `Py<struct name>` classes and their methods
//!   - `ci_tests_init.rs` – an `init` function that registers every generated
//!     class with a `PyModule`.
//!
//! `lib.rs` `include!`s both `ci_tests.rs` and calls the `init` function so that
//! both `PyO3` *AND* `pyo3_stub_gen` properly register the tests.
//!
//!
//! ////////// Program Flow //////////
//!
//! `main` calls `parse_dir` on `../ci_core/src` with a new `CITestCollector`.
//!
//! `parse_dir` recursively iterates over all files in `../ci_core/src`. If the
//! file is a `.rs` file, its abstract syntax tree (AST) is visited by the
//! `CITestCollector`. `cargo::rerun-if-changed` is emitted for every parsed
//! source file so the bindings are regenerated whenever the underlying Rust
//! sources change.
//!
//! `CITestCollector` uses the visitor pattern to traverse the AST and collect
//! (a) the names of all structs implementing `CITest` (`citest_structs`) and
//! (b) all struct definitions (`struct_defs`).
//!
//! `main` then iterates over all collected `CITest` structs and calls
//! `generate_pyo3_wrapper` with each struct's definition to generate the bindings.
//!
//! `generate_pyo3_wrapper` emits a `Py<struct name>` wrapper class (e.g.
//! `PyChiSquared`) via `quote!`. The wrapper holds the original type in
//! an `inner` field and exposes the following:
//!   - `#[new]` constructor mirroring the struct's named fields
//!   - `#[getter]`/`#[setter]` pair for each field
//!   - `run_test` method that converts `NumPy` arrays to owned `ndarray`s,
//!     passes them to `inner.run_test`, and maps results/errors back to Python.
//!
//! `main` then collects and saves all results from `generate_pyo3_wrapper` to
//! `ci_tests.rs`. It then generates the `init` function and saves it to
//! `ci_tests_init.rs`.
//!
//!
//! ////////// Example: Generated Bindings for `ChiSquared` //////////
//!
//! ```rust
//! #[gen_stub_pyclass]
//! #[pyclass(name = "ChiSquared", module = "ci_python._ci_python")]
//! pub struct PyChiSquared {
//!     inner: ::ci_core::ci_tests::ChiSquared,
//! }
//!
//! #[gen_stub_pymethods]
//! #[pymethods]
//! impl PyChiSquared {
//!     #[new]
//!     pub fn new(boolean: bool, significance_level: f64) -> Self {
//!         Self {
//!             inner: ::ci_core::ci_tests::ChiSquared {
//!                 boolean,
//!                 significance_level,
//!             },
//!         }
//!     }
//!     #[allow(clippy::needless_pass_by_value)]
//!     fn run_test(
//!         &self,
//!         py: Python<'_>,
//!         x_values: PyReadonlyArray1<'_, f64>,
//!         y_values: PyReadonlyArray1<'_, f64>,
//!         z: PyReadonlyArray2<'_, f64>,
//!     ) -> PyResult<Py<PyAny>> {
//!         test_result_to_pyobj(
//!             &self
//!                 .inner
//!                 .run_test(
//!                     x_values.as_array().to_owned(),
//!                     y_values.as_array().to_owned(),
//!                     z.as_array().to_owned(),
//!                 )
//!                 .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
//!             py,
//!         )
//!     }
//!     #[getter]
//!     pub fn boolean(&self) -> bool {
//!         #[allow(clippy::clone_on_copy)]
//!         self.inner.boolean.clone()
//!     }
//!     #[setter]
//!     pub fn set_boolean(&mut self, boolean: bool) {
//!         self.inner.boolean = boolean;
//!     }
//!     #[getter]
//!     pub fn significance_level(&self) -> f64 {
//!         #[allow(clippy::clone_on_copy)]
//!         self.inner.significance_level.clone()
//!     }
//!     #[setter]
//!     pub fn set_significance_level(&mut self, significance_level: f64) {
//!         self.inner.significance_level = significance_level;
//!     }
//! }
//! ```
//!
//! ////////// Example: Generated `ci_tests_init.rs` File //////////
//!
//! ```rust
//! use pyo3::prelude::*;
//! pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
//!     m.add_class::<super::_ci_python::PyChiSquared>()?;
//!     m.add_class::<super::_ci_python::PyCressieRead>()?;
//!     ...
//!
//!     Ok(())
//! }
//! ```

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};
use syn::{
    visit::{self, Visit},
    Fields, File, ItemImpl, ItemStruct, Type, TypePath,
};

/// Visitor that traverses the AST and collects all structs implementing `CITest` (`citest_structs`)
/// all struct definitions (`struct_defs`).
struct CITestCollector {
    citest_structs: Vec<String>,
    struct_defs: HashMap<String, ItemStruct>,
}

impl<'ast> Visit<'ast> for CITestCollector {
    /// Collect all structs implementing `CITest`.
    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        if let Some((None, trait_path, _)) = &node.trait_ {
            let is_citest = trait_path
                .segments
                .last()
                .is_some_and(|s| s.ident == "CITest");

            if is_citest {
                if let Type::Path(TypePath { path, .. }) = node.self_ty.as_ref() {
                    if let Some(seg) = path.segments.last() {
                        self.citest_structs.push(seg.ident.to_string());
                    }
                }
            }
        }
        visit::visit_item_impl(self, node);
    }

    /// Collect all struct definitions.
    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        self.struct_defs
            .insert(node.ident.to_string(), node.clone());
        visit::visit_item_struct(self, node);
    }
}

/// Run the `CITestCollector` recursively on the specified directory.
fn parse_dir(dir: &Path, collector: &mut CITestCollector) {
    for entry in
        fs::read_dir(dir).unwrap_or_else(|e| panic!("Failed to read {}: {}", dir.display(), e))
    {
        let path = entry
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", dir.display(), e))
            .path();
        if path.is_dir() {
            parse_dir(&path, collector);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let src = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
            let file: File = syn::parse_file(&src)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e));
            visit::visit_file(collector, &file);
            println!("cargo::rerun-if-changed={}", path.display());
        }
    }
}

/// Generate the `TokenStream` for the specified struct `s`.
fn generate_pyo3_wrapper(s: &ItemStruct) -> TokenStream {
    let struct_name = &s.ident.to_string();
    let struct_ident = format_ident!("{}", struct_name);
    let py_ident = format_ident!("Py{}", struct_name);
    let py_name = syn::LitStr::new(struct_name, proc_macro2::Span::call_site());

    let Fields::Named(named) = &s.fields else {
        panic!(
            "Encountered unnamed field when processing `{struct_name}`. Unnamed fields aren't supported (yet).",
        )
    };

    let field_names: Vec<_> = named
        .named
        .iter()
        .map(|f| f.ident.as_ref().expect("Named fields always have idents."))
        .collect();
    let field_types = named.named.iter().map(|f| &f.ty);

    let getters_setters = named.named.iter().map(|f| {
        let fname = f.ident.as_ref().expect("Named fields always have idents.");
        let ftype = &f.ty;
        let setter_ident = format_ident!("set_{}", fname);
        quote! {
            #[getter]
            pub fn #fname(&self) -> #ftype {
                #[allow(clippy::clone_on_copy)]  // The object does not necessarily implement `Copy` and this is easier than case distinction.
                self.inner.#fname.clone()
            }
            #[setter]
            pub fn #setter_ident(&mut self, #fname: #ftype) {
                self.inner.#fname = #fname;
            }
        }
    });

    let constructor_args = field_names.iter().zip(field_types).map(|(n, t)| {
        quote! { #n: #t }
    });
    let constructor_init = field_names.iter().map(|n| quote! { #n });

    quote! {
        #[gen_stub_pyclass]
        #[pyclass(name = #py_name, module = "ci_python._ci_python")]
        pub struct #py_ident {
            inner: ::ci_core::ci_tests::#struct_ident,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl #py_ident {
            #[new]
            pub fn new(#(#constructor_args),*) -> Self {
                Self {
                    inner: ::ci_core::ci_tests::#struct_ident { #(#constructor_init),* },
                }
            }

            // Raises e.g. the following if passed by reference:
            // the trait `pyo3::impl_::extract_argument::PyFunctionArgument<'_, '_, '_, _>` is not implemented for `&numpy::PyReadonlyArray<'_, f64, numpy::ndarray::Dim<[usize; 2]>>`
            #[allow(clippy::needless_pass_by_value)]
            fn run_test(
                &self,
                py: Python<'_>,
                x_values: PyReadonlyArray1<'_, f64>,
                y_values: PyReadonlyArray1<'_, f64>,
                z: PyReadonlyArray2<'_, f64>,
            ) -> PyResult<Py<PyAny>> {
                test_result_to_pyobj(
                    &self.inner
                        .run_test(
                            x_values.as_array().to_owned(),
                            y_values.as_array().to_owned(),
                            z.as_array().to_owned(),
                        )
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?,
                    py
                )
            }

            #(#getters_setters)*
        }
    }
}

fn main() {
    let out_dir = PathBuf::from(
        env::var("OUT_DIR").unwrap_or_else(|e| panic!("Couldn't find output directory: {e}")),
    );

    // Generate ci_tests.rs.
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|e| panic!("Couldn't find directory of crate: {e}")),
    );
    let ci_core_src = manifest_dir.join("../ci-core/src");
    assert!(
        ci_core_src.exists(),
        "`ci-core/src` not found at {}",
        ci_core_src.display()
    );

    let mut collector = CITestCollector {
        citest_structs: Vec::new(),
        struct_defs: HashMap::new(),
    };
    parse_dir(&ci_core_src, &mut collector);

    let mut tokens = TokenStream::new();

    for name in &collector.citest_structs {
        if let Some(def) = collector.struct_defs.get(name) {
            tokens.extend(generate_pyo3_wrapper(def));
        } else {
            panic!("Struct `{name}` implements `CITest` but its definition was not found in `ci-core/src`.");
        }
    }

    fs::write(out_dir.join("ci_tests.rs"), tokens.to_string())
        .unwrap_or_else(|e| panic!("Couldn't save `ci_tests.rs`: {e}"));

    // Generate ci_tests_init.rs.
    let mut tokens_init = TokenStream::new();
    let py_class_idents: Vec<_> = collector
        .citest_structs
        .iter()
        .map(|n| format_ident!("Py{}", n))
        .collect();

    tokens_init.extend(quote! {
        use pyo3::prelude::*;

        pub fn init(
            m: &Bound<'_, PyModule>,
        ) -> PyResult<()> {
            #(m.add_class::<super::_ci_python::#py_class_idents>()?;)*
            Ok(())
        }
    });
    fs::write(out_dir.join("ci_tests_init.rs"), tokens_init.to_string())
        .unwrap_or_else(|e| panic!("Couldn't save `ci_tests_init.rs`: {e}"));
}
