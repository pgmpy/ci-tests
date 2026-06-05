//! Run this using `cargo run --bin stub_gen` to generate the stub files.

use ci_python::stub_info;
use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = stub_info()?;
    stub.generate()?;
    Ok(())
}
