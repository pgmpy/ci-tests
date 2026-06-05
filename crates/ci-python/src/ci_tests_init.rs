//! Wrap the generated `ci_tests_init.rs` file so that the `init` function can
//! be included and called properly by `lib.rs`.

include!(concat!(env!("OUT_DIR"), "/ci_tests_init.rs"));
