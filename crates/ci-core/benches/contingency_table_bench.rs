#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
//otherwise we run into problems with clippy, but actual problems would only occur after 9 quadrillion

use ci_core::utils::contingency_table::contingency_table;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

fn generate_arrays(size: usize, num_categories: usize) -> (Array1<f64>, Array1<f64>) {
    let col1 = Array1::from_shape_fn(size, |i| (i % num_categories) as f64);
    //Multiply by an arbitrary prime to mix up the pairings
    let col2 = Array1::from_shape_fn(size, |i| ((i * 7) % num_categories) as f64);

    (col1, col2)
}

fn benchmark_contingency_table(c: &mut Criterion) {
    let (col1, col2) = generate_arrays(10_000, 50);

    let mut group = c.benchmark_group("Contingency Table (N=10k)");

    //call original function to see performance
    group.bench_function("contingency_original", |b| {
        b.iter(|| {
            let result = contingency_table(black_box(&col1), black_box(&col2));
            black_box(result)
        });
    });

    group.finish();
}

fn custom_criterion() -> Criterion {
    //give criterion a bit more time to run the function more times to get a more accurate benchmark
    Criterion::default().measurement_time(std::time::Duration::from_secs(10))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = benchmark_contingency_table
}
criterion_main!(benches);
