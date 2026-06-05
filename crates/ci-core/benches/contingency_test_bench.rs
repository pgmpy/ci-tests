#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
//otherwise we run into problems with clippy, but actual problems would only occur after 9 quadrillion

use ci_core::utils::contingency_test::contingency_test;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use ndarray::Array2;

fn generate_matrix(n: usize, m: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, m), |(i, j)| (i + j + 1) as f64)
}

// Helper to avoid duplication
fn bench_lambda(c: &mut Criterion, name: &str, lambda: f64, observed: &Array2<f64>) {
    c.bench_function(name, |b| {
        b.iter_batched_ref(
            || observed.clone(),
            |data| {
                let result = contingency_test(black_box(data), black_box(lambda));
                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });
}

fn benchmark_contingency_test(c: &mut Criterion) {
    let observed = generate_matrix(100, 100);

    let cases = [
        ("contingency_lambda_0", 0.0),
        ("contingency_lambda_-1", -1.0),
        ("contingency_lambda_2", 2.0),
    ];

    //benchmark for all values of lambda
    for (name, lambda) in cases {
        bench_lambda(c, name, lambda, &observed);
    }
}

fn custom_criterion() -> Criterion {
    //gives criterion a bit more time to get a more accurate benchmark
    Criterion::default().measurement_time(std::time::Duration::from_secs(10))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = benchmark_contingency_test
}
criterion_main!(benches);
