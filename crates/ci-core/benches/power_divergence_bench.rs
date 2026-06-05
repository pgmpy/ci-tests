#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
//otherwise we run into problems with clippy, but actual problems would only occur after 9 quadrillion

use ci_core::utils::power_divergence::power_divergence;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use ndarray::{Array1, Array2};

fn generate_categorical_vector(len: usize, categories: usize) -> Array1<f64> {
    Array1::from_shape_fn(len, |i| (i % categories) as f64)
}

fn generate_conditioning_matrix(rows: usize, cols: usize, categories: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i + j) % categories) as f64)
}

fn bench_power_divergence(
    c: &mut Criterion,
    name: &str,
    lambda: f64,
    x_values: &Array1<f64>,
    y_values: &Array1<f64>,
    z: &Array2<f64>,
) {
    c.bench_function(name, |b| {
        b.iter_batched_ref(
            || (x_values.clone(), y_values.clone(), z.clone()),
            |(x, y, z_data)| {
                let result = power_divergence(
                    black_box(x),
                    black_box(y),
                    black_box(z_data),
                    black_box(false),
                    black_box(0.05),
                    black_box(lambda),
                );

                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });
}

fn benchmark_power_divergence(c: &mut Criterion) {
    let n = 100;

    let x_values = generate_categorical_vector(n, 10);
    let y_values = generate_categorical_vector(n, 8);

    // Conditional case with 2 conditioning variables
    let z = generate_conditioning_matrix(n, 2, 5);

    let cases = [
        ("power_divergence_lambda_0", 0.0),
        ("power_divergence_lambda_-1", -1.0),
        ("power_divergence_lambda_2", 2.0),
    ];

    for (name, lambda) in cases {
        bench_power_divergence(c, name, lambda, &x_values, &y_values, &z);
    }
}

fn custom_criterion() -> Criterion {
    Criterion::default().measurement_time(std::time::Duration::from_secs(10))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = benchmark_power_divergence
}

criterion_main!(benches);
