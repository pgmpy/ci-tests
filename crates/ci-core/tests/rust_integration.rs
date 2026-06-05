use ci_core::ci_tests::*;
use ci_core::strategy::{CITest, TestResult};
use ndarray::{array, Array1, Array2};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn gen_normal(n: usize, mean: f64, std_dev: f64, rng: &mut SmallRng) -> Array1<f64> {
    let dist = Normal::new(mean, std_dev).unwrap();
    Array1::from_vec((0..n).map(|_| dist.sample(rng)).collect())
}

#[test]
fn test_all_discrete_tests_accept_independent_data() {
    let discrete_tests: Vec<Box<dyn CITest>> = vec![
        Box::new(ChiSquared::new(true, 0.05)),
        Box::new(FreemanTukey::new(true, 0.05)),
        Box::new(LogLikelihood::new(true, 0.05)),
        Box::new(ModifiedLikelihood::new(true, 0.05)),
        Box::new(CressieRead::new(true, 0.05)),
    ];

    let x = array![1., 1., 2., 2., 1., 1., 2., 2.];
    let y = array![1., 2., 1., 2., 1., 2., 1., 2.];
    let z = array![[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]];

    for (index, test) in discrete_tests.iter().enumerate() {
        let result = test
            .run_test(x.clone(), y.clone(), z.clone())
            .unwrap_or_else(|err| panic!("Discrete test at index {index} crashed: {err}"));

        assert!(
            matches!(result, TestResult::Boolean(true)),
            "Discrete test at index {index} failed to return true!"
        );
    }
}

#[test]
fn test_all_continuous_tests_accept_independent_data() {
    let mut rng = SmallRng::seed_from_u64(40);
    let n = 1000;

    let continuous_tests: Vec<Box<dyn CITest>> = vec![
        Box::new(PearsonCorrelation::new(true, 0.05)),
        Box::new(PearsonEquivalence::new(true, 0.05, 0.1)),
    ];

    let x = gen_normal(n, 0.0, 1.0, &mut rng);
    let y = gen_normal(n, 0.0, 1.0, &mut rng);
    let empty_z = Array2::<f64>::zeros((0, 0));

    for (index, test) in continuous_tests.iter().enumerate() {
        let result = test
            .run_test(x.clone(), y.clone(), empty_z.clone())
            .unwrap_or_else(|err| panic!("Continuous test at index {index} crashed: {err}"));

        assert!(
            matches!(result, TestResult::Boolean(true)),
            "Continuous test at index {index} failed to return true!"
        );
    }
}
