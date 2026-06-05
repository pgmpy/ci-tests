use ndarray::{Array, Array1, Array2};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::hash::BuildHasher;

/// Count how often each pair `(col1[i], col2[i])` occurs and return the
/// counts as a 2D table. Rows and columns are ordered by value, so the
/// result does not depend on the order of the inputs.
#[must_use]
pub fn contingency_table(col1: &Array1<f64>, col2: &Array1<f64>) -> Array2<f64> {
    let col1_map = build_global_category_map(col1);
    let col2_map = build_global_category_map(col2);
    contingency_table_with_categories(col1, col2, &col1_map, &col2_map)
}

//Great replacement of the contingency_table_with_categories function in power divergence, but cannot
//be used when contingency_table_with_categories is needed within another function
#[must_use]
pub fn contingency_table_from_indices<S1: BuildHasher, S2: BuildHasher>(
    indices: &[usize],
    x_values: &Array1<f64>,
    y_values: &Array1<f64>,
    x_categories: &HashMap<OrderedFloat<f64>, usize, S1>,
    y_categories: &HashMap<OrderedFloat<f64>, usize, S2>,
) -> Array2<f64> {
    let mut table = Array2::<f64>::zeros((x_categories.len(), y_categories.len()));

    for &i in indices {
        let r = x_categories[&OrderedFloat(x_values[i])];
        let c = y_categories[&OrderedFloat(y_values[i])];

        table[[r, c]] += 1.0;
    }

    table
}

/// Number every distinct value in `arr` (smallest gets 0, next gets 1,
/// and so on). Pass the same map to several calls of
/// `contingency_table_with_categories` to get tables that share the
/// exact same rows and columns.
#[must_use]
pub fn build_global_category_map(arr: &Array1<f64>) -> HashMap<OrderedFloat<f64>, usize> {
    let mut map = HashMap::new();

    for &v in arr {
        let key = OrderedFloat(v);
        let next_index = map.len();
        map.entry(key).or_insert(next_index);
    }

    map
}

/// Like `contingency_table`, but the rows and columns are fixed by the
/// maps you pass in. Use this when several tables need the same shape,
/// even if some values are missing from a particular table.
#[must_use]
pub fn contingency_table_with_categories<S1: BuildHasher, S2: BuildHasher>(
    col1: &Array1<f64>,
    col2: &Array1<f64>,
    col1_map: &HashMap<OrderedFloat<f64>, usize, S1>,
    col2_map: &HashMap<OrderedFloat<f64>, usize, S2>,
) -> Array2<f64> {
    let mut result = Array::zeros((col1_map.len(), col2_map.len()));
    for i in 0..col1.len() {
        if let (Some(&r), Some(&c)) = (
            col1_map.get(&OrderedFloat(col1[i])),
            col2_map.get(&OrderedFloat(col2[i])),
        ) {
            result[[r, c]] += 1.;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn basic_test() {
        let test1_x: Array1<f64> = array![1.0, 2.0, 3.0, 1.0, 1.0];
        let test1_y: Array1<f64> = array![1.0, 2.0, 3.0, 1.0, 2.0];
        let test1_expected: Array2<f64> = array![[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(test1_expected, contingency_table(&test1_x, &test1_y));
    }

    #[test]
    fn single_value() {
        let test3_x = array![1.0, 1.0, 1.0];
        let test3_y = array![2.0, 2.0, 2.0];
        let test3_expected = array![[3.0]];
        assert_eq!(test3_expected, contingency_table(&test3_x, &test3_y));
    }
}
