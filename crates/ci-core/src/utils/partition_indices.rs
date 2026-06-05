use ndarray::{Array2, Axis};
use ordered_float::OrderedFloat;
use std::collections::HashMap;

/// Partition the rows of `data` (indexed as `data[row][col]`) into groups
/// that share the same combination of column values.
///
/// Returns `indices[partition][i]`, where each inner `Vec` holds the row
/// indices that belong to that partition. The order of partitions and the
/// order of indices within a partition are both unspecified.
#[must_use]
pub fn partition_indices(data: &Array2<f64>) -> Vec<Vec<usize>> {
    let mut groups: HashMap<Vec<OrderedFloat<f64>>, Vec<usize>> = HashMap::new();
    for (i, row) in data.axis_iter(Axis(0)).enumerate() {
        let key: Vec<OrderedFloat<f64>> = row.iter().map(|&v| OrderedFloat(v)).collect();
        groups.entry(key).or_default().push(i);
    }
    groups.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::HashSet;

    #[test]
    fn simple_grouping() {
        let data = array![[1.0, 10.0], [1.0, 10.0], [2.0, 30.0],];

        let result = partition_indices(&data);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_empty_array() {
        let data: Array2<f64> = Array2::zeros((0, 2));
        let result = partition_indices(&data);

        assert!(result.is_empty());
    }

    #[test]
    fn test_singleton_groups() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],];

        let result = partition_indices(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|g| g.len() == 1));
    }

    #[test]
    fn test_single_partition() {
        let data = array![[1.0, 10.0], [1.0, 10.0], [1.0, 10.0],];

        let result = partition_indices(&data);

        assert_eq!(result.len(), 1);
        let mut group = result[0].clone();
        group.sort_unstable();
        assert_eq!(group, vec![0, 1, 2]);
    }

    #[test]
    fn float_rounding() {
        let data = array![[1.0, 10.0], [1.000_000_000_1, 10.0],];

        let result = partition_indices(&data);

        //To test whether it rounds it up
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn multiple_columns() {
        let data = array![
            [1.0, 2.0, 10.0],
            [1.0, 2.0, 10.0],
            [1.0, 3.0, 30.0],
            [2.0, 2.0, 40.0],
        ];

        let result = partition_indices(&data);
        assert_eq!(result.len(), 3);

        // Find the group that has 2 rows (the [1.0, 2.0, 10.0] group)
        assert!(result.iter().any(|g| g.len() == 2));
    }

    #[test]
    fn order_dependence() {
        let data = array![[1.0, 2.0], [2.0, 1.0], [2.0, 1.0], [1.0, 2.0]];
        let expected: HashSet<Vec<usize>> = [vec![0, 3], vec![1, 2]].into_iter().collect();
        let result: HashSet<Vec<usize>> = partition_indices(&data).into_iter().collect();

        assert_eq!(expected, result);
    }

    #[test]
    fn zero_column_rows() {
        let data: Array2<f64> = Array2::zeros((3, 0));
        let result = partition_indices(&data);
        assert_eq!(result.len(), 1);
        let mut group = result[0].clone();
        group.sort_unstable();
        assert_eq!(group, vec![0, 1, 2]);
    }

    #[test]
    fn single_column_rows() {
        let data = array![[1.0], [2.0], [1.0]];
        let result = partition_indices(&data);
        assert_eq!(result.len(), 2);
        let result_set: HashSet<Vec<usize>> = result.into_iter().collect();
        let expected: HashSet<Vec<usize>> = [vec![0, 2], vec![1]].into_iter().collect();
        assert_eq!(result_set, expected);
    }

    #[test]
    fn negative_values() {
        let data = array![
            [-1.0, 2.0],
            [0.0, -0.0],
            [0.0, 0.0],
            [-1.0, 2.0],
            [-1.0, -2.0],
        ];

        let expected: HashSet<Vec<usize>> = [vec![0, 3], vec![1, 2], vec![4]].into_iter().collect();
        let result: HashSet<Vec<usize>> = partition_indices(&data).into_iter().collect();

        assert_eq!(expected, result);
    }
}
